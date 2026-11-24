"""Team-level features: efficiency ratings, pace, style rates, fantasy points allowed by
position (defence vs position), Elo ratings and rest days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from elfantasy.features.common import asof_join, ewm_by

TEAM_HALFLIFE = 8          # games
LEAGUE_DVP_HALFLIFE = 300  # team-games, for the league-average normaliser
POSITIONS = ("Guard", "Forward", "Center")

TEAM_STATE_COLS = [
    "ortg", "drtg", "net", "pace", "oreb_pct", "dreb_pct", "tov_forced_rate",
    "stl_rate", "pf_per40", "blk_rate", "fp_allowed", "fp_scored",
    *[f"dvp_{p}" for p in POSITIONS],
]


def team_game_metrics(team_games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with per-game rates (team and opponent side)."""
    tg = team_games.copy()
    tg["fga"] = tg["team_fga2"] + tg["team_fga3"]
    tg["poss"] = tg["fga"] + 0.44 * tg["team_fta"] - tg["team_oreb"] + tg["team_tov"]

    played = player_games[player_games["played"]]
    fp_team = played.groupby(["season", "gamecode", "team"])["fantasy_pts"].sum().rename("fp_scored")
    fp_pos = (
        played.groupby(["season", "gamecode", "team", "position_name"])["fantasy_pts"].sum()
        .unstack("position_name").reindex(columns=list(POSITIONS)).fillna(0.0)
        .add_prefix("fp_scored_")
    )
    tg = tg.merge(fp_team, on=["season", "gamecode", "team"], how="left")
    tg = tg.merge(fp_pos, on=["season", "gamecode", "team"], how="left")

    opp_cols = ["team", "team_pts", "poss", "team_oreb", "team_dreb", "team_tov", "team_fga2",
                "fp_scored", *[f"fp_scored_{p}" for p in POSITIONS]]
    opp = tg[["season", "gamecode", *opp_cols]].rename(columns={c: f"o_{c}" for c in opp_cols})
    m = tg.merge(opp, on=["season", "gamecode"])
    m = m[m["team"] != m["o_team"]].copy()

    per40 = 40.0 / (40.0 + 5.0 * m["n_ot"].fillna(0))
    m["pace"] = (m["poss"] + m["o_poss"]) / 2 * per40
    m["ortg"] = 100 * m["team_pts"] / m["poss"]
    m["drtg"] = 100 * m["o_team_pts"] / m["o_poss"]
    m["net"] = m["ortg"] - m["drtg"]
    m["oreb_pct"] = m["team_oreb"] / (m["team_oreb"] + m["o_team_dreb"])
    m["dreb_pct"] = m["team_dreb"] / (m["team_dreb"] + m["o_team_oreb"])
    m["tov_forced_rate"] = m["o_team_tov"] / m["o_poss"]
    m["stl_rate"] = m["team_stl"] / m["o_poss"]
    m["pf_per40"] = m["team_pf"] * per40
    m["blk_rate"] = m["team_blk"] / m["o_team_fga2"].replace(0, np.nan)
    m["fp_allowed"] = m["o_fp_scored"] * per40
    m["fp_scored"] = m["fp_scored"] * per40
    for p in POSITIONS:
        m[f"dvp_{p}"] = m[f"o_fp_scored_{p}"] * per40
    keep = ["season", "gamecode", "team", "date_utc", "margin", "n_ot", "coach_pts", *TEAM_STATE_COLS]
    return m[keep].sort_values(["team", "date_utc"]).reset_index(drop=True)


def team_state(metrics: pd.DataFrame) -> pd.DataFrame:
    """Post-game EWMA state per team (includes the game on that row)."""
    st = metrics[["team", "date_utc", "season"]].copy()
    for c in TEAM_STATE_COLS:
        st[c] = ewm_by(metrics, "team", c, TEAM_HALFLIFE)
    st["games_season"] = metrics.groupby(["team", "season"]).cumcount() + 1
    st["coach_pts_season_mean"] = metrics.groupby(["team", "season"])["coach_pts"].transform(
        lambda s: s.expanding().mean())
    st = st.rename(columns={"season": "state_season"})
    return st


def league_dvp_state(metrics: pd.DataFrame) -> pd.DataFrame:
    s = metrics.sort_values("date_utc")[["date_utc", *[f"dvp_{p}" for p in POSITIONS]]].copy()
    for p in POSITIONS:
        s[f"lg_dvp_{p}"] = s[f"dvp_{p}"].ewm(halflife=LEAGUE_DVP_HALFLIFE, min_periods=1).mean()
    return s[["date_utc", *[f"lg_dvp_{p}" for p in POSITIONS]]]


def team_features(queries: pd.DataFrame, state: pd.DataFrame, side: str) -> pd.DataFrame:
    """As-of team state (from `team_state`) for `queries[side]` (side = 'team' or 'opponent')."""
    q = queries.rename(columns={side: "team_key"})[["team_key", "date_utc", "season"]]
    st = state.rename(columns={"team": "team_key"})
    out = asof_join(q, st, by="team_key")
    new_season = out["state_season"] != q["season"].to_numpy()
    out.loc[new_season, ["games_season", "coach_pts_season_mean"]] = np.nan
    out["games_season"] = out["games_season"].fillna(0)
    out = out.drop(columns=["state_season"])
    prefix = "team_" if side == "team" else "opp_"
    return out.add_prefix(prefix)


# ----------------------------------------------------------------------------- Elo

ELO_K = 20.0
ELO_HOME = 60.0          # Elo points of home-court advantage
ELO_REGRESS = 1 / 3      # pull toward 1500 between seasons
ELO_PER_POINT = 28.0     # Elo points per point of expected margin


def elo_table(games: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo for both teams of every game (margin-aware updates, FiveThirtyEight-style).

    Played games update the ratings; games not yet played get pre-game rows from the current
    ratings without updating them, so upcoming rounds have Elo features too."""
    g = games.dropna(subset=["date_utc"]).sort_values("date_utc")
    rating: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []
    for r in g.itertuples(index=False):
        for t in (r.home_team, r.away_team):
            if t not in rating:
                rating[t] = 1500.0
            elif last_season.get(t) != r.season:
                rating[t] = 1500.0 + (1 - ELO_REGRESS) * (rating[t] - 1500.0)
            last_season[t] = r.season
        h, a = rating[r.home_team], rating[r.away_team]
        hca = 0.0 if r.neutral_venue else ELO_HOME
        diff = h + hca - a
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        rows.append((r.season, r.gamecode, r.home_team, h, a, diff, p_home))
        rows.append((r.season, r.gamecode, r.away_team, a, h, -diff, 1 - p_home))
        if not (r.played and pd.notna(r.home_score) and pd.notna(r.away_score)):
            continue
        margin = r.home_score - r.away_score
        winner_diff = diff if margin > 0 else -diff
        mult = np.log(abs(margin) + 1) * (2.2 / (winner_diff * 0.001 + 2.2))
        delta = ELO_K * mult * ((1.0 if margin > 0 else 0.0) - p_home)
        rating[r.home_team] = h + delta
        rating[r.away_team] = a - delta
    df = pd.DataFrame(rows, columns=["season", "gamecode", "team", "elo", "opp_elo", "elo_diff", "elo_p_win"])
    df["elo_exp_margin"] = df["elo_diff"] / ELO_PER_POINT
    return df


# ------------------------------------------------------------------------ rest days

def rest_table(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous EuroLeague game, from the full schedule (works for
    future games too)."""
    g = games.dropna(subset=["date_utc"])
    # Drop past games that were never played (cancelled/voided): they are not real rest breaks.
    last_played = g.loc[g["played"], "date_utc"].max()
    g = g[g["played"] | (g["date_utc"] > last_played)]
    long = pd.concat([
        g[["season", "gamecode", "date_utc", "home_team"]].rename(columns={"home_team": "team"}),
        g[["season", "gamecode", "date_utc", "away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values(["team", "date_utc"])
    prev = long.groupby("team")["date_utc"].shift(1)
    long["rest_days"] = ((long["date_utc"] - prev).dt.total_seconds() / 86400).clip(upper=15)
    long["rest_days"] = long["rest_days"].fillna(15)
    long["short_rest"] = long["rest_days"] <= 2.5   # second game of a double-round week
    return long[["season", "gamecode", "team", "rest_days", "short_rest"]]
