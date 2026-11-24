"""Player-level features: form, minutes/role, per-36 production rates, shooting luck,
previous-season priors, availability streaks and teammates' vacated minutes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from elfantasy.features.common import asof_join, ewm_by, roll_by

RATE_STATS = ["pts", "oreb", "dreb", "ast", "stl", "blk", "tov", "fd", "pf",
              "missed_fg", "missed_ft", "blk_against"]
SEASON_COLS = ["gp_season", "fp_season_mean", "pir_season_mean", "pir_season_std",
               "min_season_mean", "fg_pct_l5_vs_season", "fg3_pct_l5_vs_season"]


def _prep(pg: pd.DataFrame) -> pd.DataFrame:
    p = pg[pg["played"]].sort_values(["player_code", "date_utc"]).copy()
    p["fga"] = p["fga2"] + p["fga3"]
    p["fgm"] = p["fgm2"] + p["fgm3"]
    p["missed_fg"] = p["fga"] - p["fgm"]
    p["missed_ft"] = p["fta"] - p["ftm"]
    p["usage"] = p["fga"] + 0.44 * p["fta"] + p["tov"]
    p["starter_f"] = p["is_starter"].astype(float)
    p["pm_f"] = pd.to_numeric(p["plus_minus"], errors="coerce").fillna(0.0)
    return p.reset_index(drop=True)


def player_state(pg: pd.DataFrame) -> pd.DataFrame:
    """Post-game state per player after each game he played (includes that game)."""
    p = _prep(pg)
    g, gs = "player_code", ["player_code", "season"]
    st = p[["player_code", "date_utc", "season", "team", "position_name"]].copy()

    for n in (1, 3, 5, 10):
        st[f"fp_l{n}"] = roll_by(p, g, "fantasy_pts", n)
    for n in (3, 5, 10):
        st[f"pir_l{n}"] = roll_by(p, g, "pir", n)
    st["pir_ewm_h3"] = ewm_by(p, g, "pir", 3)
    st["pir_ewm_h8"] = ewm_by(p, g, "pir", 8)
    st["fp_ewm_h5"] = ewm_by(p, g, "fantasy_pts", 5)
    st["min_l3"] = roll_by(p, g, "minutes", 3)
    st["min_l10"] = roll_by(p, g, "minutes", 10)
    st["min_ewm_h5"] = ewm_by(p, g, "minutes", 5)
    st["min_trend"] = st["min_l3"] - st["min_l10"]
    st["start_rate_l5"] = roll_by(p, g, "starter_f", 5)
    st["pm_ewm_h8"] = ewm_by(p, g, "pm_f", 8)

    min8 = ewm_by(p, g, "minutes", 8).clip(lower=1.0)
    for c in RATE_STATS:
        st[f"rate_{c}_p36"] = ewm_by(p, g, c, 8) / min8 * 36
    st["usage_p36"] = ewm_by(p, g, "usage", 8) / min8 * 36
    st["pir_p36_ewm"] = ewm_by(p, g, "pir", 5) / ewm_by(p, g, "minutes", 5).clip(lower=1.0) * 36

    def pct(m, a):
        return m / a.replace(0, np.nan)

    fgm5, fga5 = roll_by(p, g, "fgm", 5, "sum"), roll_by(p, g, "fga", 5, "sum")
    f3m5, f3a5 = roll_by(p, g, "fgm3", 5, "sum"), roll_by(p, g, "fga3", 5, "sum")
    cs = p.groupby(gs)[["fgm", "fga", "fgm3", "fga3"]].cumsum()
    st["fg_pct_l5_vs_season"] = pct(fgm5, fga5) - pct(cs["fgm"], cs["fga"])
    st["fg3_pct_l5_vs_season"] = pct(f3m5, f3a5) - pct(cs["fgm3"], cs["fga3"])

    st["gp_season"] = p.groupby(gs).cumcount() + 1
    exp = p.groupby(gs)
    st["fp_season_mean"] = exp["fantasy_pts"].transform(lambda s: s.expanding().mean())
    st["pir_season_mean"] = exp["pir"].transform(lambda s: s.expanding().mean())
    st["pir_season_std"] = exp["pir"].transform(lambda s: s.expanding().std())
    st["min_season_mean"] = exp["minutes"].transform(lambda s: s.expanding().mean())
    st["career_games"] = p.groupby(g).cumcount() + 1
    st["last_game_date"] = p["date_utc"]
    return st.rename(columns={"season": "state_season", "team": "state_team",
                              "position_name": "state_position"})


def prev_season_table(pg: pd.DataFrame) -> pd.DataFrame:
    p = pg[pg["played"]]
    agg = p.groupby(["player_code", "season"]).agg(
        prev_fp_mean=("fantasy_pts", "mean"),
        prev_pir_mean=("pir", "mean"),
        prev_gp=("pir", "size"),
        prev_min_mean=("minutes", "mean"),
    ).reset_index()
    agg["season"] = agg["season"] + 1   # joins onto the following season
    return agg


def player_features(queries: pd.DataFrame, state: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    out = asof_join(queries, state, by="player_code")
    new_season = (out["state_season"] != queries["season"].to_numpy()) | out["state_season"].isna()
    out.loc[new_season, SEASON_COLS] = np.nan
    out["gp_season"] = out["gp_season"].fillna(0)
    out["career_games"] = out["career_games"].fillna(0)
    last = pd.to_datetime(out["last_game_date"], utc=True)
    out["player_rest_days"] = ((queries["date_utc"] - last).dt.total_seconds() / 86400).clip(upper=400)
    out["changed_team"] = out["state_team"].notna() & (out["state_team"] != queries["team"].to_numpy())
    pv = queries[["player_code", "season"]].merge(prev, on=["player_code", "season"], how="left")
    pv.index = queries.index
    out = pd.concat([out, pv.drop(columns=["player_code", "season"])], axis=1)
    return out.drop(columns=["state_season", "state_team", "state_position", "last_game_date"])


# ----------------------------------------------------------------- team game index

def team_game_index(games: pd.DataFrame) -> pd.DataFrame:
    """Sequential index of each team's PLAYED games within a season.

    Only played games count: cancelled games (e.g. 2019-20 after R28, voided 2021-22 games) and
    games not played yet must not appear as games a player "missed"."""
    g = games[games["played"]].dropna(subset=["date_utc"])
    cols = ["season", "gamecode", "date_utc", "round", "phase", "home_team", "away_team"]
    long = pd.concat([
        g[cols].assign(team=g["home_team"], opponent=g["away_team"], is_home=True),
        g[cols].assign(team=g["away_team"], opponent=g["home_team"], is_home=False),
    ]).drop(columns=["home_team", "away_team"])
    long = long.sort_values(["season", "team", "date_utc", "gamecode"]).reset_index(drop=True)
    long["idx"] = long.groupby(["season", "team"]).cumcount()
    return long


def availability_features(queries: pd.DataFrame, pg: pd.DataFrame, tgi: pd.DataFrame) -> pd.DataFrame:
    """missed_streak (team games missed since his last appearance), dnp_rate_season and
    first_game_back, counting only the team's PLAYED games before the query date. Works the
    same for past games, the next round, or two rounds ahead."""
    key = ["season", "gamecode", "team"]
    grp = ["player_code", "season", "team"]
    played = pg[pg["played"]][["player_code", *key]].merge(tgi[[*key, "idx"]], on=key, how="inner")
    played = played.sort_values("idx")
    played["n_played"] = played.groupby(grp).cumcount() + 1
    first = played.groupby(grp)["idx"].min().rename("first_idx").reset_index()

    q = queries[["player_code", "season", "team", "date_utc"]].copy()
    q["_row"] = np.arange(len(q))
    q["date_utc"] = pd.to_datetime(q["date_utc"], utc=True)
    # k = number of team games played before the query date (= index the query game would get)
    team_games = tgi[["season", "team", "date_utc", "idx"]].sort_values("date_utc")
    m = pd.merge_asof(q.sort_values("date_utc"), team_games, on="date_utc", by=["season", "team"],
                      direction="backward", allow_exact_matches=False)
    m["k"] = m["idx"].fillna(-1).astype(int) + 1
    m = m.drop(columns="idx")
    # his last played game with index < k
    ps = played[[*grp, "idx", "n_played"]].rename(columns={"idx": "last_idx"})
    ps["k"] = ps["last_idx"] + 1
    m = pd.merge_asof(m.sort_values("k"), ps.sort_values("k"), on="k", by=grp,
                      direction="backward", allow_exact_matches=True)
    m = m.merge(first, on=grp, how="left")
    m["missed_streak"] = m["k"] - m["last_idx"] - 1
    span = m["k"] - m["first_idx"]
    m["dnp_rate_season"] = np.where((span > 0) & m["last_idx"].notna(), 1 - m["n_played"] / span, np.nan)
    m["first_game_back"] = (m["missed_streak"] >= 1).astype(float).where(m["missed_streak"].notna())
    m["team_game_idx"] = m["k"]
    out = m.sort_values("_row")[["missed_streak", "dnp_rate_season", "first_game_back", "team_game_idx"]]
    out.index = queries.index
    return out


def vacated_minutes(pg: pd.DataFrame, tgi: pd.DataFrame, state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Players who played their team's previous game but not this one (injury/rest).

    Returns (team totals per game, the individual missing players). The individual rows let
    the feature builder subtract a player's OWN minutes from his row, so "minutes of missing
    teammates" never contains the player himself (that would leak whether he plays)."""
    key = ["season", "gamecode", "team"]
    played = pg[pg["played"]][["player_code", "position_name", *key]].merge(tgi[[*key, "idx"]], on=key)
    exp = played[["player_code", "position_name", "season", "team", "idx"]].copy()
    exp["idx"] = exp["idx"] + 1
    exp = exp.merge(tgi[["season", "team", "idx", "gamecode", "date_utc"]], on=["season", "team", "idx"])
    now = played[["player_code", "season", "team", "idx"]].assign(played_now=True)
    exp = exp.merge(now, on=["player_code", "season", "team", "idx"], how="left")
    missing = exp[exp["played_now"].isna()].reset_index(drop=True)
    cols = [*key, "vacated_min", "vacated_min_Guard", "vacated_min_Forward", "vacated_min_Center"]
    if missing.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=[*key, "player_code", "position_name", "min_ewm_h5"])
    mins = asof_join(missing, state[["player_code", "date_utc", "min_ewm_h5"]], by="player_code")
    missing["min_ewm_h5"] = mins["min_ewm_h5"].fillna(0).to_numpy()
    tot = missing.groupby(key)["min_ewm_h5"].sum().rename("vacated_min")
    pos = (missing.groupby([*key, "position_name"])["min_ewm_h5"].sum()
           .unstack("position_name").add_prefix("vacated_min_"))
    detail = missing[[*key, "player_code", "position_name", "min_ewm_h5"]]
    return pd.concat([tot, pos], axis=1).reset_index(), detail
