"""Weekly recommendation for an upcoming round.

    from elfantasy.weekly import recommend
    rec = recommend()                       # next round, 100 credits
    rec.squad_table(); rec.coach_table(); rec.rankings("G"); rec.captain_shortlist()
    rec.save()                              # markdown under recommendations/

    python -m elfantasy.weekly --matchday 15 --out "Player Name" --budget 101.5

Pipeline: the app's line-ups for the round (price, fantasy position, P(plays)) -> features as
of now -> player, availability and game models trained on all history -> expected fantasy
points -> best squad under the rules. Availability: your overrides > the app's P(plays) >
our availability model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from elfantasy.backtest.walkforward import build_candidates
from elfantasy.data.build import build_euroleague, current_season_year
from elfantasy.data.fantasy_app import FantasyAppAPI, parse_lineups, parse_schedule
from elfantasy.data.fantasy_build import build_fantasy
from elfantasy.data.http import HttpClient
from elfantasy.data.snapshot import find_upcoming_matchday
from elfantasy.features.build import FeatureContext
from elfantasy.features.catalog import FEATURE_DESCRIPTIONS
from elfantasy.features.common import Store, asof_join
from elfantasy.models import baselines as B
from elfantasy.models.availability import AvailabilityModel
from elfantasy.models.game import GameModel, game_features, game_frame
from elfantasy.models.player_points import FEATURES, PlayerPointsModel, _matrix
from elfantasy.optimize.squad import SquadResult, best_squad
from elfantasy.paths import PROCESSED_DIR, REPO_ROOT
from elfantasy.rules import default_rules

log = logging.getLogger("elfantasy.weekly")

OVERRIDE_P = {"OUT": 0.0, "DOUBTFUL": 0.30, "PROBABLE": 0.75}
RETURNING_P = 0.85   # app says he plays but he missed his team's last game (injury return / new signing)
NO_HISTORY_P = 0.25  # no EuroLeague box score this season: usually a 12th-15th man who rarely plays
OVERRIDE_VALUES = ("OUT", "DOUBTFUL", "PROBABLE", "FIT")
TRADE_EPSILON = 1.0  # a trade must gain at least this many expected points (design §7)
POS_SHORT = {"Guard": "G", "Forward": "F", "Center": "C"}
POSITIONS = ("Guard", "Forward", "Center")


def refresh_data(euroleague: bool = True, fantasy: bool = True) -> None:
    """Fetch new box scores (cached history is reused) and the current player list / id maps."""
    if euroleague:
        build_euroleague()
    if fantasy:
        build_fantasy(with_profiles=False, with_match_logs=False)


# ----------------------------------------------------------------------------- result

@dataclass
class Recommendation:
    matchday: int
    deadline: pd.Timestamp | None
    budget: float
    players: pd.DataFrame          # every candidate, indexed by app player_id
    coaches: pd.DataFrame          # every coach, indexed by app player_id
    squad: SquadResult
    explanations: dict = field(default_factory=dict)
    trades: dict | None = None
    notes: list[str] = field(default_factory=list)
    my_squad: SquadResult | None = None   # roster mode: your squad after the suggested trades

    _COLS = ["name", "team", "opp", "pos", "price", "exp_pts", "pts_if_plays", "p_play", "real_lineup"]

    def squad_table(self, squad: SquadResult | None = None) -> pd.DataFrame:
        sq = (squad or self.squad).players.copy()
        order = {"starter": 0, "sixth": 1, "bench": 2}
        sq["order"] = sq["role"].map(order)
        sq = sq.sort_values(["order", "exp_pts"], ascending=[True, False])
        sq["role"] = np.where(sq["captain"], "starter (C)", sq["role"])
        return sq[["role"] + self._COLS].round(2).reset_index(drop=True)

    def coach_pick(self, squad: SquadResult | None = None) -> pd.Series:
        return (squad or self.squad).coach[["name", "team", "opp", "price", "exp_pts", "p_win"]]

    def my_squad_table(self) -> pd.DataFrame | None:
        """Roster mode: your squad after the suggested trades (line-up and captain)."""
        return self.squad_table(self.my_squad) if self.my_squad is not None else None

    def coach_table(self) -> pd.DataFrame:
        return (self.coaches.sort_values("exp_pts", ascending=False)
                [["name", "team", "opp", "home", "p_win", "exp_pts", "price"]].round(2).reset_index(drop=True))

    def rankings(self, pos: str | None = None, n: int = 10, by: str = "exp_pts") -> pd.DataFrame:
        p = self.players if pos is None else self.players[self.players["pos"] == pos]
        p = p.assign(pts_per_credit=p["exp_pts"] / p["price"])
        return p.sort_values(by, ascending=False).head(n)[self._COLS + ["pts_per_credit"]].round(2).reset_index(drop=True)

    def value_picks(self, pos: str | None = None, n: int = 10, min_exp: float = 8.0) -> pd.DataFrame:
        p = self.players[self.players["exp_pts"] >= min_exp]
        if pos:
            p = p[p["pos"] == pos]
        p = p.assign(pts_per_credit=p["exp_pts"] / p["price"])
        return p.sort_values("pts_per_credit", ascending=False).head(n)[self._COLS + ["pts_per_credit"]].round(2).reset_index(drop=True)

    def captain_shortlist(self, n: int = 5) -> pd.DataFrame:
        p = self.players[self.players["p_play"] >= 0.9]
        return p.sort_values("exp_pts", ascending=False).head(n)[self._COLS].round(2).reset_index(drop=True)

    def why(self, name_or_id) -> pd.DataFrame:
        pid = name_or_id if name_or_id in self.explanations else _resolve(self.players, name_or_id)
        return pd.DataFrame(self.explanations.get(pid, []), columns=["feature", "value", "effect_pts", "meaning"])

    # ------------------------------------------------------------------ export
    def to_markdown(self, banner: str | None = None) -> str:
        def md(df: pd.DataFrame) -> str:
            return df.to_markdown(index=False, floatfmt=".2f")

        sq = self.squad_table()
        c = self.coach_pick()
        lines = [
            f"# Round {self.matchday} recommendation" + (" (DRY RUN)" if banner else ""),
            "",
            *([f"> {banner}", ""] if banner else []),
            f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Deadline (first tip-off): "
            f"{self.deadline:%a %d %b %H:%M} UTC. Budget {self.budget:.1f} credits, squad cost {self.squad.cost:.1f}.",
            "",
            f"Expected squad points: **{self.squad.objective:.1f}** (captain doubled, bench halved, coach included).",
            "",
            "## Squad",
            "",
            md(sq),
            "",
            f"**Coach:** {c['name']} ({c['team']} vs {c['opp']}), {c['price']:.1f} cr, "
            f"expected {c['exp_pts']:.1f} pts, win probability {c['p_win']:.0%}.",
            "",
        ]
        if self.trades:
            lines += ["## Trades for your roster", "",
                      f"Out: {', '.join(self.trades['out']) or 'none'}  ",
                      f"In: {', '.join(self.trades['in']) or 'none'}  ",
                      f"Expected gain: {self.trades['gain']:+.1f} pts (a trade must gain ≥ {TRADE_EPSILON:.0f} pt)", ""]
            if self.my_squad is not None:
                mc = self.coach_pick(self.my_squad)
                lines += ["### Your squad after these trades", "", md(self.my_squad_table()), "",
                          f"Coach: {mc['name']} ({mc['team']}), expected {mc['exp_pts']:.1f}. "
                          f"Squad expected {self.my_squad.objective:.1f} pts, cost {self.my_squad.cost:.1f}.", ""]
        lines += ["## Captain shortlist", "", md(self.captain_shortlist()), ""]
        for pos in ("G", "F", "C"):
            lines += [f"## Top {pos}", "", md(self.rankings(pos, n=8)), ""]
        lines += ["## Value picks (points per credit, expected ≥ 8)", "", md(self.value_picks(n=10)), "",
                  "## Coaches", "", md(self.coach_table()), "", "## Why these picks", ""]
        for pid, row in self.squad.players.iterrows():
            ex = self.explanations.get(pid, [])
            parts = [f"{(e[3] or e[0]).lower()} = {_fmt(e[1])} → {e[2]:+.1f}" for e in ex[:4]]
            text = "; ".join(parts) if parts else "no EuroLeague history this season (debut average)"
            lines.append(f"- **{row['name']}** ({row['exp_pts']:.1f} expected): {text}")
        lines += ["", "Effects are each feature's contribution (in fantasy points) to the "
                  "points-if-he-plays prediction, relative to the average player.", ""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    @staticmethod
    def _squad_record(squad: SquadResult) -> dict:
        return {
            "expected_total": float(squad.objective),
            "cost": float(squad.cost),
            "squad": [
                {"player_id": int(pid), "name": r["name"], "team": r["team"], "pos": r["pos"],
                 "price": float(r["price"]), "role": r["role"], "captain": bool(r["captain"]),
                 "exp_pts": float(r["exp_pts"]), "p_play": float(r["p_play"])}
                for pid, r in squad.players.iterrows()
            ],
            "coach": {"player_id": int(squad.coach.name), "name": squad.coach["name"], "team": squad.coach["team"],
                      "price": float(squad.coach["price"]), "exp_pts": float(squad.coach["exp_pts"])},
        }

    def to_record(self) -> dict:
        """Machine-readable summary, used by `python -m elfantasy.evaluate` after the round."""
        sq = self.squad.players
        c = self.squad.coach
        return {
            "my_squad": self._squad_record(self.my_squad) if self.my_squad is not None else None,
            "season": current_season_year(),
            "matchday": int(self.matchday),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "deadline_utc": self.deadline.isoformat() if self.deadline is not None else None,
            "budget": float(self.budget),
            "cost": float(self.squad.cost),
            "expected_total": float(self.squad.objective),
            "squad": [
                {"player_id": int(pid), "name": r["name"], "team": r["team"], "pos": r["pos"],
                 "price": float(r["price"]), "role": r["role"], "captain": bool(r["captain"]),
                 "exp_pts": float(r["exp_pts"]), "p_play": float(r["p_play"])}
                for pid, r in sq.iterrows()
            ],
            "coach": {"player_id": int(c.name), "name": c["name"], "team": c["team"],
                      "price": float(c["price"]), "exp_pts": float(c["exp_pts"])},
            "notes": self.notes,
            "trades": ({k: v for k, v in self.trades.items() if k != "squad"} if self.trades else None),
        }

    def save(self, folder: Path | None = None, suffix: str = "", banner: str | None = None) -> Path:
        """Write the markdown report and a JSON record side by side; returns the .md path.
        `banner` marks a dry run (shown at the top of the report)."""
        year = current_season_year()
        folder = folder or REPO_ROOT / "recommendations" / f"{year}-{str(year + 1)[-2:]}"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"md{self.matchday:02d}_{datetime.now(timezone.utc):%Y%m%dT%H%MZ}{suffix}"
        path = folder / f"{stem}.md"
        path.write_text(self.to_markdown(banner), encoding="utf-8")
        record = self.to_record() | ({"dry_run": banner} if banner else {})
        (folder / f"{stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------- helpers

def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.0f}" if abs(f) >= 100 else f"{f:.2f}" if abs(f) < 1 else f"{f:.1f}"


def _resolve(pool: pd.DataFrame, key) -> int:
    """App player id from an id or a (part of a) name."""
    if isinstance(key, (int, np.integer)) and key in pool.index:
        return int(key)
    k = str(key).strip().lower()
    exact = pool.index[pool["name"].str.lower() == k]
    if len(exact) == 1:
        return int(exact[0])
    part = pool.index[pool["name"].str.lower().str.contains(k, regex=False)]
    if len(part) == 1:
        return int(part[0])
    options = ", ".join(pool.loc[part, "name"].head(8)) if len(part) else "no match"
    raise KeyError(f"'{key}' is ambiguous or unknown ({options})")


def _latest_positions(pg: pd.DataFrame) -> pd.Series:
    p = pg.dropna(subset=["position_name"]).sort_values("date_utc")
    return p.groupby("player_code")["position_name"].last()


def _moved_matches(app: FantasyAppAPI, cfg: dict, md: dict, matches: pd.DataFrame) -> set:
    """Matches moved outside their round, per the official rule: played after the next round
    has started or before the previous round has ended. Their players and coaches get their
    season average (captain not doubled)."""
    by_number = {m["number"]: m for m in cfg["matchdays"]}

    def starts(n: int) -> pd.Series:
        if n not in by_number:
            return pd.Series(dtype="datetime64[ns, UTC]")
        return parse_schedule(app.schedule_matchday(cfg["current_schedule_id"], by_number[n]["id"]))["started_at"]

    nxt, prv = starts(md["number"] + 1), starts(md["number"] - 1)
    moved = pd.Series(False, index=matches.index)
    if len(nxt):
        moved |= matches["started_at"] > nxt.min()
    if len(prv):
        moved |= matches["started_at"] < prv.max()
    return set(matches.loc[moved, "match_id"])


def _store_as_of(store: Store, as_of: pd.Timestamp) -> Store:
    """The data as it was at `as_of`: later games become unplayed and their box scores vanish."""
    as_of = as_of.tz_convert("UTC") if as_of.tzinfo else as_of.tz_localize("UTC")
    games = store.games.copy()
    later = games["date_utc"] >= as_of
    games.loc[later, "played"] = False
    games.loc[later, ["home_score", "away_score"]] = np.nan
    pg = store.player_games[store.player_games["date_utc"] < as_of].reset_index(drop=True)
    tg = store.team_games[store.team_games["date_utc"] < as_of].reset_index(drop=True)
    return Store(games=games, player_games=pg, team_games=tg, people=store.people)


def _last_game_players(ctx: FeatureContext, season: int) -> pd.DataFrame:
    """Players who played each team's most recent played game this season."""
    pg = ctx.store.player_games
    cur = pg[(pg["season"] == season) & pg["played"]]
    last = cur.groupby("team")["date_utc"].transform("max")
    return cur[cur["date_utc"] == last][["team", "player_code", "position_name"]]


# ------------------------------------------------------------------------------ main

def recommend(
    matchday: int | None = None,
    budget: float | None = None,
    overrides: dict | None = None,
    locked: list | tuple = (),
    excluded: list | tuple = (),
    roster: list | None = None,
    bank: float = 0.0,
    max_trades: int = 4,
    refresh: bool = False,
    snapshot: str | Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> Recommendation:
    """`snapshot`: replay a past round from a saved pre-deadline snapshot (its line-ups and
    play probabilities instead of live ones). `as_of`: use only box scores from before this
    time (defaults to the snapshot's deadline), so the replay sees exactly what was known."""
    rules = default_rules()
    budget = rules.initial_budget if budget is None else budget
    if refresh:
        refresh_data()
    notes: list[str] = []
    year = current_season_year()

    # ---- the round, from the app (live) or from a saved snapshot (replay)
    app = FantasyAppAPI(HttpClient(min_interval=0.3))
    cfg = app.config()
    snap = None
    if snapshot is not None:
        with gzip.open(snapshot, "rt", encoding="utf-8") as fh:
            snap = json.load(fh)
        if matchday is not None and int(matchday) != int(snap["matchday"]):
            raise ValueError(f"snapshot is for round {snap['matchday']}, not {matchday}")
        md = next(m for m in cfg["matchdays"] if m["number"] == int(snap["matchday"]))
        sched = snap["schedule"]
        taken = pd.Timestamp(snap["snapshot_utc"])
        notes.append(f"Replay of round {md['number']} from the snapshot taken {taken:%a %d %b %H:%M} UTC "
                     f"({snap.get('minutes_to_deadline', float('nan')):.0f} min before the deadline)")
    elif matchday is None:
        md, sched = find_upcoming_matchday(app, cfg)
    else:
        md = next((m for m in cfg["matchdays"] if m["number"] == matchday), None)
        if md is None:
            nums = [m["number"] for m in cfg["matchdays"]]
            raise ValueError(f"Round {matchday} does not exist; valid rounds are {min(nums)} to {max(nums)}")
        sched = app.schedule_matchday(cfg["current_schedule_id"], md["id"])
    matches = parse_schedule(sched)
    moved_ids = _moved_matches(app, cfg, md, matches)
    moved_teams = set(matches.loc[matches["match_id"].isin(moved_ids), ["home_abbr", "away_abbr"]].to_numpy().ravel())
    if snap is not None:
        lineups = pd.concat([parse_lineups(snap["lineups"][str(m)]) for m in matches["match_id"]
                             if str(m) in snap["lineups"]], ignore_index=True)
    else:
        lineups = pd.concat([parse_lineups(app.match_lineups(int(m))) for m in matches["match_id"]], ignore_index=True)
    lineups = lineups.drop_duplicates("player_id").set_index("player_id")
    lineups["name"] = (lineups["first_name"].fillna("") + " " + lineups["last_name"].fillna("")).str.strip()
    deadline = matches["started_at"].min()

    # ---- maps to EuroLeague codes and this round's EL games
    mp = pd.read_parquet(PROCESSED_DIR / "map_player.parquet").set_index("player_id")["person_code"]
    mt = pd.read_parquet(PROCESSED_DIR / "map_team.parquet").set_index("app_abbr")["el_code"]
    store = Store.load()
    if snap is not None and as_of is None:
        as_of = deadline
    if as_of is not None:
        store = _store_as_of(store, pd.Timestamp(as_of))
        notes.append(f"Only box scores from before {pd.Timestamp(as_of):%Y-%m-%d %H:%M} UTC are used")
    ctx = FeatureContext.from_store(store)
    g = store.games[(store.games["season"] == year) & (store.games["round"] == md["number"])]
    m = matches.assign(home_el=matches["home_abbr"].map(mt), away_el=matches["away_abbr"].map(mt))
    m = m.merge(g[["home_team", "away_team", "gamecode", "date_utc", "phase"]],
                left_on=["home_el", "away_el"], right_on=["home_team", "away_team"], how="left")
    if m["gamecode"].isna().any():
        notes.append(f"{int(m['gamecode'].isna().sum())} app matches not found in the EuroLeague schedule")
    side = pd.concat([
        m[["match_id", "home_abbr", "home_el", "away_el", "gamecode", "date_utc", "phase"]].rename(
            columns={"home_abbr": "team_abbr", "home_el": "team", "away_el": "opponent"}).assign(is_home=True),
        m[["match_id", "away_abbr", "away_el", "home_el", "gamecode", "date_utc", "phase"]].rename(
            columns={"away_abbr": "team_abbr", "away_el": "team", "home_el": "opponent"}).assign(is_home=False),
    ])
    opp_abbr = dict(zip(m["home_abbr"], m["away_abbr"])) | dict(zip(m["away_abbr"], m["home_abbr"]))

    pool = (lineups.drop(columns=["is_home", "match_id", "match_status"], errors="ignore").reset_index()
            .merge(side.drop(columns="match_id"), on="team_abbr", how="left").set_index("player_id"))
    pool["player_code"] = pool.index.map(mp)
    pool["opp"] = pool["team_abbr"].map(opp_abbr)
    pool["pos"] = pool["position"].map(POS_SHORT)
    players = pool[pool["position"] != "Head Coach"].copy()
    coaches = pool[pool["position"] == "Head Coach"].copy()

    # ---- resolve every name first, so a typo fails before the slow part
    everyone = pd.concat([players[["name"]], coaches[["name"]]])
    override = {}
    for k, v in (overrides or {}).items():
        v = str(v).strip().upper()
        if v not in OVERRIDE_VALUES:
            raise ValueError(f"override for '{k}' must be one of {', '.join(OVERRIDE_VALUES)}, not '{v}'")
        override[_resolve(players, k)] = v
    excl = {_resolve(everyone, k) for k in excluded}
    lock = {_resolve(everyone, k) for k in locked}
    current = {_resolve(everyone, k) for k in roster} if roster else None

    # ---- availability inputs: overrides, app status, who played the last game
    last = _last_game_players(ctx, year)
    played_last = set(zip(last["team"], last["player_code"]))
    pgs = store.player_games
    played_this_season = set(pgs.loc[(pgs["season"] == year) & pgs["played"], "player_code"])
    # out = your OUT override, or the app's out flag unless you overrode it
    app_out = ((players["p_play"].fillna(1.0) <= 0) | (players["lineup_status"] == "out")) & ~players.index.isin(list(override))
    is_out = app_out | players.index.isin([pid for pid, v in override.items() if v == "OUT"])

    # ---- features for matched players
    matched = players["player_code"].notna() & players["gamecode"].notna()
    pos_el = _latest_positions(store.player_games)
    q = players[matched].copy()
    q["position_name"] = q["player_code"].map(pos_el).fillna(q["position"])
    q["season"], q["round"] = year, md["number"]
    q = q.reset_index()
    X = ctx.features(q)
    X["player_id"] = q["player_id"].to_numpy()

    # live absences: last-game players who are out (or no longer in the app's list)
    now = pd.Timestamp(datetime.now(timezone.utc))
    in_app = set(zip(players["team"], players["player_code"]))
    out_codes = set(zip(players.loc[is_out, "team"], players.loc[is_out, "player_code"]))
    gone = last[[(t, c) not in in_app or (t, c) in out_codes for t, c in zip(last["team"], last["player_code"])]]
    if len(gone):
        mins = asof_join(gone.assign(date_utc=now), ctx.pstate[["player_code", "date_utc", "min_ewm_h5"]], by="player_code")
        gone = gone.assign(min_ewm_h5=mins["min_ewm_h5"].fillna(0.0).to_numpy())
    vac = gone.groupby("team")["min_ewm_h5"].sum() if len(gone) else pd.Series(dtype=float)
    vac_pos = gone.groupby(["team", "position_name"])["min_ewm_h5"].sum() if len(gone) else pd.Series(dtype=float)
    X["vacated_min"] = X["team"].map(vac).fillna(0.0).to_numpy()
    X["opp_vacated_min"] = X["opponent"].map(vac).fillna(0.0).to_numpy()
    for p in POSITIONS:
        X[f"vacated_min_{p}"] = [float(vac_pos.get((t, p), 0.0)) for t in X["team"]]
    X["vacated_min_pos"] = [float(vac_pos.get((t, p), 0.0)) for t, p in zip(X["team"], X["position_name"])]
    # never count a player's own absence in his own row (see features/build.py)
    if len(gone):
        own = {(t, c): (m_, ps) for t, c, m_, ps in zip(gone["team"], gone["player_code"], gone["min_ewm_h5"], gone["position_name"])}
        own_min = np.array([own.get((t, c), (0.0, None))[0] for t, c in zip(X["team"], X["player_code"])])
        own_pos = [own.get((t, c), (0.0, None))[1] for t, c in zip(X["team"], X["player_code"])]
        X["vacated_min"] = (X["vacated_min"] - own_min).clip(lower=0.0)
        for p in POSITIONS:
            X[f"vacated_min_{p}"] = (X[f"vacated_min_{p}"] - np.where(np.array(own_pos) == p, own_min, 0.0)).clip(lower=0.0)
        X["vacated_min_pos"] = (X["vacated_min_pos"] - np.where(np.array(own_pos) == X["position_name"].to_numpy(), own_min, 0.0)).clip(lower=0.0)

    # ---- models trained on all history
    train = ctx.training_rows()
    pm = PlayerPointsModel().fit(train)
    cand = build_candidates(ctx, seasons=tuple(range(2016, year + 1)))
    Xc = ctx.features(cand)
    av_rows = pd.concat([Xc, cand[["pool_prev", "y_played"]]], axis=1)
    am = AvailabilityModel().fit(av_rows[av_rows["pool_prev"]])
    gm = GameModel().fit(game_frame(ctx))
    fallback = B.debut_fallback(train)

    X["pts_if_plays"] = pm.predict(X)
    X["p_model"] = am.predict(X)
    players["pts_if_plays"] = np.nan
    players.loc[X["player_id"], "pts_if_plays"] = X["pts_if_plays"].to_numpy()
    players.loc[X["player_id"], "p_model"] = X["p_model"].to_numpy()
    players["pts_if_plays"] = players["pts_if_plays"].fillna(fallback)

    # ---- combine availability
    def p_final(pid: int, r: pd.Series) -> float:
        if pid in override:
            v = override[pid]
            if v in OVERRIDE_P:
                return OVERRIDE_P[v]
        elif r["p_play"] is not None and not pd.isna(r["p_play"]) and r["p_play"] < 1:
            return float(r["p_play"])
        if (r["team"], r["player_code"]) in played_last and not pd.isna(r.get("p_model")):
            return float(r["p_model"])
        if r["player_code"] not in played_this_season:
            return NO_HISTORY_P
        return RETURNING_P

    players["p_play_app"] = players["p_play"]
    players["p_play"] = [p_final(pid, r) for pid, r in players.iterrows()]
    players["exp_pts"] = players["p_play"] * players["pts_if_plays"]
    players["captain_ok"] = True
    if moved_teams:
        # game moved out of the round: everyone on these teams scores his season average
        # (injured players too) and the captain is not doubled
        season_avg = pd.Series(X["fp_season_mean"].to_numpy(), index=X["player_id"])
        on_moved = players["team_abbr"].isin(moved_teams)
        players.loc[on_moved, "exp_pts"] = players.index[on_moved].map(season_avg).fillna(0.0)
        players.loc[on_moved, "p_play"] = 1.0
        players.loc[on_moved, "captain_ok"] = False
        notes.append(f"Game(s) moved out of this round for {', '.join(sorted(moved_teams))}: their players and "
                     f"coaches score their season average and cannot be doubled as captain")
    players["real_lineup"] = players["lineup_status"]   # app projection: starter / bench / out
    players["team"] = players["team_abbr"]
    unmatched = int((~matched).sum())
    no_hist = int((~players["player_code"].isin(played_this_season)).sum())
    if unmatched or no_hist:
        notes.append(f"{no_hist} players have not played a EuroLeague game this season ({unmatched} of them "
                     f"not mapped to EuroLeague ids): debut average {fallback:.1f} pts × P(plays) "
                     f"{NO_HISTORY_P:.2f} unless you override")

    # ---- coaches: game model on this round's games (live absences included)
    base = pd.concat([
        g[["season", "gamecode", "date_utc", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"}).assign(is_home=True),
        g[["season", "gamecode", "date_utc", "home_team", "away_team"]].rename(columns={"away_team": "team", "home_team": "opponent"}).assign(is_home=False),
    ])
    vac_tbl = base[["season", "gamecode", "team"]].assign(vacated_min=base["team"].map(vac).fillna(0.0).to_numpy())
    gfe = game_features(ctx, base, vac_tbl)
    gfe["coach_exp"] = gm.coach_points(gfe)
    gfe["p_win"] = gm.p_win(gfe)
    coaches = coaches.merge(gfe[["team", "coach_exp", "p_win"]], on="team", how="left").set_axis(coaches.index)
    coaches["exp_pts"] = coaches["coach_exp"]
    if moved_teams:
        tgs = store.team_games
        coach_avg = tgs[tgs["season"] == year].groupby("team")["coach_pts"].mean()
        on_moved = coaches["team_abbr"].isin(moved_teams)
        coaches.loc[on_moved, "exp_pts"] = coaches.loc[on_moved, "team"].map(coach_avg).fillna(0.0).to_numpy()
    coaches["home"] = coaches["is_home"]
    coaches["team"] = coaches["team_abbr"]

    # ---- optimize
    squad = best_squad(players.assign(exp=players["exp_pts"]), coaches.assign(exp=coaches["exp_pts"]),
                       budget=budget, locked=lock, excluded=excl)
    if squad.status != "Optimal":
        raise RuntimeError(f"optimizer status {squad.status}: check budget/locks")

    trades, mine = None, None
    if current:
        cur_value = float(players.loc[players.index.intersection(list(current)), "price"].sum()
                          + coaches.loc[coaches.index.intersection(list(current)), "price"].sum())
        mine = best_squad(players.assign(exp=players["exp_pts"]), coaches.assign(exp=coaches["exp_pts"]),
                          budget=bank + cur_value + 1e-6, locked=lock, excluded=excl, current=current,
                          max_new=max_trades, trade_penalty=TRADE_EPSILON)
        keep = best_squad(players.assign(exp=players["exp_pts"]), coaches.assign(exp=coaches["exp_pts"]),
                          budget=bank + cur_value + 1e-6, current=current, max_new=0)
        if mine.status != "Optimal":
            raise RuntimeError(f"roster mode: optimizer status {mine.status} (is the roster complete and valid?)")
        new_ids = set(mine.players.index) | {mine.coach.name}
        trades = {
            "out": [everyone.loc[i, "name"] for i in current - new_ids],
            "in": [everyone.loc[i, "name"] for i in new_ids - current],
            "gain": mine.objective - (keep.objective if keep.status == "Optimal" else np.nan),
            "squad": mine,
        }

    # ---- explanations for the squad (LightGBM feature contributions)
    explanations = {}
    picked = set(squad.players.index) | (set(mine.players.index) if mine is not None else set())
    sel = X[X["player_id"].isin(picked)]
    if len(sel):
        contrib = pm.model.predict(_matrix(sel), pred_contrib=True)
        names = list(_matrix(sel).columns)
        for row_i, pid in enumerate(sel["player_id"]):
            c = pd.Series(contrib[row_i, :-1], index=names)
            top = c.reindex(c.abs().sort_values(ascending=False).index).head(6)
            explanations[int(pid)] = [(f, sel.iloc[row_i][f], float(v), FEATURE_DESCRIPTIONS.get(f, "")) for f, v in top.items()]

    data_through = store.player_games["date_utc"].max()
    source = "from the snapshot" if snap is not None else f"fetched {now:%Y-%m-%d %H:%M} UTC"
    notes.append(f"Box scores through {data_through:%Y-%m-%d}; app line-ups {source}")
    if snap is None and deadline is not None and deadline < now:
        notes.append("The deadline has passed: this round has already started")
    return Recommendation(matchday=md["number"], deadline=deadline, budget=budget, players=players,
                          coaches=coaches, squad=squad, explanations=explanations, trades=trades, notes=notes,
                          my_squad=mine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchday", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", action="append", default=[], help="player you know is out (repeatable)")
    ap.add_argument("--doubtful", action="append", default=[], help="30%% chance to play (repeatable)")
    ap.add_argument("--probable", action="append", default=[], help="75%% chance to play (repeatable)")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--lock", action="append", default=[])
    ap.add_argument("--refresh", action="store_true", help="fetch new box scores and player list first")
    ap.add_argument("--snapshot", default=None, help="replay a past round from a saved snapshot (.json.gz)")
    ap.add_argument("--save", action="store_true", help="write the markdown to recommendations/")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = ({n: "OUT" for n in args.out} | {n: "DOUBTFUL" for n in args.doubtful}
                 | {n: "PROBABLE" for n in args.probable})
    try:
        rec = recommend(matchday=args.matchday, budget=args.budget, overrides=overrides,
                        excluded=args.exclude, locked=args.lock, refresh=args.refresh,
                        snapshot=args.snapshot)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")
    print(rec.to_markdown())
    if args.save:
        print(f"\nsaved to {rec.save()}")


if __name__ == "__main__":
    main()
