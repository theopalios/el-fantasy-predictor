"""Check saved recommendations against what actually happened.

    python -m elfantasy.evaluate            # all completed rounds, writes the tracking report

For every recommendation JSON under recommendations/<season>/ whose round has been played,
score the recommended squad with the real fantasy points (box scores + rules file), and
compare it with the expected total and, when the round's prices are known, with the best
squad in hindsight.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from elfantasy.data.build import current_season_year
from elfantasy.optimize.squad import best_squad
from elfantasy.paths import PROCESSED_DIR, REPO_ROOT
from elfantasy.rules import Rules, default_rules

log = logging.getLogger("elfantasy.evaluate")
POS_MAP = {"Guard": "G", "Forward": "F", "Center": "C"}


def score_record(record: dict, player_pts: dict, coach_pts: float | None, rules: Rules | None = None) -> dict:
    """Actual squad points for one recommendation record.

    player_pts: app player_id -> actual fantasy points (missing = did not play = 0).
    """
    rules = rules or default_rules()
    mult = rules.multipliers
    weights = {"starter": mult["starter"], "sixth": mult["sixth_man"], "bench": mult["bench"]}
    rows, total = [], 0.0
    for p in record["squad"]:
        pts = float(player_pts.get(p["player_id"], 0.0) or 0.0)
        w = weights[p["role"]] * (rules.captain_multiplier if p["captain"] else 1.0)
        total += w * pts
        rows.append({"name": p["name"], "role": p["role"] + (" (C)" if p["captain"] else ""),
                     "expected": p["exp_pts"], "actual": pts, "played": p["player_id"] in player_pts})
    coach_actual = float(coach_pts) if coach_pts is not None else 0.0
    total += mult["head_coach"] * coach_actual
    return {"actual_total": total, "coach_actual": coach_actual, "players": pd.DataFrame(rows)}


def _round_complete(games: pd.DataFrame, season: int, rnd: int) -> bool:
    g = games[(games["season"] == season) & (games["round"] == rnd)]
    return len(g) > 0 and bool(g["played"].all())


def actual_points(season: int, rnd: int) -> tuple[dict, dict]:
    """(app player_id -> fantasy pts, app team abbr -> coach pts) for a played round."""
    pg = pd.read_parquet(PROCESSED_DIR / "player_games.parquet")
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    mp = pd.read_parquet(PROCESSED_DIR / "map_player.parquet")
    mt = pd.read_parquet(PROCESSED_DIR / "map_team.parquet")
    r = pg[(pg["season"] == season) & (pg["round"] == rnd) & pg["played"]]
    pts = r.groupby("player_code")["fantasy_pts"].sum()
    by_id = {int(pid): float(pts[code]) for pid, code in zip(mp["player_id"], mp["person_code"])
             if isinstance(code, str) and code in pts.index}
    c = tg[(tg["season"] == season) & (tg["round"] == rnd)].set_index("team")["coach_pts"]
    abbr_to_code = dict(zip(mt["app_abbr"], mt["el_code"]))
    coach = {abbr: float(c[code]) for abbr, code in abbr_to_code.items() if code in c.index}
    return by_id, coach


def hindsight_best(season: int, rnd: int, player_pts: dict, coach_pts: dict, budget: float) -> float | None:
    """Best possible squad for the round given actual points and that round's prices."""
    ml_path = PROCESSED_DIR / "fantasy_match_log.parquet"
    if not ml_path.exists() or season != current_season_year():
        return None
    ml = pd.read_parquet(ml_path)
    fp = pd.read_parquet(PROCESSED_DIR / "fantasy_players.parquet")
    prices = ml[ml["matchday"] == rnd][["player_id", "price"]].merge(
        fp[["player_id", "position", "team_abbr", "first_name", "last_name"]], on="player_id")
    if prices.empty:
        return None
    players = prices[prices["position"] != "Head Coach"].set_index("player_id")
    players = players.assign(pos=players["position"].map(POS_MAP), team=players["team_abbr"],
                             exp=[player_pts.get(int(i), 0.0) for i in players.index])
    coaches = prices[prices["position"] == "Head Coach"].set_index("player_id")
    coaches = coaches.assign(team=coaches["team_abbr"], exp=[coach_pts.get(t, 0.0) for t in coaches["team_abbr"]])
    res = best_squad(players, coaches, budget=budget)
    return res.objective if res.status == "Optimal" else None


def evaluate_all(folder: Path | None = None) -> pd.DataFrame:
    year = current_season_year()
    folder = folder or REPO_ROOT / "recommendations" / f"{year}-{str(year + 1)[-2:]}"
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    rows, details = [], []
    for path in sorted(folder.glob("md*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        season, rnd = int(rec["season"]), int(rec["matchday"])
        if not _round_complete(games, season, rnd):
            rows.append({"file": path.name, "round": rnd, "status": "round not finished"})
            continue
        player_pts, coach_pts = actual_points(season, rnd)
        res = score_record(rec, player_pts, coach_pts.get(rec["coach"]["team"]))
        best = hindsight_best(season, rnd, player_pts, coach_pts, rec["budget"])
        rows.append({"file": path.name, "round": rnd, "status": "scored",
                     "expected": rec["expected_total"], "actual": res["actual_total"],
                     "hindsight_best": best,
                     "pct_of_hindsight": 100 * res["actual_total"] / best if best else np.nan})
        details.append((path.name, res["players"].assign(coach_actual=res["coach_actual"])))
    summary = pd.DataFrame(rows)
    report = folder / "tracking.md"
    lines = ["# Recommendation tracking", "", "Generated by `python -m elfantasy.evaluate`.", ""]
    lines += [summary.to_markdown(index=False, floatfmt=".1f") if len(summary) else "No recommendations yet.", ""]
    for name, d in details:
        lines += [f"## {name}", "", d.to_markdown(index=False, floatfmt=".1f"), ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", report)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = evaluate_all()
    print(summary.to_markdown(index=False, floatfmt=".1f") if len(summary) else "No recommendations found.")


if __name__ == "__main__":
    main()
