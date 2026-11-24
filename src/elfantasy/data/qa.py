"""Data quality checks on the processed tables.

    python -m elfantasy.data.qa

1. PIR recomputed from box-score components equals the official Valuation.
2. Coverage per season (games, player rows, missing positions).
3. Our fantasy points (EuroLeague box scores + rules file) equal the app's fantasy points
   for the current season, player by player and round by round, and the same for coaches.
"""

from __future__ import annotations

import logging

import pandas as pd

from elfantasy.data.build import current_season_year
from elfantasy.paths import PROCESSED_DIR
from elfantasy.scoring import pir_from_stats

log = logging.getLogger("elfantasy.qa")


def _load(name: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def check_pir(pg: pd.DataFrame) -> float:
    mismatch = float((pir_from_stats(pg) != pg["pir"]).mean())
    log.info("[1] PIR from components != Valuation: %.4f%% of %d rows", 100 * mismatch, len(pg))
    return mismatch


def check_coverage(pg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    played = pg[pg["played"]]
    cov = played.groupby("season").agg(
        games=("gamecode", "nunique"),
        player_rows=("player_code", "size"),
        players=("player_code", "nunique"),
        mean_fp=("fantasy_pts", "mean"),
        pos_missing=("position_name", lambda s: s.isna().mean()),
    )
    sched = games[games["played"]].groupby("season").size().rename("games_played_in_schedule")
    cov = cov.join(sched)
    log.info("[2] coverage per season:\n%s", cov.round(3).to_string())
    return cov


def check_vs_app(pg: pd.DataFrame, tg: pd.DataFrame) -> dict[str, float]:
    ml, pmap, tmap = _load("fantasy_match_log"), _load("map_player"), _load("map_team")
    players = _load("fantasy_players")
    if ml is None or pmap is None or tmap is None or players is None:
        log.warning("[3] fantasy tables missing; run python -m elfantasy.data.fantasy_build")
        return {}
    year = current_season_year()
    ours = pg[(pg["season"] == year) & pg["played"] & (pg["phase"] == "RS")]
    app = ml[(~ml["is_future"]) & (ml["minutes"] > 0)].merge(
        pmap[["player_id", "person_code"]].dropna(), on="player_id"
    ).merge(players[["player_id", "position"]], on="player_id")
    app_players = app[app["position"] != "Head Coach"]
    joined = app_players.merge(
        ours[["player_code", "round", "fantasy_pts", "pir"]],
        left_on=["person_code", "matchday"], right_on=["player_code", "round"],
        how="left", suffixes=("_app", "_ours"),
    )
    found = joined["fantasy_pts_ours"].notna()
    exact = ((joined["fantasy_pts_app"] - joined["fantasy_pts_ours"]).abs() <= 0.051)[found]
    log.info(
        "[3] players: %d app player-rounds, %.1f%% found in box scores, %.2f%% of those match exactly",
        len(joined), 100 * found.mean(), 100 * exact.mean(),
    )
    bad = joined[found & ~((joined["fantasy_pts_app"] - joined["fantasy_pts_ours"]).abs() <= 0.051)]
    if len(bad):
        log.info("    mismatches (first 10):\n%s", bad.head(10).to_string())

    # coaches: app trend/match-log points vs our coach_pts by team and round
    coaches = ml[(~ml["is_future"])].merge(players[players["position"] == "Head Coach"][["player_id", "team_abbr"]], on="player_id")
    coaches = coaches.merge(tmap[["app_abbr", "el_code"]], left_on="team_abbr", right_on="app_abbr")
    tc = tg[(tg["season"] == year) & (tg["phase"] == "RS")][["team", "round", "coach_pts"]]
    cj = coaches.merge(tc, left_on=["el_code", "matchday"], right_on=["team", "round"], how="inner")
    c_exact = float((cj["fantasy_pts"] == cj["coach_pts"]).mean()) if len(cj) else float("nan")
    log.info("[3] coaches: %d coach-rounds compared, %.2f%% match exactly", len(cj), 100 * c_exact)
    return {"player_found": float(found.mean()), "player_exact": float(exact.mean()), "coach_exact": c_exact}


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pg, tg, games = _load("player_games"), _load("team_games"), _load("games")
    if pg is None:
        raise SystemExit("run python -m elfantasy.data.build first")
    check_pir(pg)
    check_coverage(pg, games)
    check_vs_app(pg, tg)


if __name__ == "__main__":
    main()
