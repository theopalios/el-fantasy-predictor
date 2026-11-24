"""Build the historical EuroLeague dataset (schedule, box scores, rosters) as parquet tables.

    python -m elfantasy.data.build                 # 2016-17 .. current season
    python -m elfantasy.data.build --first 2023    # subset

Completed resources are cached under data/raw, so re-runs only fetch new games.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd

from elfantasy.data.euroleague import (
    LIVE,
    V2,
    EuroLeagueAPI,
    season_code,
    add_game_context,
    parse_boxscore,
    parse_game_stats,
    parse_games,
    parse_people,
)
from elfantasy.data.http import HttpClient
from elfantasy.paths import PROCESSED_DIR, RAW_CACHE_DIR
from elfantasy.rules import default_rules
from elfantasy.scoring import coach_points_series, pir_from_stats, player_fantasy_points
import numpy as np

log = logging.getLogger("elfantasy.build")

FIRST_SEASON = 2016  # single round-robin era starts in 2016-17


def current_season_year(today: datetime | None = None) -> int:
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


def _v2_url(year: int, gamecode: int) -> tuple[str, None]:
    return (f"{V2}/competitions/E/seasons/{season_code(year)}/games/{gamecode}/stats", None)


def _legacy_req(year: int, gamecode: int) -> tuple[str, dict]:
    return (f"{LIVE}/Boxscore", {"gamecode": gamecode, "seasoncode": season_code(year)})


def build_euroleague(
    first: int = FIRST_SEASON,
    last: int | None = None,
    refresh_days: int = 3,
    workers: int = 2,
    min_interval: float = 0.5,
    prefer_legacy: bool = False,
) -> dict[str, pd.DataFrame]:
    # The API rate-limits bursts (~2,000 requests in 10 min triggered a 5-min block), so the
    # default pace is ~2 requests/s. Everything fetched is cached, so re-runs are cheap.
    cur = current_season_year()
    last = last if last is not None else cur
    http = HttpClient(cache_dir=RAW_CACHE_DIR, min_interval=min_interval)
    api = EuroLeagueAPI(http)
    # The legacy box-score host is slower (~2.5 s/request) but rate-limited separately,
    # so it gets its own client and pacing.
    legacy_api = EuroLeagueAPI(HttpClient(cache_dir=RAW_CACHE_DIR, min_interval=0.4, max_retries=3))
    now = datetime.now(timezone.utc)

    games_l, players_l, teams_l, people_l, failures = [], [], [], [], []
    for year in range(first, last + 1):
        is_current = year == cur
        games = parse_games(api.games(year, cache=not is_current), year)
        if games.empty:
            # e.g. from 1 July until the new season's calendar is published
            log.warning("season %s-%s has no games yet: skipped", year, str(year + 1)[-2:])
            continue
        games_l.append(games)
        played = games[games["played"]]
        log.info("season %s-%s: %d games (%d played)", year, str(year + 1)[-2:], len(games), len(played))

        def fetch_v2(g, recent):
            raw = api.game_stats(year, g.gamecode, cache=not recent)
            return parse_game_stats(raw, year, g.gamecode, g.home_team, g.away_team)

        def fetch_legacy(g, recent):
            box = legacy_api.boxscore(year, g.gamecode, cache=not recent)
            return parse_boxscore(box, year, g.gamecode)

        def fetch(g):
            recent = is_current and (now - g.date_utc.to_pydatetime()) < timedelta(days=refresh_days)
            # A game already cached from either source is reused without a request.
            order = (fetch_legacy, fetch_v2) if prefer_legacy else (fetch_v2, fetch_legacy)
            if not recent:
                for fn, cached in ((fetch_v2, api.http.is_cached(*_v2_url(year, g.gamecode))),
                                   (fetch_legacy, legacy_api.http.is_cached(*_legacy_req(year, g.gamecode)))):
                    if cached:
                        return fn(g, recent)
            try:
                return order[0](g, recent)
            except Exception as exc:
                log.warning("%s %s/%s failed (%s); trying the other source", order[0].__name__, year, g.gamecode, exc)
                return order[1](g, recent)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, g): g.gamecode for g in played.itertuples(index=False)}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    p, t = fut.result()
                except Exception as exc:  # keep going; report at the end
                    failures.append((year, futures[fut], repr(exc)))
                    log.warning("game %s/%s failed: %s", year, futures[fut], exc)
                    continue
                players_l.append(p)
                teams_l.append(t)
                if i % 100 == 0:
                    log.info("  %s: %d/%d games", year, i, len(played))

        clubs = sorted(set(games["home_team"].dropna()) | set(games["away_team"].dropna()))
        for club in clubs:
            try:
                people_l.append(parse_people(api.club_people(year, club, cache=not is_current), year, club))
            except Exception as exc:
                failures.append((year, f"people:{club}", repr(exc)))
                log.warning("people %s/%s failed: %s", year, club, exc)

    games = pd.concat(games_l, ignore_index=True)
    player_games = pd.concat(players_l, ignore_index=True)
    team_games = pd.concat(teams_l, ignore_index=True)
    people = pd.concat(people_l, ignore_index=True) if people_l else pd.DataFrame()

    # A game whose download failed this time keeps its previously downloaded rows, so a
    # temporary API problem never silently removes games from the tables.
    failed_games = {(y, g) for y, g, _ in failures if isinstance(g, (int, np.integer))}
    if failed_games and (PROCESSED_DIR / "player_games.parquet").exists():
        keys = ["season", "gamecode"]
        failed_df = pd.DataFrame(sorted(failed_games), columns=keys)
        old_pg = pd.read_parquet(PROCESSED_DIR / "player_games.parquet").merge(failed_df, on=keys)
        old_tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet").merge(failed_df, on=keys)
        base_cols = [c for c in player_games.columns if c in old_pg.columns]
        player_games = pd.concat([player_games, old_pg[base_cols]], ignore_index=True)
        team_games = pd.concat([team_games, old_tg[[c for c in team_games.columns if c in old_tg.columns]]], ignore_index=True)
        log.warning("%d games failed to download; kept %d previously downloaded player rows for them",
                    len(failed_games), len(old_pg))

    player_games, team_games = add_game_context(player_games, team_games, games)
    add_fantasy_scores(player_games, team_games)
    fill_positions(player_games, people)

    tables = {
        "games": games,
        "player_games": player_games,
        "team_games": team_games,
        "people": people,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        log.info("wrote %s (%d rows)", PROCESSED_DIR / f"{name}.parquet", len(df))

    qa_report(player_games, failures)
    return tables


def add_fantasy_scores(player_games: pd.DataFrame, team_games: pd.DataFrame) -> None:
    """Add fantasy_pts (players) and coach_pts (teams) in place, using the rules file."""
    rules = default_rules()
    player_games["played"] = player_games["minutes"] > 0
    player_games["fantasy_pts"] = 0.0
    mask = player_games["played"]
    player_games.loc[mask, "fantasy_pts"] = player_fantasy_points(
        player_games.loc[mask, "pir"], player_games.loc[mask, "win"], rules
    )
    team_games["coach_pts"] = coach_points_series(team_games["margin"], team_games["n_ot"] > 0, rules)


def fill_positions(player_games: pd.DataFrame, people: pd.DataFrame) -> None:
    """Fill missing position_name (legacy box scores lack it) from that season's rosters,
    falling back to the player's most common position in other seasons."""
    if "position_name" not in player_games:
        player_games["position_name"] = None
    if people.empty:
        return
    roster = (
        people[(people["type"] == "J") & people["position_name"].notna()]
        .drop_duplicates(["season", "person_code"])
        .set_index(["season", "person_code"])["position_name"]
    )
    key = pd.MultiIndex.from_arrays([player_games["season"], player_games["player_code"]])
    missing = player_games["position_name"].isna()
    player_games.loc[missing, "position_name"] = roster.reindex(key[missing.to_numpy()]).to_numpy()
    usual = (
        player_games.dropna(subset=["position_name"]).groupby("player_code")["position_name"]
        .agg(lambda s: s.mode().iloc[0])
    )
    still = player_games["position_name"].isna()
    player_games.loc[still, "position_name"] = player_games.loc[still, "player_code"].map(usual)


def rescore() -> None:
    """Recompute derived columns on the saved tables (after a rules change), no downloads."""
    pg = pd.read_parquet(PROCESSED_DIR / "player_games.parquet")
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    people = pd.read_parquet(PROCESSED_DIR / "people.parquet")
    add_fantasy_scores(pg, tg)
    fill_positions(pg, people)
    pg.to_parquet(PROCESSED_DIR / "player_games.parquet", index=False)
    tg.to_parquet(PROCESSED_DIR / "team_games.parquet", index=False)
    log.info("rescored %d player rows and %d team rows; positions missing: %.2f%%",
             len(pg), len(tg), 100 * pg["position_name"].isna().mean())


def qa_report(player_games: pd.DataFrame, failures: list) -> None:
    recomputed = pir_from_stats(player_games)
    mismatch = (recomputed != player_games["pir"]).mean()
    log.info("QA: PIR recomputed from stats != official Valuation in %.3f%% of rows", 100 * mismatch)
    per_season = player_games[player_games["played"]].groupby("season").agg(
        games=("gamecode", "nunique"), player_rows=("player_code", "size"), mean_fp=("fantasy_pts", "mean")
    )
    log.info("QA per season:\n%s", per_season.to_string())
    if failures:
        log.warning("%d failures: %s", len(failures), failures[:10])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=FIRST_SEASON, help="first season start year (default 2016)")
    ap.add_argument("--last", type=int, default=None, help="last season start year (default current)")
    ap.add_argument("--refresh-days", type=int, default=3, help="re-fetch box scores of games played in the last N days")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (keep small: public API)")
    ap.add_argument("--min-interval", type=float, default=0.5, help="seconds between request starts")
    ap.add_argument("--prefer-legacy", action="store_true",
                    help="fetch box scores from the legacy endpoint first (use when v2 is rate-limited)")
    ap.add_argument("--rescore-only", action="store_true", help="only recompute fantasy points on saved tables")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.rescore_only:
        rescore()
        return
    build_euroleague(args.first, args.last, args.refresh_days, args.workers, args.min_interval, args.prefer_legacy)


if __name__ == "__main__":
    main()
