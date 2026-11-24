"""Build current-season fantasy tables from the public app endpoints and map them to the
EuroLeague stats API (teams and players).

    python -m elfantasy.data.fantasy_build

Writes to data/processed/:
  fantasy_matches          schedule of every matchday (app match ids, turns, scores)
  fantasy_lineups_current  current matchday, per match x player: price, projected status, P(plays)
  fantasy_players          player list with fantasy position, team, price, popularity, injured flag
  fantasy_match_log   per player x matchday: price before round, price change, fantasy points, box score
  map_team            app team abbreviation <-> EuroLeague club code
  map_player          app player id <-> EuroLeague person code
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re

import pandas as pd
from unidecode import unidecode

from elfantasy.data.build import current_season_year
from elfantasy.data.euroleague import EuroLeagueAPI, parse_games, parse_people
from elfantasy.data.fantasy_app import (
    FantasyAppAPI,
    parse_lineups,
    parse_match_log,
    parse_profile,
    parse_schedule,
)
from elfantasy.data.http import HttpClient
from elfantasy.paths import PROCESSED_DIR, RAW_CACHE_DIR

log = logging.getLogger("elfantasy.fantasy_build")

_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


def norm_name(s: str | None) -> str:
    s = unidecode(s or "").upper()
    s = re.sub(r"['.`]", "", s)          # M'baye -> MBAYE, J.J. -> JJ
    s = re.sub(r"[^A-Z ]+", " ", s)      # hyphens etc. -> space
    return " ".join(t for t in s.split() if t not in _SUFFIXES)


def split_el_name(name: str) -> tuple[str, str]:
    """'KABENGELE, MFIONDU' -> ('KABENGELE', 'MFIONDU')."""
    last, _, first = (name or "").partition(",")
    return norm_name(last), norm_name(first)


def map_teams(app_matches: pd.DataFrame, el_games: pd.DataFrame) -> pd.DataFrame:
    """Pair app matches with EuroLeague games by start time (within 3h) and read off codes."""
    pairs = []
    el = el_games.dropna(subset=["date_utc"])
    for m in app_matches.dropna(subset=["started_at"]).itertuples(index=False):
        delta = (el["date_utc"] - m.started_at).abs()
        cand = el[delta <= pd.Timedelta(hours=3)]
        for g in cand.itertuples(index=False):
            pairs.append((m.home_abbr, m.home_team_id, g.home_team))
            pairs.append((m.away_abbr, m.away_team_id, g.away_team))
    df = pd.DataFrame(pairs, columns=["app_abbr", "app_team_id", "el_code"])
    # Simultaneous tip-offs create wrong pairs; the true code is the most frequent one.
    best = (
        df.groupby(["app_abbr", "app_team_id", "el_code"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False).drop_duplicates("app_abbr")
    )
    return best.drop(columns="n").reset_index(drop=True)


def map_players(app_players: pd.DataFrame, el_people: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    abbr_to_code = dict(zip(team_map["app_abbr"], team_map["el_code"]))
    people = el_people[el_people["type"].isin(["J", "E"])].copy()
    people[["last", "first"]] = people["name"].apply(lambda n: pd.Series(split_el_name(n)))
    people["full"] = (people["first"] + " " + people["last"]).str.strip()

    out = []
    for p in app_players.itertuples(index=False):
        club = abbr_to_code.get(p.team_abbr)
        want_type = "E" if p.position == "Head Coach" else "J"
        pool = people[(people["club"] == club) & (people["type"] == want_type)]
        last, first = norm_name(p.last_name), norm_name(p.first_name)
        method, hit = None, pd.DataFrame()
        for method_name, mask in (
            ("exact", (pool["last"] == last) & (pool["first"] == first)),
            ("last+initial", (pool["last"] == last) & (pool["first"].str[:1] == first[:1])),
            ("last", pool["last"] == last),
        ):
            hit = pool[mask].drop_duplicates("person_code")
            if len(hit) == 1:
                method = method_name
                break
        if method is None and len(pool):
            # Fuzzy only on the surname, and the first initial must agree, so that e.g.
            # MIJAILOVIC never pairs with DANILOVIC just because both are called Uros.
            same_initial = pool[pool["first"].str[:1] == first[:1]]
            if len(same_initial):
                scores = same_initial["last"].apply(lambda l: difflib.SequenceMatcher(None, l, last).ratio())
                if scores.max() >= 0.85:
                    hit = same_initial.loc[[scores.idxmax()]]
                    method = f"fuzzy:{scores.max():.2f}"
        out.append(
            {
                "player_id": p.player_id,
                "app_name": f"{p.first_name} {p.last_name}",
                "team_abbr": p.team_abbr,
                "el_code": club,
                "person_code": hit["person_code"].iloc[0] if method else None,
                "el_name": hit["name"].iloc[0] if method else None,
                "match_method": method,
            }
        )
    return pd.DataFrame(out)


def build_fantasy(with_profiles: bool = True, with_match_logs: bool = True) -> dict[str, pd.DataFrame]:
    """with_match_logs=False skips the per-player price/points history (only the backtest needs
    it), which makes a weekly refresh of the player list and id maps take ~30 s."""
    year = current_season_year()
    app = FantasyAppAPI(HttpClient(min_interval=0.3))
    el = EuroLeagueAPI(HttpClient(cache_dir=RAW_CACHE_DIR))

    cfg = app.config()
    schedule_id = cfg["current_schedule_id"]
    current_md = cfg["current_matchday"]["number"]
    matchdays = cfg["matchdays"]

    sched = []
    for md in matchdays:
        sched.append(parse_schedule(app.schedule_matchday(schedule_id, md["id"])))
    matches = pd.concat(sched, ignore_index=True)
    log.info("schedule: %d matchdays, %d matches (current matchday %s)", len(matchdays), len(matches), current_md)

    # Lineups reflect the app's CURRENT projection even for past matches (verified: same
    # "out" list and today's prices on every past matchday), so only the current
    # matchday is read here. Pre-deadline history comes from snapshots/ only.
    lineups = []
    for mid in matches.loc[matches["matchday"] == current_md, "match_id"]:
        lineups.append(parse_lineups(app.match_lineups(int(mid))))
    lineups = pd.concat(lineups, ignore_index=True).merge(
        matches[["match_id", "matchday"]], on="match_id", how="left"
    )
    log.info("current lineups: %d rows, %d distinct players", len(lineups), lineups["player_id"].nunique())

    players = (
        lineups.drop_duplicates("player_id", keep="last")
        [["player_id", "first_name", "last_name", "position", "team_id", "team_abbr", "price",
          "lineup_status", "p_play"]]
        .reset_index(drop=True)
    )

    logs, profiles = [], []
    for i, pid in enumerate(players["player_id"], 1):
        if with_match_logs:
            logs.append(parse_match_log(app.player_match_log(int(pid)), int(pid)))
        if with_profiles:
            profiles.append(parse_profile(app.player_profile(int(pid))))
        if i % 100 == 0 and (with_match_logs or with_profiles):
            log.info("  players: %d/%d", i, len(players))
    match_log = pd.concat(logs, ignore_index=True) if logs else None
    if match_log is not None:
        match_log["is_future"] = match_log["matchday"] >= current_md
    if profiles:
        prof = pd.DataFrame(profiles)[
            ["player_id", "avg_fantasy_pts", "total_price_change", "popularity", "is_on_fire", "is_injured", "started_from_bench"]
        ]
        players = players.merge(prof, on="player_id", how="left")

    el_games = parse_games(el.games(year), year)
    team_map = map_teams(matches, el_games)
    clubs = sorted(team_map["el_code"].unique())
    el_people = pd.concat([parse_people(el.club_people(year, c), year, c) for c in clubs], ignore_index=True)
    # Some players are missing from the roster endpoint but appear in box scores: add them.
    pg_path = PROCESSED_DIR / "player_games.parquet"
    if pg_path.exists():
        pg = pd.read_parquet(pg_path, columns=["season", "team", "player_code", "player_name"])
        extra = (
            pg[pg["season"] == year].drop_duplicates(["team", "player_code"])
            .rename(columns={"team": "club", "player_code": "person_code", "player_name": "name"})
            .assign(type="J")
        )
        extra = extra[~extra["person_code"].isin(el_people["person_code"])]
        el_people = pd.concat([el_people, extra[["club", "person_code", "name", "type"]]], ignore_index=True)
    player_map = map_players(players, el_people, team_map)
    log.info("team map: %d/%d teams; player map: %d/%d matched",
             len(team_map), matches["home_abbr"].nunique(),
             player_map["match_method"].notna().sum(), len(player_map))

    tables = {
        "fantasy_matches": matches,
        "fantasy_lineups_current": lineups,
        "fantasy_players": players,
        "fantasy_match_log": match_log,
        "map_team": team_map,
        "map_player": player_map,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is None:
            continue
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        log.info("wrote %s (%d rows)", name, len(df))
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-profiles", action="store_true", help="skip per-player profile calls")
    ap.add_argument("--no-match-logs", action="store_true", help="skip per-player price/points history")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_fantasy(with_profiles=not args.no_profiles, with_match_logs=not args.no_match_logs)


if __name__ == "__main__":
    main()
