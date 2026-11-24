"""Snapshot the upcoming fantasy round before its deadline.

    python -m elfantasy.data.snapshot

Saves prices, projected lineups (starter/bench/out), probability of playing, ownership and
injured flags for every player in the next round's matches. The app does not keep the
pre-deadline state, so these snapshots are the only history of it: they are committed
to the repo under snapshots/<season>/md<NN>/<timestamp>.json.gz.

If ELF_FANTASY_TOKEN is set, in-app odds and the injury list are saved too.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from elfantasy.data.build import current_season_year
from elfantasy.data.fantasy_app import FantasyAppAPI, parse_schedule
from elfantasy.data.http import HttpClient
from elfantasy.paths import SNAPSHOT_DIR

log = logging.getLogger("elfantasy.snapshot")


def find_upcoming_matchday(app: FantasyAppAPI, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """First matchday (from the current one on) that still has unplayed matches."""
    schedule_id = cfg["current_schedule_id"]
    current = cfg["current_matchday"]["number"]
    for md in sorted(cfg["matchdays"], key=lambda m: m["number"]):
        if md["number"] < current:
            continue
        raw = app.schedule_matchday(schedule_id, md["id"])
        statuses = {m["status"] for r in raw.get("rounds", []) for m in r.get("matches", [])}
        if statuses - {"played"}:
            return md, raw
    raise RuntimeError("no upcoming matchday found")


def take_snapshot(with_profiles: bool = True, matchday: int | None = None) -> Path:
    app = FantasyAppAPI(HttpClient(min_interval=0.3))
    cfg = app.config()
    if matchday is None:
        md, sched = find_upcoming_matchday(app, cfg)
    else:
        md = next(m for m in cfg["matchdays"] if m["number"] == matchday)
        sched = app.schedule_matchday(cfg["current_schedule_id"], md["id"])

    matches = parse_schedule(sched)
    now = datetime.now(timezone.utc)
    deadline = matches["started_at"].min()
    out: dict[str, Any] = {
        "snapshot_utc": now.isoformat(),
        "matchday": md["number"],
        "matchday_id": md["id"],
        "deadline_utc": deadline.isoformat() if deadline is not None else None,
        "minutes_to_deadline": round((deadline - now).total_seconds() / 60, 1) if deadline is not None else None,
        "schedule": sched,
        "lineups": {},
        "profiles": {},
    }

    player_ids: set[int] = set()
    for mid in matches["match_id"]:
        lu = app.match_lineups(int(mid))
        out["lineups"][str(mid)] = lu
        for side in ("home_team", "away_team"):
            player_ids.update(p["id"] for p in (lu.get(side) or {}).get("lineups") or [])

    if with_profiles:
        for pid in sorted(player_ids):
            out["profiles"][str(pid)] = app.player_profile(pid)

    if app.has_token:
        out["odds"] = {}
        for mid in matches["match_id"]:
            try:
                out["odds"][str(mid)] = app.match_odds(int(mid))
            except Exception as exc:
                log.warning("odds for match %s failed: %s", mid, exc)
        try:
            out["injuries"] = app.injuries()
        except Exception as exc:
            log.warning("injuries failed: %s", exc)

    year = current_season_year()
    folder = SNAPSHOT_DIR / f"{year}-{str(year + 1)[-2:]}" / f"md{md['number']:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    log.info(
        "saved %s: matchday %s, %d matches, %d players, %s min before deadline%s",
        path, md["number"], len(matches), len(player_ids), out["minutes_to_deadline"],
        " (+odds, injuries)" if app.has_token else "",
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-profiles", action="store_true", help="skip per-player profile calls (ownership, injured flag)")
    ap.add_argument("--matchday", type=int, default=None, help="snapshot a specific matchday number")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    take_snapshot(with_profiles=not args.no_profiles, matchday=args.matchday)


if __name__ == "__main__":
    main()
