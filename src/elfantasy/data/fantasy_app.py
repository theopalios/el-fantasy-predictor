"""EuroLeague Fantasy Challenge app backend (fantaking / Dunkest).

Public endpoints (no login), checked 24 Nov 2026:
  /leagues/{league}/config                         matchday ids, teams, game-mode config
  /schedules/{schedule}/matchdays/{matchday}       matches of a round (ids, start, score, OT)
  /matches/{match}/lineups                         every player: fantasy position, price, projected
                                                   starter/bench/out, probability of playing
  /basketball/leagues/{league}/players/{id}/match-log/rendered
                                                   per round: price before round (CR), price change
                                                   (PLUS), fantasy points (FPT), minutes, box score
  /players/{id}/trend?league=                      fantasy points per round (players and coaches)
  /players/{id}/profile?league=                    price, popularity (ownership), injured flag

Login-only endpoints (Bearer token in ELF_FANTASY_TOKEN):
  /odds/matches/{match}, /leagues/{league}/injuries, /players-lists/.../players
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from elfantasy.data.http import HttpClient

BASE = "https://fantaking-api.dunkest.com/api/v1"
LEAGUE_ID = 10  # EuroLeague Fantasy Challenge
TOKEN_ENV = "ELF_FANTASY_TOKEN"


class FantasyAppAPI:
    def __init__(self, http: HttpClient, league_id: int = LEAGUE_ID, token: str | None = None) -> None:
        load_dotenv()
        self.http = http
        self.league_id = league_id
        self.token = token or os.environ.get(TOKEN_ENV) or None

    # ---------------------------------------------------------------- helpers
    def _get(self, path: str, params: dict[str, Any] | None = None, cache: bool = False, auth: bool = False) -> Any:
        headers = None
        if auth:
            if not self.token:
                raise PermissionError(f"{path} needs a login token in ${TOKEN_ENV}")
            headers = {"Authorization": f"Bearer {self.token}"}
        data = self.http.get_json(f"{BASE}{path}", params=params, cache=cache, headers=headers)
        return data.get("data", data) if isinstance(data, dict) else data

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    # ---------------------------------------------------------- public calls
    def config(self) -> dict[str, Any]:
        return self._get(f"/leagues/{self.league_id}/config")

    def schedule_matchday(self, schedule_id: int, matchday_id: int, cache: bool = False) -> dict[str, Any]:
        return self._get(f"/schedules/{schedule_id}/matchdays/{matchday_id}", cache=cache)

    def match_lineups(self, match_id: int) -> dict[str, Any]:
        return self._get(f"/matches/{match_id}/lineups")

    def player_match_log(self, player_id: int) -> dict[str, Any]:
        return self._get(f"/basketball/leagues/{self.league_id}/players/{player_id}/match-log/rendered")

    def player_trend(self, player_id: int) -> dict[str, Any]:
        return self._get(f"/players/{player_id}/trend", params={"league": self.league_id})

    def player_profile(self, player_id: int) -> dict[str, Any]:
        return self._get(f"/players/{player_id}/profile", params={"league": self.league_id})

    # ----------------------------------------------------------- login calls
    def match_odds(self, match_id: int) -> Any:
        return self._get(f"/odds/matches/{match_id}", auth=True)

    def injuries(self) -> Any:
        return self._get(f"/leagues/{self.league_id}/injuries", auth=True)


# --------------------------------------------------------------------------- parsers

def parse_schedule(raw: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rnd in raw.get("rounds") or []:
        for m in rnd.get("matches") or []:
            home, away = m.get("home_team") or {}, m.get("away_team") or {}
            rows.append(
                {
                    "matchday_id": raw.get("id"),
                    "matchday": raw.get("number"),
                    "turn_id": rnd.get("id"),
                    "turn": rnd.get("number"),
                    "match_id": m.get("id"),
                    "status": m.get("status"),
                    "ot": bool(m.get("ot")),
                    "started_at": pd.to_datetime(m.get("started_at"), utc=True),
                    "home_team_id": home.get("id"),
                    "home_abbr": home.get("abbreviation"),
                    "home_score": home.get("score"),
                    "away_team_id": away.get("id"),
                    "away_abbr": away.get("abbreviation"),
                    "away_score": away.get("score"),
                }
            )
    return pd.DataFrame(rows)


def parse_lineups(raw: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for side in ("home_team", "away_team"):
        team = raw.get(side) or {}
        for p in team.get("lineups") or []:
            rows.append(
                {
                    "match_id": raw.get("id"),
                    "match_status": raw.get("status"),
                    "team_id": team.get("id"),
                    "team_abbr": team.get("abbreviation"),
                    "is_home": side == "home_team",
                    "player_id": p.get("id"),
                    "first_name": p.get("first_name"),
                    "last_name": p.get("last_name"),
                    "position": p.get("position"),          # Guard / Forward / Center / Head Coach
                    "price": p.get("quotation"),
                    "lineup_status": p.get("status"),       # starter / bench / out
                    "p_play": p.get("probability_of_playing"),
                    "is_on_fire": p.get("is_on_fire"),
                    "fantasy_pts": p.get("pts"),
                }
            )
    return pd.DataFrame(rows)


def _num(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace("+", "")
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _minutes(x: Any) -> float:
    m = re.fullmatch(r"(\d+):(\d{1,2})", str(x or "").strip())
    return int(m.group(1)) + int(m.group(2)) / 60.0 if m else 0.0


_LOG_RENAME = {
    "MD": "matchday", "CR": "price", "PLUS": "price_change", "FPT": "fantasy_pts", "ST": "started",
    "PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "BA": "blk_against",
    "FGM": "fgm", "FGA": "fga", "3PM": "fgm3", "3PA": "fga3", "FTM": "ftm", "FTA": "fta",
    "OREB": "oreb", "DREB": "dreb", "TOV": "tov", "PF": "pf", "FD": "fd", "+/-": "plus_minus",
}


def parse_match_log(raw: dict[str, Any], player_id: int) -> pd.DataFrame:
    """One row per matchday. `price` is the price going INTO the round, `price_change`
    the move applied after it (verified: CR[md] + PLUS[md] == CR[md+1])."""
    cols = raw.get("columns") or []
    rows = []
    for r in raw.get("rows") or []:
        rec = dict(zip(cols, r))
        vs = str(rec.get("VS") or "")
        out: dict[str, Any] = {
            "player_id": player_id,
            "is_home": not vs.startswith("@"),
            "opp_abbr": vs.replace("@", "").replace("VS", "").strip(),
            "minutes": _minutes(rec.get("MIN")),
        }
        for src, dst in _LOG_RENAME.items():
            if src in rec:
                out[dst] = rec[src] if src == "ST" else _num(rec[src])
        if "started" in out:
            out["started"] = str(out["started"]).upper() == "Y"
        rows.append(out)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["matchday"] = df["matchday"].astype(int)
    return df


def parse_profile(raw: dict[str, Any]) -> dict[str, Any]:
    pos = raw.get("position")
    team = raw.get("team") or {}
    return {
        "player_id": raw.get("id"),
        "first_name": raw.get("first_name"),
        "last_name": raw.get("last_name"),
        "position": pos.get("name") if isinstance(pos, dict) else pos,
        "team_id": team.get("id"),
        "team_abbr": team.get("abbreviation"),
        "price": raw.get("quotation"),
        "avg_fantasy_pts": raw.get("avg_fantasy_pts"),
        "total_price_change": raw.get("total_plus"),
        "popularity": raw.get("popularity"),
        "is_on_fire": raw.get("is_on_fire"),
        "is_injured": raw.get("is_injured"),
        "started_from_bench": raw.get("started_from_bench"),
    }
