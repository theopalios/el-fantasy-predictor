"""Public EuroLeague stats API: schedule/results, box scores and club rosters.

Endpoints (no authentication):
  v2 games   https://api-live.euroleague.net/v2/competitions/E/seasons/E2025/games
  v2 people  https://api-live.euroleague.net/v2/competitions/E/seasons/E2025/clubs/{club}/people
  boxscore   https://live.euroleague.net/api/Boxscore?gamecode=1&seasoncode=E2025
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from elfantasy.data.http import HttpClient

V2 = "https://api-live.euroleague.net/v2"
LIVE = "https://live.euroleague.net/api"


def season_code(year: int, competition: str = "E") -> str:
    """Season starting in `year`, e.g. 2026 -> 'E2026' (2026-27)."""
    return f"{competition}{year}"


class EuroLeagueAPI:
    def __init__(self, http: HttpClient, competition: str = "E") -> None:
        self.http = http
        self.competition = competition

    def games(self, year: int, cache: bool = False) -> list[dict[str, Any]]:
        url = f"{V2}/competitions/{self.competition}/seasons/{season_code(year, self.competition)}/games"
        data = self.http.get_json(url, cache=cache)
        return data["data"] if isinstance(data, dict) else data

    def game_stats(self, year: int, gamecode: int, cache: bool = True) -> dict[str, Any]:
        """v2 per-game stats: ~4x faster than the legacy Boxscore and carries roster codes."""
        url = (
            f"{V2}/competitions/{self.competition}/seasons/"
            f"{season_code(year, self.competition)}/games/{gamecode}/stats"
        )
        return self.http.get_json(url, cache=cache)

    def boxscore(self, year: int, gamecode: int, cache: bool = True) -> dict[str, Any]:
        params = {"gamecode": gamecode, "seasoncode": season_code(year, self.competition)}
        return self.http.get_json(f"{LIVE}/Boxscore", params=params, cache=cache)

    def club_people(self, year: int, club: str, cache: bool = False) -> list[dict[str, Any]]:
        url = (
            f"{V2}/competitions/{self.competition}/seasons/"
            f"{season_code(year, self.competition)}/clubs/{club}/people"
        )
        data = self.http.get_json(url, cache=cache)
        return data["data"] if isinstance(data, dict) and "data" in data else data


# --------------------------------------------------------------------------- parsers

def _n_extra_periods(partials: dict[str, Any] | None) -> int:
    if not partials:
        return 0
    extra = partials.get("extraPeriods") or {}
    return sum(1 for v in extra.values() if v is not None)


def parse_games(raw: list[dict[str, Any]], year: int) -> pd.DataFrame:
    rows = []
    for g in raw:
        local, road = g.get("local") or {}, g.get("road") or {}
        rows.append(
            {
                "season": year,
                "gamecode": int(g["gameCode"]),
                "round": int(g["round"]) if g.get("round") is not None else None,
                "phase": (g.get("phaseType") or {}).get("code"),
                "date_utc": pd.to_datetime(g.get("utcDate") or g.get("date"), utc=True),
                "played": bool(g.get("played")),
                "home_team": (local.get("club") or {}).get("code"),
                "away_team": (road.get("club") or {}).get("code"),
                "home_score": local.get("score"),
                "away_score": road.get("score"),
                "n_ot": max(_n_extra_periods(local.get("partials")), _n_extra_periods(road.get("partials"))),
                "neutral_venue": bool(g.get("isNeutralVenue")),
                "venue_code": (g.get("venue") or {}).get("code"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["season", "gamecode"]).reset_index(drop=True)


def parse_minutes(value: Any) -> float:
    """'33:21' -> 33.35; 'DNP'/None/'' -> 0.0."""
    if value is None:
        return 0.0
    s = str(value).strip()
    m = re.fullmatch(r"(\d+):(\d{1,2})", s)
    if not m:
        return 0.0
    return int(m.group(1)) + int(m.group(2)) / 60.0


_BOX_COLS = {
    "Points": "pts",
    "FieldGoalsMade2": "fgm2",
    "FieldGoalsAttempted2": "fga2",
    "FieldGoalsMade3": "fgm3",
    "FieldGoalsAttempted3": "fga3",
    "FreeThrowsMade": "ftm",
    "FreeThrowsAttempted": "fta",
    "OffensiveRebounds": "oreb",
    "DefensiveRebounds": "dreb",
    "TotalRebounds": "reb",
    "Assistances": "ast",
    "Steals": "stl",
    "Turnovers": "tov",
    "BlocksFavour": "blk",
    "BlocksAgainst": "blk_against",
    "FoulsCommited": "pf",
    "FoulsReceived": "fd",
    "Valuation": "pir",
    "Plusminus": "plus_minus",
}


def normalize_player_code(player_id: str) -> str:
    """Box-score ids look like 'P007200   '; roster codes look like '007200'."""
    return re.sub(r"^P", "", (player_id or "").strip())


def parse_boxscore(raw: dict[str, Any], year: int, gamecode: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player rows, team rows) for one game."""
    players, teams = [], []
    for side in raw.get("Stats") or []:
        team_code = None
        for p in side.get("PlayersStats") or []:
            team_code = (p.get("Team") or "").strip() or team_code
            row = {
                "season": year,
                "gamecode": gamecode,
                "team": (p.get("Team") or "").strip(),
                "player_code": normalize_player_code(p.get("Player_ID")),
                "player_name": (p.get("Player") or "").strip(),
                "dorsal": (p.get("Dorsal") or "").strip(),
                "is_starter": bool(p.get("IsStarter")),
                "minutes": parse_minutes(p.get("Minutes")),
                "dnp": str(p.get("Minutes") or "").strip().upper() == "DNP",
            }
            for src, dst in _BOX_COLS.items():
                row[dst] = p.get(src)
            players.append(row)
        tot = side.get("totr") or {}
        tmr = side.get("tmr") or {}
        teams.append(
            {
                "season": year,
                "gamecode": gamecode,
                "team": team_code,
                "team_name": side.get("Team"),
                "coach_name": side.get("Coach"),
                **{f"team_{dst}": tot.get(src) for src, dst in _BOX_COLS.items() if src != "Plusminus"},
                "team_reb_team": tmr.get("TotalRebounds"),
            }
        )
    pdf = pd.DataFrame(players)
    if not pdf.empty:
        stat_cols = [c for c in _BOX_COLS.values()]
        pdf[stat_cols] = pdf[stat_cols].apply(pd.to_numeric, errors="coerce")
        pdf[[c for c in stat_cols if c != "plus_minus"]] = (
            pdf[[c for c in stat_cols if c != "plus_minus"]].fillna(0).astype(int)
        )
    return pdf, pd.DataFrame(teams)


_V2_COLS = {
    "points": "pts",
    "fieldGoalsMade2": "fgm2",
    "fieldGoalsAttempted2": "fga2",
    "fieldGoalsMade3": "fgm3",
    "fieldGoalsAttempted3": "fga3",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "offensiveRebounds": "oreb",
    "defensiveRebounds": "dreb",
    "totalRebounds": "reb",
    "assistances": "ast",
    "steals": "stl",
    "turnovers": "tov",
    "blocksFavour": "blk",
    "blocksAgainst": "blk_against",
    "foulsCommited": "pf",
    "foulsReceived": "fd",
    "valuation": "pir",
    "plusMinus": "plus_minus",
}


def parse_game_stats(
    raw: dict[str, Any], year: int, gamecode: int, home_team: str, away_team: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the v2 /games/{code}/stats payload into (player rows, team rows).

    Same columns as `parse_boxscore`, plus position_name and coach_code."""
    players, teams = [], []
    for side, team_code in (("local", home_team), ("road", away_team)):
        block = raw.get(side) or {}
        coach = block.get("coach") or {}
        for p in block.get("players") or []:
            meta = p.get("player") or {}
            person = meta.get("person") or {}
            st = p.get("stats") or {}
            seconds = st.get("timePlayed") or 0.0
            row = {
                "season": year,
                "gamecode": gamecode,
                "team": team_code,
                "player_code": (person.get("code") or "").strip(),
                "player_name": (person.get("name") or "").strip(),
                "position_name": meta.get("positionName"),
                "dorsal": str(st.get("dorsal") or meta.get("dorsal") or ""),
                "is_starter": bool(st.get("startFive")),
                "minutes": float(seconds) / 60.0,
                "dnp": float(seconds) == 0.0,
            }
            for src, dst in _V2_COLS.items():
                row[dst] = st.get(src)
            players.append(row)
        tot = block.get("total") or {}
        tm = block.get("team") or {}
        teams.append(
            {
                "season": year,
                "gamecode": gamecode,
                "team": team_code,
                "coach_code": coach.get("code"),
                "coach_name": coach.get("name"),
                **{f"team_{dst}": tot.get(src) for src, dst in _V2_COLS.items() if src != "plusMinus"},
                "team_reb_team": tm.get("totalRebounds"),
            }
        )
    pdf = pd.DataFrame(players)
    if not pdf.empty:
        ints = [c for c in _V2_COLS.values() if c != "plus_minus"]
        pdf[ints] = pdf[ints].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        pdf["plus_minus"] = pd.to_numeric(pdf["plus_minus"], errors="coerce")
    return pdf, pd.DataFrame(teams)


def parse_people(raw: list[dict[str, Any]], year: int, club: str) -> pd.DataFrame:
    rows = []
    for x in raw or []:
        person = x.get("person") or {}
        rows.append(
            {
                "season": year,
                "club": club,
                "person_code": (person.get("code") or "").strip(),
                "name": person.get("name"),
                "type": x.get("type"),            # J = player, E = head coach, A = assistant
                "type_name": x.get("typeName"),
                "position_name": x.get("positionName"),  # Guard / Forward / Center
                "height_cm": person.get("height") or np.nan,
                "birth_date": pd.to_datetime(person.get("birthDate"), errors="coerce"),
                "country": (person.get("country") or {}).get("code"),
                "start_date": pd.to_datetime(x.get("startDate"), errors="coerce"),
                "end_date": pd.to_datetime(x.get("endDate"), errors="coerce"),
                "active": bool(x.get("active")),
                "dorsal": x.get("dorsal"),
            }
        )
    return pd.DataFrame(rows)


def add_game_context(player_games: pd.DataFrame, team_games: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach opponent, home flag, scores, margin and win to player and team rows."""
    g = games[["season", "gamecode", "round", "phase", "date_utc", "home_team", "away_team",
               "home_score", "away_score", "n_ot"]]

    def _ctx(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(g, on=["season", "gamecode"], how="left", validate="many_to_one")
        out["is_home"] = out["team"] == out["home_team"]
        out["opponent"] = np.where(out["is_home"], out["away_team"], out["home_team"])
        out["team_score"] = np.where(out["is_home"], out["home_score"], out["away_score"])
        out["opp_score"] = np.where(out["is_home"], out["away_score"], out["home_score"])
        out["margin"] = out["team_score"] - out["opp_score"]
        out["win"] = out["margin"] > 0
        return out.drop(columns=["home_team", "away_team", "home_score", "away_score"])

    return _ctx(player_games), _ctx(team_games)
