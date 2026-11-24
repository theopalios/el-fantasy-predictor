"""Load the game rules from config/game_rules.yaml.

The YAML file is the single source of truth: the scorer, backtester and optimizer read
every rule from here instead of hard-coding numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from elfantasy.paths import RULES_PATH


@dataclass(frozen=True)
class CoachBucket:
    result: str            # "win" | "loss"
    margin_min: int
    margin_max: int | None  # None = open-ended
    points: float

    def contains(self, abs_margin: int) -> bool:
        upper_ok = self.margin_max is None or abs_margin <= self.margin_max
        return abs_margin >= self.margin_min and upper_ok


@dataclass(frozen=True)
class Rules:
    raw: dict[str, Any] = field(repr=False)

    # squad / lineup
    players_per_position: dict[str, int]
    max_players_per_real_team: int
    coach_counts_toward_team_limit: bool
    formations: tuple[tuple[int, int, int], ...]
    multipliers: dict[str, float]
    captain_multiplier: float
    initial_budget: float

    # scoring
    player_weights: dict[str, float]
    win_bonus_pct: float
    win_bonus_mode: str  # "absolute": score + pct*|score| (verified); "multiplicative": score*(1+pct)
    coach_buckets: tuple[CoachBucket, ...]
    coach_overtime_points: dict[str, float]

    # trades
    trades_per_round: int
    unlimited_windows_after_rounds: tuple[int, ...]

    def formation_bounds(self) -> dict[str, tuple[int, int]]:
        """Min/max starters per position implied by the allowed formations."""
        g = [f[0] for f in self.formations]
        fw = [f[1] for f in self.formations]
        c = [f[2] for f in self.formations]
        return {"G": (min(g), max(g)), "F": (min(fw), max(fw)), "C": (min(c), max(c))}


_STAT_KEYS = (
    "points", "rebounds", "assists", "steals", "turnovers", "blocks_made",
    "blocks_received", "fouls_drawn", "fouls_committed", "missed_field_goals",
    "missed_free_throws",
)


def load_rules(path: str | Path = RULES_PATH) -> Rules:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    squad = raw["squad"]
    lineup = raw["lineup"]
    player_scoring = raw["scoring"]["player"]
    coach = raw["scoring"]["head_coach"]

    # Unknown rule details are resolved conservatively until verified against app data:
    # anything other than an explicit `false` means the coach counts toward the team limit.
    coach_counts = squad.get("coach_counts_toward_team_limit")

    return Rules(
        raw=raw,
        players_per_position=dict(squad["players_per_position"]),
        max_players_per_real_team=int(squad["max_players_per_real_team"]),
        coach_counts_toward_team_limit=coach_counts is not False,
        formations=tuple(tuple(f) for f in lineup["formations"]),
        multipliers={k: float(v) for k, v in lineup["score_multipliers"].items()},
        captain_multiplier=float(lineup["captain"]["multiplier"]),
        initial_budget=float(raw["budget"]["initial_credits"]),
        player_weights={k: float(player_scoring[k]) for k in _STAT_KEYS},
        win_bonus_pct=float(player_scoring["team_win_bonus_pct"]),
        win_bonus_mode=str(player_scoring.get("win_bonus_mode", "absolute")),
        coach_buckets=tuple(CoachBucket(**b) for b in coach["buckets"]),
        coach_overtime_points={k: float(v) for k, v in coach["overtime_points"].items()},
        trades_per_round=int(raw["trades"]["per_round"]),
        unlimited_windows_after_rounds=tuple(raw["trades"]["unlimited_windows_after_rounds"]),
    )


@lru_cache(maxsize=1)
def default_rules() -> Rules:
    return load_rules()
