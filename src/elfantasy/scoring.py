"""Fantasy scoring: player points from box-score stats and coach points from the margin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from elfantasy.rules import Rules, default_rules

# Box-score column -> rules key. Missed shots are derived (attempts - makes).
_DIRECT_STATS = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "stl": "steals",
    "tov": "turnovers",
    "blk": "blocks_made",
    "blk_against": "blocks_received",
    "fd": "fouls_drawn",
    "pf": "fouls_committed",
}


def pir_from_stats(df: pd.DataFrame, rules: Rules | None = None) -> pd.Series:
    """Recompute PIR from the individual stat columns using the rule weights.

    Expects columns: pts, reb, ast, stl, tov, blk, blk_against, fd, pf,
    fgm2, fga2, fgm3, fga3, ftm, fta.
    """
    rules = rules or default_rules()
    w = rules.player_weights
    total = sum(w[key] * df[col] for col, key in _DIRECT_STATS.items())
    missed_fg = (df["fga2"] - df["fgm2"]) + (df["fga3"] - df["fgm3"])
    missed_ft = df["fta"] - df["ftm"]
    return total + w["missed_field_goals"] * missed_fg + w["missed_free_throws"] * missed_ft


def player_fantasy_points(pir, team_won, rules: Rules | None = None):
    """Fantasy points for a player who played: PIR plus the team-win bonus.

    The bonus is 10% of the absolute score (verified against the app), so in a win
    PIR 15 -> 16.5 and PIR -2 -> -1.8. Works on scalars or array-likes. Players who did
    not play score 0; handle that upstream.
    """
    rules = rules or default_rules()
    pir = np.asarray(pir, dtype=float)
    won = np.asarray(team_won, dtype=bool)
    pct = rules.win_bonus_pct / 100.0
    if rules.win_bonus_mode == "absolute":
        bonus = pct * np.abs(pir)
    elif rules.win_bonus_mode == "multiplicative":
        bonus = pct * pir
    else:
        raise ValueError(f"unknown win_bonus_mode {rules.win_bonus_mode!r}")
    out = np.where(won, pir + bonus, pir)
    return out.item() if out.ndim == 0 else out


def coach_points(margin: int, overtime: bool = False, rules: Rules | None = None) -> float:
    """Coach fantasy points from the final margin (coach's team perspective, + = win)."""
    rules = rules or default_rules()
    if margin == 0:
        raise ValueError("Basketball games cannot end tied")
    result = "win" if margin > 0 else "loss"
    if overtime:
        return rules.coach_overtime_points[result]
    abs_margin = abs(int(margin))
    for bucket in rules.coach_buckets:
        if bucket.result == result and bucket.contains(abs_margin):
            return bucket.points
    raise ValueError(f"No coach bucket for margin {margin}")


def coach_points_series(margin: pd.Series, overtime: pd.Series, rules: Rules | None = None) -> pd.Series:
    rules = rules or default_rules()
    return pd.Series(
        [coach_points(int(m), bool(o), rules) for m, o in zip(margin, overtime)],
        index=margin.index,
        dtype=float,
    )
