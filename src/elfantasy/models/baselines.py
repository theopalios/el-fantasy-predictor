"""Baselines every model must beat. All predict E[fantasy points | the player plays]."""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import erf, sqrt

from elfantasy.rules import Rules, default_rules


def debut_fallback(train: pd.DataFrame) -> float:
    """Mean fantasy points of players in their first EuroLeague game (no history)."""
    rows = train[(train["y_played"]) & (train["career_games"] == 0)]
    return float(rows["y_fantasy_pts"].mean()) if len(rows) else 4.0


def season_average(X: pd.DataFrame, fallback: float) -> pd.Series:
    """Season-to-date average; previous season if no games yet; debut mean otherwise."""
    out = X["fp_season_mean"].where(X["gp_season"] > 0)
    out = out.fillna(X["prev_fp_mean"]).fillna(fallback)
    return out.rename("pred_season_avg")


def ewma(X: pd.DataFrame, fallback: float) -> pd.Series:
    """Exponentially weighted recent form (half-life 5 games, carried across seasons)."""
    return X["fp_ewm_h5"].fillna(X["prev_fp_mean"]).fillna(fallback).rename("pred_ewma")


def fit_price_line(train: pd.DataFrame) -> tuple[float, float]:
    """OLS fantasy_pts ~ price on rows where the player played."""
    t = train.dropna(subset=["price", "y_fantasy_pts"])
    t = t[t["y_played"]]
    if len(t) < 20:
        return np.nan, np.nan
    b, a = np.polyfit(t["price"].to_numpy(float), t["y_fantasy_pts"].to_numpy(float), 1)
    return float(a), float(b)


def price_implied(X: pd.DataFrame, a: float, b: float) -> pd.Series:
    return (a + b * X["price"]).rename("pred_price")


# --------------------------------------------------------------------------- coaches

def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def coach_points_from_margin(mu: np.ndarray, sigma: float, rules: Rules | None = None) -> np.ndarray:
    """Expected coach points if the margin ~ Normal(mu, sigma), using the rules' buckets.
    Integer margins: a bucket [lo, hi] covers (lo - 0.5, hi + 0.5)."""
    rules = rules or default_rules()
    mu = np.asarray(mu, dtype=float)
    out = np.zeros_like(mu)
    for bkt in rules.coach_buckets:
        lo, hi = bkt.margin_min - 0.5, (bkt.margin_max + 0.5) if bkt.margin_max is not None else np.inf
        if bkt.result == "loss":
            lo, hi = -hi, -lo
        p = np.array([_phi((hi - m) / sigma) - _phi((lo - m) / sigma) for m in mu])
        out += p * bkt.points
    return out


def fit_margin_sigma(elo_rows: pd.DataFrame, margins: pd.Series) -> float:
    resid = margins.to_numpy(float) - elo_rows["elo_exp_margin"].to_numpy(float)
    return float(np.nanstd(resid))
