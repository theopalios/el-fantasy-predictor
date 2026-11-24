"""Synthetic fantasy prices for seasons without price history.

The game's price before round r tracks how a player has performed so far, so we learn
price = f(season-to-date form, previous season, minutes, career, position) on the real
2026-27 prices and apply it to earlier seasons. Coaches: price = linear in team Elo.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

PRICE_FEATURES = ["fp_season_mean", "gp_season", "prev_fp_mean", "prev_gp", "fp_ewm_h5",
                  "fp_l5", "min_ewm_h5", "career_games", "pos_code"]
POS_CODE = {"Guard": 0, "Forward": 1, "Center": 2}


def _m(rows: pd.DataFrame) -> pd.DataFrame:
    M = rows.copy()
    M["pos_code"] = M["position_name"].map(POS_CODE)
    return M[PRICE_FEATURES].astype(float)


@dataclass
class PriceModel:
    model: lgb.LGBMRegressor | None = None
    coach_coef: tuple[float, float] = (6.0, 0.0)
    floor: float = 4.0

    def fit(self, rows: pd.DataFrame, coach_rows: pd.DataFrame) -> "PriceModel":
        r = rows.dropna(subset=["price"])
        self.model = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.05,
                                       num_leaves=15, min_child_samples=30, verbose=-1)
        self.model.fit(_m(r), r["price"])
        self.floor = float(r["price"].min())
        c = coach_rows.dropna(subset=["price", "elo"])
        b, a = np.polyfit(c["elo"], c["price"], 1)
        self.coach_coef = (float(a), float(b))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(_m(rows)), self.floor).round(1)

    def predict_coach(self, elo: pd.Series) -> np.ndarray:
        a, b = self.coach_coef
        return np.clip(a + b * elo.to_numpy(float), 4.0, None).round(1)
