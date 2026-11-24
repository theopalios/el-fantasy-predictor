"""Availability model: P(plays this game | played the team's previous game), from box-score
history only. At prediction time the app's own probability of playing overrides it for
players the app flags (design §6.2)."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

AVAIL_FEATURES = ["dnp_rate_season", "min_ewm_h5", "min_l3", "min_trend", "start_rate_l5",
                  "career_games", "gp_season", "player_rest_days", "fp_ewm_h5", "rest_days",
                  "is_playoffs", "prev_gp", "team_games_season",
                  "vacated_min"]   # minutes of missing TEAMMATES (the player's own minutes are excluded)

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=200,
              n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, deterministic=True, force_row_wise=True, verbose=-1)


@dataclass
class AvailabilityModel:
    model: lgb.LGBMClassifier | None = None

    def fit(self, rows: pd.DataFrame) -> "AvailabilityModel":
        self.model = lgb.LGBMClassifier(**PARAMS)
        self.model.fit(rows[AVAIL_FEATURES].astype(float), rows["y_played"].astype(int))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(rows[AVAIL_FEATURES].astype(float))[:, 1]
