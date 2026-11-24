"""Player points model: E[fantasy points | the player plays], gradient-boosted trees."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

# Feature groups (documented in docs/design.md §5). Ids, dates and targets are excluded.
FORM = ["fp_l1", "fp_l3", "fp_l5", "fp_l10", "pir_l3", "pir_l5", "pir_l10",
        "pir_ewm_h3", "pir_ewm_h8", "fp_ewm_h5", "fp_season_mean", "pir_season_mean",
        "pir_season_std", "gp_season", "career_games", "prev_fp_mean", "prev_pir_mean",
        "prev_gp", "prev_min_mean", "fg_pct_l5_vs_season", "fg3_pct_l5_vs_season", "pm_ewm_h8"]
ROLE = ["min_l3", "min_l10", "min_ewm_h5", "min_trend", "min_season_mean", "start_rate_l5",
        "usage_p36", "pir_p36_ewm", "vacated_min", "vacated_min_pos", "changed_team"]
RATES = [f"rate_{s}_p36" for s in ("pts", "oreb", "dreb", "ast", "stl", "blk", "tov", "fd", "pf",
                                    "missed_fg", "missed_ft", "blk_against")]
AVAIL = ["missed_streak", "dnp_rate_season", "first_game_back", "player_rest_days"]
TEAM = ["team_ortg", "team_drtg", "team_net", "team_pace", "team_fp_scored", "team_games_season",
        "elo", "elo_p_win", "elo_exp_margin"]
OPP = ["opp_ortg", "opp_drtg", "opp_net", "opp_pace", "opp_oreb_pct", "opp_dreb_pct",
       "opp_tov_forced_rate", "opp_stl_rate", "opp_pf_per40", "opp_blk_rate", "opp_fp_allowed",
       "opp_dvp_rel", "opp_vacated_min", "opp_elo"]
CONTEXT = ["is_home", "rest_days", "short_rest", "opp_rest_days", "is_playoffs"]
CATEGORICAL = ["position_name"]
FEATURES = FORM + ROLE + RATES + AVAIL + TEAM + OPP + CONTEXT + CATEGORICAL

# Tuned on 2023-24 (train 2016-17..2022-23): early stopping at ~380 trees. Squared error is
# kept on purpose: the optimizer needs EXPECTED points, which is what L2 estimates.
PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_estimators=400,
    verbose=-1,
)


def _matrix(X: pd.DataFrame) -> pd.DataFrame:
    M = X[FEATURES].copy()
    for c in ("changed_team", "short_rest"):
        M[c] = M[c].astype(float)
    M["position_name"] = pd.Categorical(M["position_name"], categories=["Guard", "Forward", "Center"])
    return M


@dataclass
class PlayerPointsModel:
    params: dict = field(default_factory=lambda: dict(PARAMS))
    model: lgb.LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame) -> "PlayerPointsModel":
        rows = train[train["y_played"].astype(bool)]
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(_matrix(rows), rows["y_fantasy_pts"].astype(float), categorical_feature=CATEGORICAL)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_matrix(X))

    def importance(self) -> pd.Series:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        s = pd.Series(gain, index=self.model.booster_.feature_name()).sort_values(ascending=False)
        return s / s.sum()
