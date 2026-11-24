"""Game model: expected margin -> P(win) and expected coach points.

margin ~ Normal(mu, sigma), mu = linear model on Elo difference, team rating gap, rest gap
and absences. Trained on team-perspective rows of past games.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from elfantasy.features.build import FeatureContext
from elfantasy.features.team import team_features
from elfantasy.models.baselines import coach_points_from_margin

GAME_FEATURES = ["elo_diff", "net_diff", "rest_diff", "vac_diff", "is_home"]


def game_frame(ctx: FeatureContext) -> pd.DataFrame:
    """One row per team per played game with pre-game features and outcomes."""
    tg = ctx.store.team_games
    base = tg[["season", "gamecode", "team", "opponent", "date_utc", "is_home", "margin", "n_ot",
               "coach_pts", "phase", "round"]].copy()
    return game_features(ctx, base, ctx.vacated)


def game_features(ctx: FeatureContext, base: pd.DataFrame, vacated: pd.DataFrame) -> pd.DataFrame:
    """Pre-game features for team-perspective rows (season, gamecode, team, opponent,
    date_utc, is_home). `vacated` has season, gamecode, team, vacated_min (actual absences
    for past games; the app's injury list for upcoming ones)."""
    g = base.copy()
    tf = team_features(g, ctx.tstate, "team")
    of = team_features(g, ctx.tstate, "opponent")
    g["net_diff"] = (tf["team_net"] - of["opp_net"]).fillna(0.0).to_numpy()
    key = ["season", "gamecode", "team"]
    g = g.merge(ctx.elo[key + ["elo", "elo_diff", "elo_exp_margin", "elo_p_win"]], on=key, how="left")
    g = g.merge(ctx.rest[key + ["rest_days"]], on=key, how="left")
    opp_rest = ctx.rest.rename(columns={"team": "opponent", "rest_days": "opp_rest_days"})
    g = g.merge(opp_rest[["season", "gamecode", "opponent", "opp_rest_days"]], on=["season", "gamecode", "opponent"], how="left")
    g["rest_diff"] = (g["rest_days"].clip(upper=7) - g["opp_rest_days"].clip(upper=7)).fillna(0.0)
    vac = vacated[key + ["vacated_min"]]
    g = g.merge(vac, on=key, how="left")
    g = g.merge(vac.rename(columns={"team": "opponent", "vacated_min": "opp_vacated_min"}),
                on=["season", "gamecode", "opponent"], how="left")
    g["vac_diff"] = g["vacated_min"].fillna(0.0) - g["opp_vacated_min"].fillna(0.0)
    g["is_home"] = g["is_home"].astype(float)
    g["elo_diff"] = g["elo_diff"].fillna(0.0)
    return g


@dataclass
class GameModel:
    alpha: float = 10.0
    reg: Ridge | None = None
    sigma: float = 12.0

    def fit(self, rows: pd.DataFrame) -> "GameModel":
        r = rows.dropna(subset=["margin"])
        self.reg = Ridge(alpha=self.alpha).fit(r[GAME_FEATURES], r["margin"])
        self.sigma = float(np.std(r["margin"] - self.reg.predict(r[GAME_FEATURES])))
        return self

    def expected_margin(self, rows: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(rows[GAME_FEATURES])

    def p_win(self, rows: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.expected_margin(rows) / self.sigma)

    def coach_points(self, rows: pd.DataFrame) -> np.ndarray:
        return coach_points_from_margin(self.expected_margin(rows), self.sigma)

    def coefficients(self) -> pd.Series:
        return pd.Series(self.reg.coef_, index=GAME_FEATURES)
