from pathlib import Path

import numpy as np
import pytest

from elfantasy.features.build import FeatureContext
from elfantasy.features.common import Store
from elfantasy.models.baselines import coach_points_from_margin
from elfantasy.models.game import GameModel, game_frame
from elfantasy.models.player_points import FEATURES, PlayerPointsModel

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


@pytest.fixture(scope="module")
def ctx():
    return FeatureContext.from_store(Store.load(FIXTURE))


def test_coach_points_from_margin_limits():
    big_win, big_loss, even = coach_points_from_margin(np.array([60.0, -60.0, 0.0]), 12.0)
    assert big_win == pytest.approx(25, abs=0.01)
    assert big_loss == pytest.approx(-20, abs=0.01)
    assert -5 < even < 10   # a coin flip lands between the close-loss and close-win values


def test_player_model_trains_and_predicts(ctx):
    rows = ctx.training_rows()
    assert set(FEATURES) <= set(rows.columns)
    m = PlayerPointsModel(params={**PlayerPointsModel().params, "n_estimators": 20, "min_child_samples": 10}).fit(rows)
    pred = m.predict(rows)
    assert len(pred) == len(rows) and np.isfinite(pred).all()
    imp = m.importance()
    assert imp.sum() == pytest.approx(1.0)


def test_game_model_outputs_are_sane(ctx):
    # The 48-game fixture is too small for meaningful coefficients (home advantage is
    # +5.2 points on the full data, see docs/backtest_models.md); check structure only.
    gf = game_frame(ctx)
    gm = GameModel().fit(gf)
    p = gm.p_win(gf)
    assert ((p > 0) & (p < 1)).all()
    assert 5 < gm.sigma < 25
    coach = gm.coach_points(gf)
    assert ((coach >= -20) & (coach <= 25)).all()
    # the two rows of a game are mirror images: their win probabilities sum to about 1
    s = gf.assign(p=p).groupby(["season", "gamecode"])["p"].sum()
    assert ((s > 0.7) & (s < 1.3)).all()
