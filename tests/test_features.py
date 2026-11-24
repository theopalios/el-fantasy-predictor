"""Leakage tests on a small committed sample (2016-17, rounds 1-6)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elfantasy.features.build import FeatureContext
from elfantasy.features.common import Store

FIXTURE = Path(__file__).parent / "fixtures" / "mini"
STAT_COLS = ["pts", "reb", "ast", "pir", "fantasy_pts", "dreb", "fd"]


def _ctx(store: Store) -> FeatureContext:
    return FeatureContext.from_store(store)


@pytest.fixture(scope="module")
def base():
    store = Store.load(FIXTURE)
    ctx = _ctx(store)
    return store, ctx, ctx.features(store.player_games)


def _perturbed_store(store: Store, gamecode: int) -> Store:
    pg = store.player_games.copy()
    mask = (pg["gamecode"] == gamecode) & pg["played"]
    pg.loc[mask, STAT_COLS] = pg.loc[mask, STAT_COLS] + 30
    tg = store.team_games.copy()
    tmask = tg["gamecode"] == gamecode
    tg.loc[tmask, ["team_pts", "team_fga2", "coach_pts"]] = tg.loc[tmask, ["team_pts", "team_fga2", "coach_pts"]] + 30
    return Store(games=store.games, player_games=pg, team_games=tg, people=store.people)


def _feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("player_code", "team", "opponent", "phase", "date_utc", "position_name")]


def test_features_of_a_game_ignore_that_game(base):
    store, _, X0 = base
    games = store.games.sort_values("date_utc")
    target = int(games[games["round"] == 4]["gamecode"].iloc[0])
    X1 = _ctx(_perturbed_store(store, target)).features(store.player_games)

    rows = store.player_games["gamecode"] == target
    cols = _feature_cols(X0)
    a = X0.loc[rows, cols].astype(float)
    b = X1.loc[rows, cols].astype(float)
    pd.testing.assert_frame_equal(a, b, check_exact=False)


def test_later_games_see_the_change(base):
    store, _, X0 = base
    games = store.games.sort_values("date_utc")
    target = int(games[games["round"] == 4]["gamecode"].iloc[0])
    X1 = _ctx(_perturbed_store(store, target)).features(store.player_games)

    pg = store.player_games
    players = pg[(pg["gamecode"] == target) & pg["played"]]["player_code"].unique()
    later = pg["player_code"].isin(players) & (pg["round"] == 5)
    assert later.any()
    assert (X1.loc[later, "fp_l1"] - X0.loc[later, "fp_l1"]).abs().max() > 0
    assert (X1.loc[later, "team_ortg"] - X0.loc[later, "team_ortg"]).abs().max() > 0


def test_sitting_out_does_not_change_own_features(base):
    """Regression test: a player's own absence must never appear in his own features
    (it used to leak through vacated_min into the availability model)."""
    store, _, X0 = base
    pg = store.player_games.copy()
    games = store.games.sort_values("date_utc")
    target = int(games[games["round"] == 4]["gamecode"].iloc[0])
    rows = pg.index[(pg["gamecode"] == target) & pg["played"]]
    # the player with the most minutes in that game "sits out"
    who = pg.loc[rows].sort_values("minutes").index[-1]
    pg.loc[who, ["minutes", "played"]] = [0.0, False]
    X1 = _ctx(Store(games=store.games, player_games=pg, team_games=store.team_games, people=store.people)).features(pg)
    cols = [c for c in _feature_cols(X0)]
    a = X0.loc[[who], cols].astype(float)
    b = X1.loc[[who], cols].astype(float)
    pd.testing.assert_frame_equal(a, b, check_exact=False)


def test_unplayed_games_are_not_missed_games(base):
    """A game that was never played (cancelled) must not count as a game a player missed."""
    store, _, X0 = base
    games = store.games.copy()
    pg = store.player_games
    order = games.sort_values("date_utc")
    cancelled = int(order[order["round"] == 3]["gamecode"].iloc[0])
    teams = set(order.loc[order["gamecode"] == cancelled, ["home_team", "away_team"]].iloc[0])
    games.loc[games["gamecode"] == cancelled, "played"] = False
    pg2 = pg[pg["gamecode"] != cancelled].copy()
    tg2 = store.team_games[store.team_games["gamecode"] != cancelled].copy()
    ctx = _ctx(Store(games=games, player_games=pg2, team_games=tg2, people=store.people))
    X1 = ctx.features(pg2)
    nxt = pg2["team"].isin(teams) & (pg2["round"] == 4) & pg2["played"]
    regulars = X1.loc[nxt & (X1["gp_season"] >= 2)]
    assert len(regulars) > 0
    assert (regulars["missed_streak"] == 0).all()


def test_first_round_has_no_history(base):
    store, _, X0 = base
    r1 = store.player_games["round"] == 1
    assert X0.loc[r1, "fp_ewm_h5"].isna().all()
    assert (X0.loc[r1, "gp_season"] == 0).all()


def test_backtest_candidate_pools(base):
    from elfantasy.backtest.walkforward import build_candidates

    store, ctx, _ = base
    cand = build_candidates(ctx, seasons=(2016,))
    pg = store.player_games
    assert (cand["round"] >= 2).all()
    # "played" pool = exactly the players who played that game
    played = cand[cand["pool_played"]]
    assert played["y_played"].all()
    n_played = pg[(pg["played"]) & (pg["round"] >= 2)].shape[0]
    assert len(played) == n_played
    # "prev" pool = players who played their team's previous game; those who then miss score 0
    missed = cand[cand["pool_prev"] & ~cand["y_played"]]
    assert (missed["y_fantasy_pts"] == 0).all()


def test_elo_is_pre_game_and_symmetric(base):
    _, ctx, _ = base
    e = ctx.elo
    g = e.groupby(["season", "gamecode"])
    assert np.allclose(g["elo_p_win"].sum(), 1.0)
    first = e[e["gamecode"] == e["gamecode"].min()]
    assert np.allclose(first["elo"], 1500.0)
