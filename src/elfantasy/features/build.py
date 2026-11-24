"""Assemble the full feature frame for any set of (player, team, game) queries.

A query row needs: player_code, team, opponent, season, gamecode, round, phase, date_utc,
is_home, position_name. The same code builds training rows (actual box-score rows),
backtest candidates and, later, live predictions, so train and serve cannot drift apart.

    ctx = FeatureContext.from_store(Store.load())
    X = ctx.features(queries)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from elfantasy.features.common import Store
from elfantasy.features.player import (
    availability_features,
    player_features,
    player_state,
    prev_season_table,
    team_game_index,
    vacated_minutes,
)
from elfantasy.features.team import (
    POSITIONS,
    elo_table,
    league_dvp_state,
    rest_table,
    team_features,
    team_game_metrics,
    team_state,
)
from elfantasy.features.common import asof_join

QUERY_COLS = ["player_code", "team", "opponent", "season", "gamecode", "round", "phase",
              "date_utc", "is_home", "position_name"]


@dataclass
class FeatureContext:
    store: Store
    pstate: pd.DataFrame
    prev: pd.DataFrame
    tgi: pd.DataFrame
    tstate: pd.DataFrame
    lg_dvp: pd.DataFrame
    elo: pd.DataFrame
    rest: pd.DataFrame
    vacated: pd.DataFrame

    @classmethod
    def from_store(cls, store: Store) -> "FeatureContext":
        pg, tg, games = store.player_games, store.team_games, store.games
        pstate = player_state(pg)
        tgi = team_game_index(games)
        metrics = team_game_metrics(tg, pg)
        return cls(
            store=store,
            pstate=pstate,
            prev=prev_season_table(pg),
            tgi=tgi,
            tstate=team_state(metrics),
            lg_dvp=league_dvp_state(metrics),
            elo=elo_table(games),
            rest=rest_table(games),
            vacated=vacated_minutes(pg, tgi, pstate),
        )

    # ------------------------------------------------------------------ public
    def features(self, queries: pd.DataFrame) -> pd.DataFrame:
        q = queries[QUERY_COLS].copy()
        q["date_utc"] = pd.to_datetime(q["date_utc"], utc=True)
        parts = [
            player_features(q, self.pstate, self.prev),
            availability_features(q, self.store.player_games, self.tgi),
            team_features(q, self.tstate, "team"),
            team_features(q, self.tstate, "opponent"),
        ]
        X = pd.concat([q] + parts, axis=1)

        # defence vs position: what this opponent allows to the player's position, relative to league
        lg = asof_join(q, self.lg_dvp, by=None)
        for p in POSITIONS:
            X[f"opp_dvp_rel_{p}"] = X[f"opp_dvp_{p}"] / lg[f"lg_dvp_{p}"]
        pos = X["position_name"].fillna("Forward")
        X["opp_dvp_rel"] = np.select([pos == p for p in POSITIONS],
                                     [X[f"opp_dvp_rel_{p}"] for p in POSITIONS], np.nan)

        key = ["season", "gamecode", "team"]
        X = self._merge(X, self.elo, key)
        X = self._merge(X, self.rest, key)
        opp_rest = self.rest.rename(columns={"team": "opponent", "rest_days": "opp_rest_days",
                                             "short_rest": "opp_short_rest"})
        X = self._merge(X, opp_rest, ["season", "gamecode", "opponent"])

        X = self._merge(X, self.vacated, key)
        X["vacated_min"] = X["vacated_min"].fillna(0.0)
        for p in POSITIONS:
            col = f"vacated_min_{p}"
            if col not in X:
                X[col] = 0.0
            X[col] = X[col].fillna(0.0)
        X["vacated_min_pos"] = np.select([pos == p for p in POSITIONS],
                                         [X[f"vacated_min_{p}"] for p in POSITIONS], 0.0)
        opp_vac = self.vacated[["season", "gamecode", "team", "vacated_min"]].rename(
            columns={"team": "opponent", "vacated_min": "opp_vacated_min"})
        X = self._merge(X, opp_vac, ["season", "gamecode", "opponent"])
        X["opp_vacated_min"] = X["opp_vacated_min"].fillna(0.0)

        X = X.copy()  # de-fragment after many column inserts
        X["is_home"] = X["is_home"].astype(float)
        X["is_playoffs"] = X["phase"].isin(["PI", "PO", "FF"]).astype(float)
        return X

    def training_rows(self) -> pd.DataFrame:
        """Feature frame for every dressed player in every played game, with targets."""
        pg = self.store.player_games
        X = self.features(pg)
        targets = pd.DataFrame(
            {f"y_{c}": pg[c].to_numpy() for c in ("fantasy_pts", "pir", "minutes", "played", "is_starter", "win")},
            index=X.index,
        )
        return pd.concat([X, targets], axis=1)

    @staticmethod
    def _merge(X: pd.DataFrame, table: pd.DataFrame, on: list[str]) -> pd.DataFrame:
        idx = X.index
        out = X.merge(table.drop_duplicates(on), on=on, how="left")
        out.index = idx
        return out
