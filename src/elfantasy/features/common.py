"""Shared helpers: loading the processed tables and as-of joins."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from elfantasy.paths import PROCESSED_DIR


@dataclass
class Store:
    games: pd.DataFrame
    player_games: pd.DataFrame
    team_games: pd.DataFrame
    people: pd.DataFrame

    @classmethod
    def load(cls, directory=PROCESSED_DIR) -> "Store":
        def rd(name: str) -> pd.DataFrame:
            return pd.read_parquet(directory / f"{name}.parquet")

        games = rd("games")
        pg = rd("player_games")
        tg = rd("team_games")
        people = rd("people")
        for df in (games, pg, tg):
            df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
        # The Final Four is played at a neutral venue even if the API does not flag it.
        games["neutral_venue"] = games["neutral_venue"].astype(bool) | (games["phase"] == "FF")
        # Deterministic row order: downloads finish in arbitrary order, and model training
        # (bagging) depends on row order, so the same data must always give the same model.
        games = games.sort_values(["season", "gamecode"]).reset_index(drop=True)
        pg = pg.sort_values(["season", "gamecode", "team", "player_code"]).reset_index(drop=True)
        tg = tg.sort_values(["season", "gamecode", "team"]).reset_index(drop=True)
        people = people.sort_values(["season", "club", "person_code"]).reset_index(drop=True)
        return cls(games=games, player_games=pg, team_games=tg, people=people)


def asof_join(
    queries: pd.DataFrame, state: pd.DataFrame, by: str | list[str] | None, prefix: str = ""
) -> pd.DataFrame:
    """For each query row, attach the latest `state` row strictly BEFORE the query date.

    `state` holds post-game values (including that game) keyed by `date_utc`; taking the
    last one strictly earlier than the query's date means only past games are used.
    Returns the state columns aligned to `queries.index`.
    """
    by_cols = [by] if isinstance(by, str) else (by or [])
    q = queries[by_cols + ["date_utc"]].copy()
    q["_row"] = range(len(q))
    q = q.sort_values("date_utc")
    s = state.sort_values("date_utc")
    value_cols = [c for c in s.columns if c not in by_cols + ["date_utc"]]
    merged = pd.merge_asof(
        q, s, on="date_utc", by=by_cols or None, direction="backward", allow_exact_matches=False
    )
    merged = merged.sort_values("_row")
    out = merged[value_cols].reset_index(drop=True)
    out.index = queries.index
    if prefix:
        out = out.add_prefix(prefix)
    return out


def ewm_by(df: pd.DataFrame, group: str | list[str], col: str, halflife: float) -> pd.Series:
    """EWMA of `col` within groups (df must be sorted by time within group)."""
    return df.groupby(group, sort=False)[col].transform(lambda s: s.ewm(halflife=halflife, min_periods=1).mean())


def roll_by(df: pd.DataFrame, group: str | list[str], col: str, window: int, func: str = "mean") -> pd.Series:
    return df.groupby(group, sort=False)[col].transform(lambda s: getattr(s.rolling(window, min_periods=1), func)())
