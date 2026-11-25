import pandas as pd
import pytest

from elfantasy.weekly import _fmt, _resolve


@pytest.fixture
def pool():
    return pd.DataFrame({"name": ["Sasha Vezenkov", "Mike James", "Jimmy Clark III", "Mike Tobey"]},
                        index=[3001, 3765, 4884, 9000])


def test_resolve_by_id_full_name_and_unique_part(pool):
    assert _resolve(pool, 3765) == 3765
    assert _resolve(pool, "sasha vezenkov") == 3001
    assert _resolve(pool, "Clark") == 4884


def test_resolve_rejects_ambiguous_or_unknown(pool):
    with pytest.raises(KeyError, match="ambiguous"):
        _resolve(pool, "Mike")
    with pytest.raises(KeyError):
        _resolve(pool, "Nobody")


def test_fmt():
    assert _fmt(114.53) == "115"
    assert _fmt(0.68) == "0.68"
    assert _fmt(26.8) == "26.8"


def test_store_as_of_hides_later_games():
    from pathlib import Path

    from elfantasy.features.common import Store
    from elfantasy.weekly import _store_as_of

    store = Store.load(Path(__file__).parent / "fixtures" / "mini")
    cut = store.games.sort_values("date_utc")["date_utc"].iloc[24]
    past = _store_as_of(store, cut)
    assert (past.player_games["date_utc"] < cut).all()
    assert (past.team_games["date_utc"] < cut).all()
    later = past.games["date_utc"] >= cut
    assert not past.games.loc[later, "played"].any()
    assert past.games.loc[later, "home_score"].isna().all()
    assert past.games.loc[~later, "played"].all()
