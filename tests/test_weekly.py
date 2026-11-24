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
