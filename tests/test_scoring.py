import pandas as pd
import pytest

from elfantasy.rules import default_rules
from elfantasy.scoring import coach_points, pir_from_stats, player_fantasy_points


def _row(**kw):
    base = dict(pts=0, reb=0, ast=0, stl=0, tov=0, blk=0, blk_against=0, fd=0, pf=0,
                fgm2=0, fga2=0, fgm3=0, fga3=0, ftm=0, fta=0)
    base.update(kw)
    return pd.DataFrame([base])


def test_pir_matches_official_valuation_larkin():
    # Shane Larkin, 2025-26 R1 box score: official Valuation = 12.
    df = _row(pts=14, reb=4, ast=5, stl=0, tov=2, blk=0, blk_against=1, fd=4, pf=3,
              fgm2=5, fga2=8, fgm3=0, fga3=4, ftm=4, fta=6)
    assert pir_from_stats(df).iloc[0] == 12


def test_win_bonus_positive_score():
    # Kabengele, 2026-27 MD13: PIR 15 in a win -> 16.5 in the app.
    assert player_fantasy_points(15, True) == pytest.approx(16.5)
    assert player_fantasy_points(15, False) == pytest.approx(15)


def test_win_bonus_is_on_absolute_score():
    # Verified in the app: PIR -2 in a win -> -1.8 (not -2.2).
    assert player_fantasy_points(-2, True) == pytest.approx(-1.8)
    assert player_fantasy_points(-2, False) == pytest.approx(-2)


def test_vectorized_fantasy_points():
    out = player_fantasy_points([10, -4, 0], [True, True, False])
    assert list(out) == pytest.approx([11.0, -3.6, 0.0])


@pytest.mark.parametrize(
    "margin,ot,expected",
    [
        (1, False, 10), (10, False, 10), (11, False, 20), (20, False, 20), (21, False, 25), (40, False, 25),
        (-1, False, -5), (-10, False, -5), (-11, False, -10), (-20, False, -10), (-21, False, -20),
        (3, True, 10), (-2, True, -5),
    ],
)
def test_coach_points(margin, ot, expected):
    assert coach_points(margin, ot) == expected


def test_coach_points_rejects_tie():
    with pytest.raises(ValueError):
        coach_points(0)


def test_rules_formations_and_bounds():
    r = default_rules()
    assert r.players_per_position == {"G": 4, "F": 4, "C": 2}
    assert set(r.formations) == {(2, 2, 1), (1, 2, 2), (2, 1, 2), (1, 3, 1), (3, 1, 1)}
    b = r.formation_bounds()
    assert b == {"G": (1, 3), "F": (1, 3), "C": (1, 2)}
    # The bounds plus "5 starters" generate exactly the allowed formations.
    gen = {(g, f, 5 - g - f) for g in range(1, 4) for f in range(1, 4) if 1 <= 5 - g - f <= 2}
    assert gen == set(r.formations)
    assert r.captain_multiplier == 2.0
    assert r.multipliers["bench"] == 0.5 and r.multipliers["sixth_man"] == 1.0
    assert r.coach_counts_toward_team_limit is True  # unknown -> stricter rule
