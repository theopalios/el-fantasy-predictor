import pytest

from elfantasy.evaluate import score_record


def _record():
    squad = [
        {"player_id": 1, "name": "Cap", "role": "starter", "captain": True, "exp_pts": 20.0},
        {"player_id": 2, "name": "S2", "role": "starter", "captain": False, "exp_pts": 15.0},
        {"player_id": 3, "name": "S3", "role": "starter", "captain": False, "exp_pts": 12.0},
        {"player_id": 4, "name": "S4", "role": "starter", "captain": False, "exp_pts": 11.0},
        {"player_id": 5, "name": "S5", "role": "starter", "captain": False, "exp_pts": 10.0},
        {"player_id": 6, "name": "Sixth", "role": "sixth", "captain": False, "exp_pts": 9.0},
        {"player_id": 7, "name": "B1", "role": "bench", "captain": False, "exp_pts": 6.0},
        {"player_id": 8, "name": "B2", "role": "bench", "captain": False, "exp_pts": 5.0},
        {"player_id": 9, "name": "B3", "role": "bench", "captain": False, "exp_pts": 4.0},
        {"player_id": 10, "name": "B4", "role": "bench", "captain": False, "exp_pts": 4.0},
    ]
    return {"squad": squad, "coach": {"player_id": 99, "team": "OLY", "exp_pts": 10.0}}


def test_score_record_applies_multipliers():
    pts = {1: 30.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0}  # 10 did not play
    res = score_record(_record(), pts, coach_pts=20.0)
    # captain 2x30 + 4 starters 40 + sixth 8 + bench (6+4+2+0)/2 + coach 20
    assert res["actual_total"] == pytest.approx(60 + 40 + 8 + 6 + 20)
    assert not res["players"].set_index("name").loc["B4", "played"]


def test_score_record_without_coach_result():
    res = score_record(_record(), {}, coach_pts=None)
    assert res["actual_total"] == 0.0
