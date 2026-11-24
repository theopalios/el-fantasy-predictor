from elfantasy.data.euroleague import normalize_player_code, parse_game_stats, parse_minutes
from elfantasy.data.fantasy_app import parse_lineups, parse_match_log
from elfantasy.data.fantasy_build import norm_name, split_el_name


def test_parse_minutes():
    assert parse_minutes("33:21") == 33 + 21 / 60
    assert parse_minutes("DNP") == 0.0
    assert parse_minutes(None) == 0.0


def test_normalize_player_code():
    assert normalize_player_code("P007200   ") == "007200"


def test_names():
    assert split_el_name("KABENGELE, MFIONDU") == ("KABENGELE", "MFIONDU")
    assert norm_name("Jimmy Clark III") == "JIMMY CLARK"
    assert norm_name("Džanan Musa") == "DZANAN MUSA"


def _player(code, name, seconds, starter, **stats):
    st = {"timePlayed": seconds, "startFive": starter, "dorsal": 1, "valuation": stats.pop("valuation", 0)}
    st.update(stats)
    return {"player": {"person": {"code": code, "name": name}, "positionName": "Guard"}, "stats": st}


def test_parse_game_stats():
    raw = {
        "local": {
            "coach": {"code": "C1", "name": "COACH, HOME"},
            "players": [_player("001", "A, B", 600.0, True, points=10, valuation=9),
                        _player("002", "C, D", 0.0, False)],
            "team": {"totalRebounds": 2},
            "total": {"points": 80},
        },
        "road": {"coach": {"code": "C2"}, "players": [], "team": {}, "total": {"points": 70}},
    }
    players, teams = parse_game_stats(raw, 2026, 1, "HOM", "AWY")
    assert list(players["team"]) == ["HOM", "HOM"]
    assert players.loc[0, "minutes"] == 10.0 and players.loc[0, "is_starter"]
    assert players.loc[1, "dnp"] and players.loc[1, "minutes"] == 0.0
    assert players.loc[0, "pts"] == 10 and players.loc[0, "pir"] == 9
    assert list(teams["team"]) == ["HOM", "AWY"] and teams.loc[0, "coach_code"] == "C1"


def test_parse_match_log():
    raw = {
        "columns": ["MD", "VS", "CR", "PLUS", "FPT", "MIN", "ST", "PTS", "+/-"],
        "rows": [["13", "@MTA", "16.2", "0.0", "16.50", "26:26", "Y", "9", "4"],
                 ["12", "VS EFS", "15.8", "+0.4", "27.50", "30:53", "N", "18", "12"]],
    }
    df = parse_match_log(raw, 4233)
    assert list(df["matchday"]) == [13, 12]
    assert list(df["is_home"]) == [False, True]
    assert list(df["opp_abbr"]) == ["MTA", "EFS"]
    assert df.loc[1, "price_change"] == 0.4 and df.loc[0, "fantasy_pts"] == 16.5
    assert df.loc[0, "started"] and not df.loc[1, "started"]


def test_parse_lineups():
    raw = {"id": 1, "status": "scheduled",
           "home_team": {"id": 10, "abbreviation": "BJK", "lineups": [
               {"id": 3776, "first_name": "Wenyen", "last_name": "Gabriel", "position": "Center",
                "pts": 0, "quotation": 8.1, "status": "starter", "probability_of_playing": 0.25,
                "is_on_fire": False}]},
           "away_team": {"id": 11, "abbreviation": "PBB", "lineups": []}}
    df = parse_lineups(raw)
    assert df.loc[0, "p_play"] == 0.25 and df.loc[0, "price"] == 8.1 and df.loc[0, "is_home"]
