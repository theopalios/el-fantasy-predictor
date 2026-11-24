import numpy as np
import pandas as pd

from elfantasy.optimize.squad import best_squad, squad_points
from elfantasy.rules import default_rules


def _pool(seed=0, n_per_pos=12, n_teams=8):
    rng = np.random.default_rng(seed)
    rows = []
    for pos, n in (("G", n_per_pos), ("F", n_per_pos), ("C", n_per_pos)):
        for k in range(n):
            price = float(rng.uniform(4, 20))
            rows.append({"pos": pos, "team": f"T{rng.integers(n_teams)}", "price": price,
                         "exp": price * 1.1 + rng.normal(0, 2)})
    players = pd.DataFrame(rows)
    players.index = [f"p{i}" for i in range(len(players))]
    coaches = pd.DataFrame({"team": [f"T{i}" for i in range(n_teams)],
                            "price": rng.uniform(4, 10, n_teams), "exp": rng.uniform(-5, 15, n_teams)})
    coaches.index = [f"c{i}" for i in range(n_teams)]
    return players, coaches


def test_squad_respects_every_rule():
    rules = default_rules()
    players, coaches = _pool()
    res = best_squad(players, coaches)
    assert res.status == "Optimal"
    sq = res.players
    assert sq["pos"].value_counts().to_dict() == {"G": 4, "F": 4, "C": 2}
    assert (sq["role"] == "starter").sum() == 5 and (sq["role"] == "sixth").sum() == 1 and (sq["role"] == "bench").sum() == 4
    starters = sq[sq["role"] == "starter"]["pos"].value_counts()
    formation = (starters.get("G", 0), starters.get("F", 0), starters.get("C", 0))
    assert formation in set(rules.formations)
    assert sq["captain"].sum() == 1 and sq.loc[sq["captain"], "role"].iloc[0] == "starter"
    assert res.cost <= rules.initial_budget + 1e-6
    counts = sq["team"].value_counts()
    counts[res.coach["team"]] = counts.get(res.coach["team"], 0) + 1   # coach counted (strict rule)
    assert counts.max() <= rules.max_players_per_real_team


def test_captain_is_best_starter_and_points_add_up():
    players, coaches = _pool(seed=3)
    res = best_squad(players, coaches)
    sq = res.players
    starters = sq[sq["role"] == "starter"]
    assert sq.loc[sq["captain"], "exp"].iloc[0] == starters["exp"].max()
    # scoring with expected values reproduces the objective
    pts = squad_points(res, players["exp"], res.coach["exp"])
    assert abs(pts - res.objective) < 1e-6


def test_budget_binds():
    players, coaches = _pool(seed=5)
    tight = best_squad(players, coaches, budget=95)
    loose = best_squad(players, coaches, budget=140)
    assert tight.status == "Optimal" and loose.status == "Optimal"
    assert tight.cost <= 95 + 1e-6
    assert loose.objective >= tight.objective


def test_infeasible_budget_is_reported():
    players, coaches = _pool(seed=5)
    res = best_squad(players, coaches, budget=10)
    assert res.status != "Optimal" and res.players.empty
