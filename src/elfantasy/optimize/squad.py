"""Best squad as an integer linear program. Every constraint comes from the rules file.

    best_squad(players, coaches, value_col="exp")

players: DataFrame (unique index) with columns pos (G/F/C), team, price, <value_col>
coaches: DataFrame (unique index) with columns team, price, <value_col>
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from elfantasy.rules import Rules, default_rules


@dataclass
class SquadResult:
    status: str
    players: pd.DataFrame      # the 10 players with a 'role' column: starter / sixth / bench, plus 'captain'
    coach: pd.Series
    objective: float
    cost: float


def best_squad(
    players: pd.DataFrame,
    coaches: pd.DataFrame,
    value_col: str = "exp",
    rules: Rules | None = None,
    budget: float | None = None,
    locked: set | None = None,
    excluded: set | None = None,
) -> SquadResult:
    rules = rules or default_rules()
    budget = rules.initial_budget if budget is None else budget
    mult = rules.multipliers
    captain_extra = (rules.captain_multiplier - 1.0) * mult["starter"]
    locked, excluded = locked or set(), excluded or set()

    P = players.dropna(subset=["price", value_col, "pos", "team"])
    P = P[~P.index.isin(excluded)]
    C = coaches.dropna(subset=["price", value_col, "team"])

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    s = {i: pulp.LpVariable(f"s_{n}", cat="Binary") for n, i in enumerate(P.index)}
    m = {i: pulp.LpVariable(f"m_{n}", cat="Binary") for n, i in enumerate(P.index)}
    b = {i: pulp.LpVariable(f"b_{n}", cat="Binary") for n, i in enumerate(P.index)}
    c = {i: pulp.LpVariable(f"c_{n}", cat="Binary") for n, i in enumerate(P.index)}
    h = {k: pulp.LpVariable(f"h_{n}", cat="Binary") for n, k in enumerate(C.index)}
    x = {i: s[i] + m[i] + b[i] for i in P.index}

    v = P[value_col].to_dict()
    vc = C[value_col].to_dict()
    prob += (
        pulp.lpSum(v[i] * (mult["starter"] * s[i] + mult["sixth_man"] * m[i] + mult["bench"] * b[i]
                           + captain_extra * c[i]) for i in P.index)
        + pulp.lpSum(vc[k] * mult["head_coach"] * h[k] for k in C.index)
    )

    lineup = rules.raw["lineup"]
    for i in P.index:
        prob += x[i] <= 1
        prob += c[i] <= s[i]
    for pos, n in rules.players_per_position.items():
        idx = P.index[P["pos"] == pos]
        prob += pulp.lpSum(x[i] for i in idx) == n
        lo, hi = rules.formation_bounds()[pos]
        prob += pulp.lpSum(s[i] for i in idx) >= lo
        prob += pulp.lpSum(s[i] for i in idx) <= hi
    prob += pulp.lpSum(s.values()) == lineup["starters"]
    prob += pulp.lpSum(m.values()) == lineup["sixth_man"]
    prob += pulp.lpSum(b.values()) == lineup["bench"]
    prob += pulp.lpSum(c.values()) == 1
    prob += pulp.lpSum(h.values()) == 1

    price, cprice = P["price"].to_dict(), C["price"].to_dict()
    prob += pulp.lpSum(price[i] * x[i] for i in P.index) + pulp.lpSum(cprice[k] * h[k] for k in C.index) <= budget

    for team in set(P["team"]) | set(C["team"]):
        n_players = pulp.lpSum(x[i] for i in P.index[P["team"] == team])
        n_coach = pulp.lpSum(h[k] for k in C.index[C["team"] == team]) if rules.coach_counts_toward_team_limit else 0
        prob += n_players + n_coach <= rules.max_players_per_real_team

    for i in locked & set(P.index):
        prob += x[i] == 1

    available = pulp.listSolvers(onlyAvailable=True)
    solver = pulp.HiGHS(msg=False) if "HiGHS" in available else pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return SquadResult(status, P.iloc[0:0], pd.Series(dtype=object), float("nan"), float("nan"))

    chosen = [i for i in P.index if x[i].value() > 0.5]
    out = P.loc[chosen].copy()
    out["role"] = ["starter" if s[i].value() > 0.5 else "sixth" if m[i].value() > 0.5 else "bench" for i in chosen]
    out["captain"] = [c[i].value() > 0.5 for i in chosen]
    k = next(k for k in C.index if h[k].value() > 0.5)
    coach = C.loc[k]
    cost = float(out["price"].sum() + coach["price"])
    return SquadResult(status, out.sort_values(["role", "pos"]), coach, float(pulp.value(prob.objective)), cost)


def squad_points(result: SquadResult, player_pts: pd.Series, coach_pts: float, rules: Rules | None = None) -> float:
    """Fantasy points the squad actually scored, given actual points per player index."""
    rules = rules or default_rules()
    mult = rules.multipliers
    total = 0.0
    for i, row in result.players.iterrows():
        pts = float(player_pts.get(i, 0.0) or 0.0)
        w = {"starter": mult["starter"], "sixth": mult["sixth_man"], "bench": mult["bench"]}[row["role"]]
        if row["captain"]:
            w *= rules.captain_multiplier
        total += w * pts
    return total + mult["head_coach"] * float(coach_pts)
