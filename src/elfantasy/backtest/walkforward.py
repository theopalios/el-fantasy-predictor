"""Walk-forward backtest, round by round.

    python -m elfantasy.backtest.walkforward

For each regular-season round r >= 2 of the evaluation seasons, every prediction uses only
games before round r. Two candidate pools per round:

  prev    players who played their team's previous game. This is what we would know
          before the deadline without injury news; players who then miss the game score 0.
  played  players who actually played round r ("known availability", an upper bound).

Prediction metrics are computed on players who played (the models predict points GIVEN that
the player plays). The squad simulation (2026-27, the only season with real prices) picks
the best 11 within 100 credits each round and scores it with actual points, against the
perfect-hindsight squad.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from elfantasy.features.build import FeatureContext
from elfantasy.features.common import Store
from elfantasy.features.team import team_features
from elfantasy.models import baselines as B
from elfantasy.optimize.squad import best_squad, squad_points
from elfantasy.paths import PROCESSED_DIR, REPO_ROOT

log = logging.getLogger("elfantasy.backtest")

EVAL_SEASONS = (2023, 2024, 2025, 2026)
PRICE_SEASON = 2026
POS_MAP = {"Guard": "G", "Forward": "F", "Center": "C"}


# ------------------------------------------------------------------------ candidates

def build_candidates(ctx: FeatureContext, seasons=EVAL_SEASONS) -> pd.DataFrame:
    pg, tgi = ctx.store.player_games, ctx.tgi
    key = ["season", "gamecode", "team"]
    ctx_cols = ["season", "team", "idx", "gamecode", "date_utc", "round", "phase", "opponent", "is_home"]
    played = pg[pg["played"]][["player_code", "position_name", *key]].merge(tgi[[*key, "idx"]], on=key)

    prev = played[["player_code", "position_name", "season", "team", "idx"]].copy()
    prev["idx"] = prev["idx"] + 1
    prev = prev.merge(tgi[ctx_cols], on=["season", "team", "idx"])
    prev["pool_prev"] = True

    act = played.merge(tgi[ctx_cols], on=["season", "team", "idx", "gamecode"])
    act["pool_played"] = True

    ids = ["player_code", "season", "team", "gamecode"]
    shared = ["position_name", "date_utc", "round", "phase", "opponent", "is_home", "idx"]
    cand = prev.merge(act[ids + shared + ["pool_played"]], on=ids, how="outer", suffixes=("", "_a"))
    for c in shared:
        cand[c] = cand[c].fillna(cand[f"{c}_a"])
    cand = cand.drop(columns=[f"{c}_a" for c in shared])
    cand["pool_prev"] = cand["pool_prev"].fillna(False).astype(bool)
    cand["pool_played"] = cand["pool_played"].fillna(False).astype(bool)
    cand = cand[cand["season"].isin(seasons) & (cand["phase"] == "RS") & (cand["round"] >= 2)]

    actual = pg[["season", "gamecode", "player_code", "fantasy_pts", "played", "minutes"]]
    cand = cand.merge(actual, on=["season", "gamecode", "player_code"], how="left")
    cand["y_fantasy_pts"] = cand["fantasy_pts"].fillna(0.0)
    cand["y_played"] = cand["played"].fillna(False).astype(bool)
    cand["round"] = cand["round"].astype(int)
    return cand.drop(columns=["fantasy_pts", "played"]).reset_index(drop=True)


# ----------------------------------------------------------------- fantasy prices

def fantasy_price_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player prices, coach prices) per round for the current season, keyed by EL codes."""
    ml = pd.read_parquet(PROCESSED_DIR / "fantasy_match_log.parquet")
    fp = pd.read_parquet(PROCESSED_DIR / "fantasy_players.parquet")
    mp = pd.read_parquet(PROCESSED_DIR / "map_player.parquet")
    mt = pd.read_parquet(PROCESSED_DIR / "map_team.parquet")

    players = (ml[["player_id", "matchday", "price"]]
               .merge(mp[["player_id", "person_code"]].dropna(), on="player_id")
               .merge(fp[["player_id", "position"]], on="player_id"))
    players = players[players["position"] != "Head Coach"].rename(
        columns={"person_code": "player_code", "matchday": "round"})
    players["pos"] = players["position"].map(POS_MAP)
    players = players.drop_duplicates(["player_code", "round"])

    coaches = (ml[["player_id", "matchday", "price"]]
               .merge(fp[fp["position"] == "Head Coach"][["player_id", "team_abbr"]], on="player_id")
               .merge(mt[["app_abbr", "el_code"]], left_on="team_abbr", right_on="app_abbr"))
    coaches = coaches.rename(columns={"el_code": "team", "matchday": "round"})[["player_id", "team", "round", "price"]]
    coaches["price"] = coaches.groupby("team")["price"].transform(lambda s: s.bfill().ffill())
    return players[["player_code", "player_id", "round", "price", "pos"]], coaches


# ------------------------------------------------------------------------ metrics

def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return np.nan
    return float(a.rank().corr(b.rank()))


def prediction_metrics(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """MAE / RMSE / bias / within-round Spearman on players who played, per season."""
    rows = []
    d = df[df["y_played"] & df["pool_played"]]
    for season, ds in d.groupby("season"):
        for col in pred_cols:
            x = ds.dropna(subset=[col])
            if x.empty:
                continue
            err = x[col] - x["y_fantasy_pts"]
            sp = x.groupby("round").apply(lambda g: _spearman(g[col], g["y_fantasy_pts"]), include_groups=False)
            rows.append({"season": season, "method": col.replace("pred_", ""), "n": len(x),
                         "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
                         "bias": err.mean(), "spearman_per_round": sp.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- squad simulation

def squad_simulation(df: pd.DataFrame, coach_rows: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Best 11 per round (budget 100) for each prediction column and pool, vs hindsight."""
    out = []
    for r, R in df.groupby("round"):
        R = R.dropna(subset=["price", "pos"]).drop_duplicates("player_code").set_index("player_code")
        C = coach_rows[coach_rows["round"] == r].drop_duplicates("team").set_index("team", drop=False)
        if R.empty or C.empty:
            continue
        actual = R["y_fantasy_pts"]
        played = R[R["pool_played"]]
        oracle = best_squad(played.assign(exp=played["y_fantasy_pts"]), C.assign(exp=C["coach_actual"]))
        best = squad_points(oracle, actual, oracle.coach["coach_actual"])
        out.append({"round": r, "pool": "hindsight", "method": "oracle", "points": best})
        for pool in ("prev", "played"):
            P = R[R[f"pool_{pool}"]]
            for col in pred_cols:
                res = best_squad(P.assign(exp=P[col]), C.assign(exp=C["coach_pred"]))
                if res.status != "Optimal":
                    continue
                pts = squad_points(res, actual, res.coach["coach_actual"])
                out.append({"round": r, "pool": pool, "method": col.replace("pred_", ""), "points": pts,
                            "expected": res.objective, "cost": res.cost})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- runner

def run(seasons=EVAL_SEASONS) -> dict[str, pd.DataFrame]:
    store = Store.load()
    ctx = FeatureContext.from_store(store)
    train = ctx.training_rows()
    cand = build_candidates(ctx, seasons)
    X = ctx.features(cand)
    df = pd.concat([X, cand[["pool_prev", "pool_played", "y_fantasy_pts", "y_played"]]], axis=1)
    log.info("candidates: %d rows (%d played)", len(df), int(df["y_played"].sum()))

    # baselines (fallback = debut mean from seasons before the evaluated one)
    preds = []
    for season, ds in df.groupby("season"):
        fb = B.debut_fallback(train[train["season"] < season])
        ds = ds.copy()
        ds["pred_season_avg"] = B.season_average(ds, fb)
        ds["pred_ewma"] = B.ewma(ds, fb)
        preds.append(ds)
    df = pd.concat(preds).sort_index()

    # prices (current season) and the price-implied baseline, refit each round on earlier rounds
    prices, coach_prices = fantasy_price_table()
    df = df.merge(prices, on=["player_code", "round"], how="left").set_axis(df.index)
    df.loc[df["season"] != PRICE_SEASON, ["price", "pos", "player_id"]] = np.nan
    tr = train[train["season"] == PRICE_SEASON].merge(prices[["player_code", "round", "price"]],
                                                      on=["player_code", "round"], how="left")
    df["pred_price"] = np.nan
    for r in sorted(df.loc[df["season"] == PRICE_SEASON, "round"].unique()):
        a, b = B.fit_price_line(tr[tr["round"] < r])
        m = (df["season"] == PRICE_SEASON) & (df["round"] == r)
        df.loc[m, "pred_price"] = B.price_implied(df[m], a, b)

    # coaches: Elo-based expected points (sigma fitted on seasons before 2026), season average, price line
    hist = ctx.elo.merge(store.team_games[["season", "gamecode", "team", "margin"]], on=["season", "gamecode", "team"])
    sigma = B.fit_margin_sigma(hist[hist["season"] < PRICE_SEASON], hist.loc[hist["season"] < PRICE_SEASON, "margin"])
    tg = store.team_games
    coach = tg[(tg["season"] == PRICE_SEASON) & (tg["phase"] == "RS")][
        ["season", "gamecode", "team", "round", "coach_pts", "date_utc"]].copy()
    coach = coach.merge(ctx.elo[["season", "gamecode", "team", "elo_exp_margin"]], on=["season", "gamecode", "team"])
    coach["coach_pred"] = B.coach_points_from_margin(coach["elo_exp_margin"].to_numpy(), sigma)
    coach = coach.rename(columns={"coach_pts": "coach_actual"}).merge(coach_prices, on=["team", "round"], how="left")
    cs = team_features(coach, ctx.tstate, "team")
    coach["coach_season_avg"] = cs["team_coach_pts_season_mean"].fillna(0.0).to_numpy()

    pred_cols = ["pred_season_avg", "pred_ewma", "pred_price"]
    metrics = prediction_metrics(df, pred_cols)
    coach_metrics = pd.DataFrame([
        {"method": "elo", "MAE": (coach["coach_pred"] - coach["coach_actual"]).abs().mean()},
        {"method": "season_avg", "MAE": (coach["coach_season_avg"] - coach["coach_actual"]).abs().mean()},
        {"method": "zero", "MAE": coach["coach_actual"].abs().mean()},
    ])
    sim = squad_simulation(df[df["season"] == PRICE_SEASON], coach, pred_cols)
    df.to_parquet(PROCESSED_DIR / "backtest_baselines.parquet", index=False)
    return {"metrics": metrics, "coach_metrics": coach_metrics, "sim": sim, "sigma": pd.Series({"sigma": sigma})}


def write_report(res: dict[str, pd.DataFrame], path: Path) -> str:
    m = res["metrics"]
    overall = (m.assign(w=m["n"]).groupby("method")
               .apply(lambda g: pd.Series({
                   "MAE": np.average(g["MAE"], weights=g["w"]),
                   "RMSE": np.sqrt(np.average(g["RMSE"] ** 2, weights=g["w"])),
                   "spearman": g["spearman_per_round"].mean()}), include_groups=False))
    sim = res["sim"]
    tot = sim.groupby(["pool", "method"])["points"].agg(["sum", "mean", "count"]).reset_index()
    oracle = tot.loc[tot["method"] == "oracle", "sum"].iloc[0] if (tot["method"] == "oracle").any() else np.nan
    tot["pct_of_hindsight"] = 100 * tot["sum"] / oracle
    per_round = sim.pivot_table(index="round", columns=["pool", "method"], values="points")

    def md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
        return df.to_markdown(floatfmt=floatfmt)

    text = "\n".join([
        "# Baseline backtest",
        "",
        "Generated by `python -m elfantasy.backtest.walkforward`. Regular season, rounds 2+, every",
        "prediction uses only games before the round.",
        "",
        "## Prediction accuracy (players who played)",
        "",
        "Per season:",
        "",
        md(m.set_index(["season", "method"])),
        "",
        "All evaluated seasons (2023-24 to 2026-27 R13):",
        "",
        md(overall),
        "",
        "`spearman` is the average within-round rank correlation between predicted and actual",
        "points, which is what matters for picking players.",
        "",
        "## Coach points (2026-27)",
        "",
        md(res["coach_metrics"].set_index("method")),
        "",
        f"Elo margin model: sigma = {res['sigma']['sigma']:.1f} points.",
        "",
        "## Best-11 squad simulation (2026-27, real prices, 100 credits)",
        "",
        "Each round the optimizer picks the best 11 from each method's predictions; the squad is",
        "scored with actual points. `prev` = players who played their team's previous game",
        "(realistic, no injury news); `played` = players who actually played (known availability).",
        "`oracle` = best possible squad in hindsight.",
        "",
        md(tot.set_index(["pool", "method"]), ".1f"),
        "",
        "Per round:",
        "",
        md(per_round, ".1f"),
        "",
    ])
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(REPO_ROOT / "docs" / "backtest_baselines.md"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run()
    print(write_report(res, Path(args.report)))


if __name__ == "__main__":
    main()
