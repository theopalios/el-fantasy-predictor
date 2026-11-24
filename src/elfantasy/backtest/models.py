"""Walk-forward backtest of the trained models against the baselines (milestone M3).

    python -m elfantasy.backtest.models

* Player points model: retrained every 5 rounds on everything before the block.
* Availability model and game model: retrained before each season.
* 2023-24 was used to tune the player model; 2024-25 onward is the held-out test.
* Squad simulation over four seasons: real prices for 2026-27, synthetic prices (fitted
  on 2026-27's real prices) for 2023-24 to 2025-26.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from elfantasy.backtest.synthetic_prices import PriceModel
from elfantasy.backtest.walkforward import (
    POS_MAP,
    build_candidates,
    fantasy_price_table,
    prediction_metrics,
    squad_simulation,
)
from elfantasy.features.build import FeatureContext
from elfantasy.features.catalog import FEATURE_DESCRIPTIONS
from elfantasy.features.common import Store
from elfantasy.models import baselines as B
from elfantasy.models.availability import AvailabilityModel
from elfantasy.models.game import GameModel, game_frame
from elfantasy.models.player_points import PlayerPointsModel
from elfantasy.paths import PROCESSED_DIR, REPO_ROOT

log = logging.getLogger("elfantasy.backtest.models")

EVAL = (2023, 2024, 2025, 2026)
TUNING_SEASON = 2023
PRICE_SEASON = 2026
BLOCK = 5


def _ols_blend(prior: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    if len(prior) < 150:
        return None
    A = np.column_stack([np.ones(len(prior))] + [prior[c].to_numpy(float) for c in cols])
    coef, *_ = np.linalg.lstsq(A, prior["y_fantasy_pts"].to_numpy(float), rcond=None)
    return coef


def run() -> dict:
    store = Store.load()
    ctx = FeatureContext.from_store(store)
    train = ctx.training_rows()
    cand = build_candidates(ctx, seasons=tuple(range(2016, 2027)))
    X = ctx.features(cand)
    df_all = pd.concat([X, cand[["pool_prev", "pool_played", "y_fantasy_pts", "y_played"]]], axis=1)
    df = df_all[df_all["season"].isin(EVAL)].copy()
    out: dict = {}

    # ---- baselines
    for s in EVAL:
        m = df["season"] == s
        fb = B.debut_fallback(train[train["season"] < s])
        df.loc[m, "pred_season_avg"] = B.season_average(df[m], fb)
        df.loc[m, "pred_ewma"] = B.ewma(df[m], fb)

    # ---- player points model, retrained every BLOCK rounds
    df["pred_model"] = np.nan
    model = None
    for s in EVAL:
        rounds = sorted(df.loc[df["season"] == s, "round"].unique())
        for b in range(min(rounds), max(rounds) + 1, BLOCK):
            tr = train[(train["season"] < s) | ((train["season"] == s) & (train["phase"] == "RS") & (train["round"] < b))]
            model = PlayerPointsModel().fit(tr)
            m = (df["season"] == s) & (df["round"] >= b) & (df["round"] < b + BLOCK)
            df.loc[m, "pred_model"] = model.predict(df[m])
        log.info("player model: season %s done", s)
    out["importance"] = model.importance()

    # ---- availability model (per season)
    rows = []
    for s in EVAL:
        tr = df_all[(df_all["season"] < s) & df_all["pool_prev"]]
        am = AvailabilityModel().fit(tr)
        m = df["season"] == s
        df.loc[m, "p_play"] = am.predict(df[m])
        ev = df[m & df["pool_prev"]]
        base = tr["y_played"].mean()
        y = ev["y_played"].astype(float)
        p = ev["p_play"].clip(1e-4, 1 - 1e-4)
        rows.append({"season": s, "n": len(ev), "share_played": y.mean(),
                     "brier_model": ((p - y) ** 2).mean(), "brier_constant": ((base - y) ** 2).mean(),
                     "logloss_model": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
                     "logloss_constant": -(y * np.log(base) + (1 - y) * np.log(1 - base)).mean()})
    out["availability"] = pd.DataFrame(rows)
    df["pred_model_avail"] = df["pred_model"] * df["p_play"]

    # ---- game model (per season)
    gf = game_frame(ctx)
    rows = []
    for s in EVAL:
        hist = gf[gf["season"] < s]
        gm = GameModel().fit(hist)
        m = gf["season"] == s
        gf.loc[m, "coach_model"] = gm.coach_points(gf[m])
        gf.loc[m, "p_win_model"] = gm.p_win(gf[m])
        sig = B.fit_margin_sigma(hist, hist["margin"])
        gf.loc[m, "coach_elo"] = B.coach_points_from_margin(gf.loc[m, "elo_exp_margin"].to_numpy(), sig)
        e = gf[m]
        win = (e["margin"] > 0).astype(float)
        rows.append({"season": s, "games": len(e) // 2,
                     "coach_MAE_model": (e["coach_model"] - e["coach_pts"]).abs().mean(),
                     "coach_MAE_elo": (e["coach_elo"] - e["coach_pts"]).abs().mean(),
                     "brier_win_model": ((e["p_win_model"] - win) ** 2).mean(),
                     "brier_win_elo": ((e["elo_p_win"] - win) ** 2).mean(),
                     "sigma": gm.sigma})
    out["game"] = pd.DataFrame(rows)
    out["game_coef"] = gm.coefficients()

    # ---- prices: real for 2026-27, synthetic for earlier seasons
    prices, coach_prices = fantasy_price_table()
    df = df.merge(prices[["player_code", "round", "price", "pos"]], on=["player_code", "round"], how="left").set_axis(df.index)
    df.loc[df["season"] != PRICE_SEASON, ["price", "pos"]] = np.nan
    coach = gf[gf["season"].isin(EVAL) & (gf["phase"] == "RS")].merge(
        coach_prices[["team", "round", "price"]].assign(season=PRICE_SEASON), on=["season", "team", "round"], how="left")
    cur = df[df["season"] == PRICE_SEASON]
    # accuracy check: fit on rounds <= 9, test on later rounds
    pm_check = PriceModel().fit(cur[cur["round"] <= 9], coach[coach["season"] == PRICE_SEASON])
    test = cur[(cur["round"] > 9)].dropna(subset=["price"])
    pred = pm_check.predict(test)
    out["price_check"] = pd.Series({
        "MAE_credits": float(np.abs(pred - test["price"]).mean()),
        "R2": float(1 - ((pred - test["price"]) ** 2).sum() / ((test["price"] - test["price"].mean()) ** 2).sum()),
        "n": len(test)})
    pm = PriceModel().fit(cur, coach[coach["season"] == PRICE_SEASON])
    hist = df["season"] != PRICE_SEASON
    df.loc[hist, "price"] = pm.predict(df[hist])
    df.loc[hist, "pos"] = df.loc[hist, "position_name"].map(POS_MAP)
    ch = coach["season"] != PRICE_SEASON
    coach.loc[ch, "price"] = pm.predict_coach(coach.loc[ch, "elo"])
    coach = coach.rename(columns={"coach_pts": "coach_actual", "coach_model": "coach_pred"})

    # ---- 2026-27: blend the model with the app's price, refit each round on earlier rounds
    df["pred_model_price"] = np.nan
    for r in sorted(df.loc[df["season"] == PRICE_SEASON, "round"].unique()):
        prior = df[(df["season"] == PRICE_SEASON) & (df["round"] < r) & df["y_played"] & df["pool_played"]]
        prior = prior.dropna(subset=["pred_model", "price"])
        coef = _ols_blend(prior, ["pred_model", "price"])
        m = (df["season"] == PRICE_SEASON) & (df["round"] == r)
        if coef is None:
            df.loc[m, "pred_model_price"] = df.loc[m, "pred_model"]
        else:
            df.loc[m, "pred_model_price"] = coef[0] + coef[1] * df.loc[m, "pred_model"] + coef[2] * df.loc[m, "price"]
    df["pred_model_price_avail"] = df["pred_model_price"] * df["p_play"]

    # ---- metrics
    pred_cols = ["pred_season_avg", "pred_ewma", "pred_model", "pred_model_price"]
    out["metrics"] = prediction_metrics(df, pred_cols)

    sims = []
    for s in EVAL:
        prev_cols = ["pred_season_avg", "pred_ewma", "pred_model", "pred_model_avail"]
        played_cols = ["pred_season_avg", "pred_ewma", "pred_model"]
        if s == PRICE_SEASON:
            prev_cols += ["pred_model_price", "pred_model_price_avail"]
            played_cols += ["pred_model_price"]
        sim = squad_simulation(df[df["season"] == s], coach[coach["season"] == s], pred_cols,
                               pools={"prev": prev_cols, "played": played_cols})
        sims.append(sim.assign(season=s))
        log.info("squad simulation: season %s done (%d rounds)", s, sim["round"].nunique())
    out["sim"] = pd.concat(sims, ignore_index=True)
    out["df"] = df
    df.to_parquet(PROCESSED_DIR / "backtest_models.parquet", index=False)
    return out


# --------------------------------------------------------------------------- report

def _md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=fmt)


def write_report(res: dict, path: Path) -> str:
    m = res["metrics"].copy()
    m["split"] = np.where(m["season"] == TUNING_SEASON, "tuning", "held-out")
    held = m[m["split"] == "held-out"]
    seasons_held = held["season"].nunique()
    full = held.groupby("method")["season"].transform("nunique") == seasons_held   # same rows only

    def pool(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": g["n"].sum(),
            "MAE": np.average(g["MAE"], weights=g["n"]),
            "RMSE": np.sqrt(np.average(g["RMSE"] ** 2, weights=g["n"])),
            "spearman": np.average(g["spearman_per_round"], weights=g["n"])})

    overall = held[full].groupby("method").apply(pool, include_groups=False)
    cur = m[m["season"] == PRICE_SEASON].set_index("method")[["n", "MAE", "RMSE", "spearman_per_round"]]

    sim = res["sim"]
    tot = sim.groupby(["season", "pool", "method"])["points"].agg(["mean", "count"]).reset_index()
    oracle = tot[tot["method"] == "oracle"].set_index("season")["mean"]
    tot["pct_of_hindsight"] = 100 * tot["mean"] / tot["season"].map(oracle)
    wide = tot.pivot_table(index=["pool", "method"], columns="season", values="mean")
    pct = tot.pivot_table(index=["pool", "method"], columns="season", values="pct_of_hindsight")
    allseasons = (sim.groupby(["pool", "method"])["points"].mean().rename("avg_pts_per_round").to_frame())
    allseasons["pct_of_hindsight"] = 100 * allseasons["avg_pts_per_round"] / sim[sim["method"] == "oracle"]["points"].mean()

    imp = res["importance"].head(25).rename("gain_share").to_frame()
    imp["description"] = [FEATURE_DESCRIPTIONS.get(f, "") for f in imp.index]

    text = "\n".join([
        "# Model backtest (M3)",
        "",
        "Generated by `python -m elfantasy.backtest.models`. Regular season, rounds 2+. The player",
        "model is retrained every 5 rounds on data before the block; the game and availability",
        "models are retrained before each season. 2023-24 was used for tuning; the other seasons",
        "are held out.",
        "",
        "## Player points (players who played)",
        "",
        "Held-out seasons (2024-25, 2025-26, 2026-27 R2-R13), methods available in all of them:",
        "",
        _md(overall),
        "",
        "2026-27 only (same rows for every method, including the price blend):",
        "",
        _md(cur),
        "",
        "Per season:",
        "",
        _md(m.set_index(["season", "method"]).drop(columns="split")),
        "",
        "`model_price` (2026-27 only) blends the model with the app's price, refitted each round.",
        "",
        "## Availability: P(plays | played the team's previous game)",
        "",
        "Features use only information known before the deadline. `vacated_min` counts missing",
        "teammates and never the player himself. An earlier version counted him, which leaked",
        "the label; that was found in review on 24 Nov and fixed.",
        "",
        _md(res["availability"].set_index("season")),
        "",
        "## Game model: coach points and win probability",
        "",
        _md(res["game"].set_index("season")),
        "",
        "Margin model coefficients (points of expected margin per unit):",
        "",
        _md(res["game_coef"].rename("coef").to_frame()),
        "",
        "## Synthetic prices",
        "",
        "Price model fitted on 2026-27 rounds 2-9, tested on rounds 10-13:",
        "",
        _md(res["price_check"].to_frame("value")),
        "",
        "## Best-11 squad simulation (100 credits, actual points)",
        "",
        "`prev` = realistic pool (players who played their team's previous game; no injury news).",
        "`played` = known availability. `_avail` methods multiply by the availability model's",
        "P(plays). Coaches are picked with the game model for every method. 2023-24 to 2025-26 use",
        "synthetic prices, 2026-27 real ones.",
        "",
        "Average points per round:",
        "",
        _md(wide, ".1f"),
        "",
        "Share of the hindsight-optimal squad (%):",
        "",
        _md(pct, ".1f"),
        "",
        "All seasons pooled:",
        "",
        _md(allseasons, ".1f"),
        "",
        "## Top 25 features (share of total gain, last trained model)",
        "",
        _md(imp, ".3f"),
        "",
    ])
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(REPO_ROOT / "docs" / "backtest_models.md"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run()
    print(write_report(res, Path(args.report)))


if __name__ == "__main__":
    main()
