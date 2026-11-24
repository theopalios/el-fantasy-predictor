# EuroLeague Fantasy Predictor: Design v0.2

*Status: design agreed, 24 Nov 2026. The data layer (M0/M1) is built; nothing is trained
yet. v0.2 adds the user's answers (generic recommendations, no paid data, token optional)
and what we learned from the data (verified scoring rules, public app endpoints).*

---

## 0. Summary

Each round, the system predicts every player's and coach's **expected fantasy score**.
An **integer-programming optimizer** then picks the squad that maximizes total expected
points under the game's rules. The recommendations are **generic, not tied to any
fantasy team**:

* **Main output: the best 11.** The best 10 players + coach within the budget (100 credits
  by default, adjustable), with starting five, sixth man, bench and captain.
* **Rankings any manager can use:** expected points and points per credit by position, a
  captain shortlist and a coach ranking.
* **Optional:** paste any roster and get the best ≤4 trades from it. This is not linked
  to an account.

The prediction is built from three components:

1. **Game model.** Predicts the margin of each game. From that we get the win probability
   (for the +10% win bonus) and the coach's expected points. It is trained on our own
   team ratings, because there is no free historical odds data. Free live odds can be
   blended in at prediction time.
2. **Availability model.** The probability that the player plays (injuries, rest,
   coach's choice). The app publishes its own probability of playing for every player.
3. **Player points model.** Expected PIR *given that he plays*. A gradient-boosted model
   on form, role, opponent and game-context variables.

`E[fantasy pts] ≈ P(plays) × E[PIR | plays] × (1 + 0.10 × P(team wins))` (exact form in §3)

Delivery is a Jupyter notebook. The MVP goes live this season.

---

## 1. Decisions taken

| Topic | Decision |
|---|---|
| Output | **Generic recommendations, not for a specific team** (v0.2): best 11 + rankings; trades only for a pasted roster |
| Fantasy prices, positions, injury flags | Pulled from the fantasy app's backend (fantaking/Dunkest). Everything we use is **public** (§8). A login token was tested and adds nothing: there are no in-app odds for EuroLeague and the injury list is empty. |
| Betting odds | **No paid data** (v0.2). No free historical EuroLeague odds dataset exists, and the app has no odds (§8). The MVP game model uses **our own team ratings only**. Free live odds (e.g. a free-tier odds API key) can be added later if the backtest shows the game model is the weak link. |
| Training seasons | 2016-17 onward (confirmed) |
| Objective | Overall season ranking → **risk-neutral**: maximize expected points |
| Price changes / team value | **Ignored in the objective** (points only). Prices still set the budget constraint. |
| In-round turns | **Pre-round only**: one lineup + captain before the round, no swap logic between game days |
| Delivery | Jupyter notebook |
| Timeline | MVP live this season (2026-27, currently Round 14 of 38) |

---

## 2. Game rules that drive the design

The full rules are encoded in [`config/game_rules.yaml`](../config/game_rules.yaml). The
optimizer and scorer read them from there, never from hard-coded constants. The rules
that matter for modelling:

* **Squad:** 100 credits for 4 G + 4 F + 2 C + 1 head coach. At most 6 players from the
  same real team.
* **Lineup:** 5 starters in one of the formations 2-2-1, 1-2-2, 2-1-2, 1-3-1, 3-1-1 (G-F-C).
  The starters, the **sixth man** and the coach score 100%. The other **4 bench players
  score 50%**. The **captain** (a starter) scores ×2.
* **Player score = PIR + 10% of |PIR| if his team wins.** A negative score therefore rises:
  −2 becomes −1.8. This was verified against the app on all 2,914 player-rounds of 2026-27
  R1–R13, with 100% match. PIR is exactly the official "Valuation"
  in EuroLeague box scores, so historical targets can be computed from public data.
  Checked on a real line (Shane Larkin, 2025-26 R1): (14 pts + 4 reb + 5 ast + 0 stl +
  0 blk + 4 fouls drawn) − (7 missed FG + 2 missed FT + 2 TOV + 1 blocked + 3 fouls) = 12,
  which equals the `Valuation` field.
* **Coach score** depends only on the final margin:

  | Result | 1–10 or OT | 11–20 | 21+ |
  |---|---|---|---|
  | Win | +10 | +20 | +25 |
  | Loss | −5 | −10 | −20 |

* **One fantasy round = one EuroLeague round = one game per team.** A double-round week is
  two fantasy rounds with a short trade window between them.
* **Trades:** 4 per round, and changing the coach counts as one. The windows after
  R6, R13, R18, R23, R28 and R34 are unlimited, as are the play-in, playoffs and Final Four.
* **Prices** change after every round, driven by the score and the starting price. The
  exact formula is not public. Selling refunds the current price.

Rule checks:

* **Win bonus on negative scores:** verified. It is 10% of the absolute score (see above).
* **Margin of exactly 20:** verified from app coach points. 20 → +20, −20 → −10, 21 → +25,
  and every overtime game → +10/−5.
* **Coach in the 6-per-team limit:** still unknown. The app models the coach as an 11th
  "player" with a team, so the optimizer **counts the coach** (the stricter rule). A squad
  valid under the strict rule is valid either way.
* Also confirmed from the app config: captain ×2, max 6 per club, 5 bench slots
  (sixth man + 4).

---

## 3. What we predict (targets)

For player *i* in round *r*:

```
FP_i,r = 0                                    if he does not play
       = PIR_i,r + 0.10·|PIR_i,r|·[team wins]  otherwise
```

We split this into components that we model separately:

| Component | Symbol | Model | Training target (historical) |
|---|---|---|---|
| Plays at all | `p_play` | Classifier / rules | 1 if he appears with minutes > 0 in the box score of a game his team played |
| PIR if he plays | `μ_pir` | Gradient-boosted regressor (+ quantile versions for p10/p90) | `Valuation` from the box score, rows with minutes > 0 |
| Team margin | `M ~ Normal(μ_g, σ_g)` | Game model | Final score difference |

Combined:

```
p_win      = Φ(μ_g / σ_g)
E[FP_i]    = p_play × (μ_pir + 0.10 × p_win × E|PIR|)     (≈ μ_pir·(1 + 0.10·p_win) when μ_pir ≫ 0)
E[Coach_k] = Σ_buckets points_b × P(M ∈ bucket_b)
```

PIR and winning are positively correlated, so `E[PIR × W] > E[PIR]·P(W)`. This makes the
formula slightly underestimate players on winning teams. The effect is ≤10% of a small
covariance and is acceptable for v1. In v2 we can fit `μ_pir|win` and `μ_pir|loss`
separately.

---

## 4. Architecture

```mermaid
flowchart LR
  subgraph Sources
    EL[EuroLeague stats API<br/>schedule, box scores, rosters]
    FA[Fantasy app API<br/>prices, positions, P plays, projected lineups, ownership]
    OD[Free live odds<br/>in-app odds with token, optional free-tier API]
    MO[Manual overrides<br/>injury news, lock/exclude]
  end
  EL --> DB[(DuckDB / Parquet store)]
  FA --> DB
  OD --> DB
  DB --> FE[Feature builder<br/>as-of round r]
  MO --> FE
  FE --> GM[Game model<br/>margin, P(win), coach pts]
  FE --> AV[Availability<br/>P(plays)]
  FE --> PM[Player model<br/>E PIR given plays]
  GM --> PM
  GM --> EP[Expected fantasy pts<br/>per player and coach]
  AV --> EP
  PM --> EP
  EP --> OPT[ILP optimizer<br/>rules from game_rules.yaml]
  EP --> RK[Rankings by position,<br/>captain shortlist, coaches]
  OPT --> B[Best 11 + lineup + captain]
  OPT -.optional.-> A[Trades for a pasted roster]
```

---

## 5. Variables (feature catalogue)

**Leakage rule:** every feature for round *r* uses only information available before the
first tip-off of round *r*. That means box scores of earlier games, the app snapshot taken
before the deadline, and odds as of the deadline. The feature builder takes an `as_of`
timestamp, and the backtester calls it the same way the live notebook does.

"MVP" marks variables in the first trained model. "v2" marks variables added once the data
exists or the MVP is stable.

### A. Player form and production (source: EuroLeague box scores)

| Variable | Definition | Phase |
|---|---|---|
| `pir_l1`, `pir_l3`, `pir_l5`, `pir_l10` | Mean PIR over the last N games played | MVP |
| `pir_ewm_h3`, `pir_ewm_h8` | Exponentially weighted PIR, half-life 3 and 8 games | MVP |
| `pir_season_mean`, `pir_season_std`, `gp_season` | Season to date | MVP |
| `pir_prev_season`, `gp_prev_season` | Previous EuroLeague season (EuroCup if he was there; same API, competition `U`) | MVP |
| `min_l3`, `min_ewm`, `min_season`, `min_slope_l5` | Minutes: level and trend | MVP |
| `pir_per36_ewm`, `pir_per36_season` | Production rate per 36 minutes | MVP |
| `start_rate_l5` | Share of the last 5 games started (`IsStarter`) | MVP |
| `usage_per36` | (FGA + 0.44·FTA + TOV) per 36 | MVP |
| `rate_<stat>_per36` | Per-36 rate of each PIR component: PTS, OREB, DREB, AST, STL, BLK, TOV, fouls drawn, fouls committed, missed FG, missed FT, blocks against. Lets the model learn matchups (e.g. rebounders vs weak rebounding teams). | MVP |
| `fg_pct_l5_vs_season`, `fg3_pct_l5_vs_season` | Hot/cold shooting, expected to regress to the mean | MVP |
| `plusminus_ewm` | Plus/minus trend (a role-quality signal) | v2 |
| `home_away_split` | Shrunk home − away PIR difference | v2 |

### B. Role and opportunity

| Variable | Definition | Phase |
|---|---|---|
| `rotation_rank` | Rank within team by minutes EWMA | MVP |
| `team_pir_share_ewm` | Player's share of team PIR | MVP |
| `vacated_min_team`, `vacated_min_pos` | Season-average minutes of teammates who are OUT this round (all / same position) | MVP |
| `games_with_team`, `is_new_signing` | Integration / small-sample flags | MVP |
| `first_game_back` | First game after ≥1 missed game (likely minutes restriction) | MVP |
| `vacated_usage_team` | Usage-weighted version of vacated minutes | v2 |
| `coach_changed_recently` | Team changed head coach in the last 5 games | v2 |

### C. Opponent

| Variable | Definition | Phase |
|---|---|---|
| `opp_drtg_ewm` | Opponent points allowed per 100 possessions | MVP |
| `opp_pace_ewm` | Opponent possessions per 40 minutes (more possessions → more stats) | MVP |
| `opp_dvp_pos` | Fantasy points the opponent allows to this fantasy position (G/F/C), relative to the league average, for the season and the last 5 games | MVP |
| `opp_dreb_pct`, `opp_oreb_pct` | Rebounding environment | MVP |
| `opp_tov_forced_rate`, `opp_stl_rate` | Pressure on ball handlers | MVP |
| `opp_pf_rate` | Fouls the opponent commits → our fouls drawn and free throws | MVP |
| `opp_blk_rate` | Blocks-against risk | MVP |
| `opp_vacated_min` | Opponent minutes OUT through injury | MVP |
| `player_vs_opp_hist` | Heavily shrunk history vs this opponent (weak signal) | v2 |

### D. Game context

| Variable | Definition | Phase |
|---|---|---|
| `is_home` | Home game | MVP |
| `spread`, `total` | Market point spread (team's perspective) and over/under | MVP (live) |
| `exp_margin`, `p_win` | Output of the game model (§6.1) | MVP |
| `team_exp_points` | (total + exp_margin) / 2: expected team scoring, a strong scale factor for PIR | MVP |
| `rest_days` | Days since the team's last EuroLeague game | MVP |
| `second_game_double_week` | Second game in a double-round week (2 games in ~3 days) | MVP |
| `phase`, `round_number` | Regular season / play-in / playoffs; round index | MVP |
| `travel_km` | Distance from the previous venue | v2 |
| `domestic_game_between` | Domestic-league game since the last EuroLeague game (fatigue). Needs domestic schedules. | v2 |
| `stakes` | Late-season motivation (clinched / eliminated / fighting for top 6 / play-in) | v2 |

### E. Availability

| Variable | Definition | Phase |
|---|---|---|
| `app_p_play` | The app's own probability of playing (0 / 0.25 / 0.5 / 1), from the pre-deadline snapshot | MVP (live; history only from our snapshots) |
| `app_projected_status` | App's projected starter / bench / out | MVP (live; history only from snapshots) |
| `app_injured` | Injured flag from the player profile | MVP (live) |
| `manual_status` | Your override in the notebook: OUT / DOUBTFUL / PROBABLE / FIT | MVP |
| `missed_streak` | Consecutive team games missed | MVP |
| `dnp_rate_season` | Share of team games missed this season | MVP |

### F. Team strength (used by the game model and as player context)

| Variable | Definition | Phase |
|---|---|---|
| `team_ortg_ewm`, `team_drtg_ewm`, `team_net_ewm` | Offensive / defensive / net rating per 100 possessions | MVP |
| `team_pace_ewm` | Possessions per 40 minutes | MVP |
| `team_elo` | Margin-aware Elo carried across seasons with regression to the mean | MVP |

### G. Fantasy market (source: fantasy app)

| Variable | Definition | Phase |
|---|---|---|
| `price` | Credit value before the round. The app's match log gives the **real per-round price history for 2026-27**; older seasons have none. Used as the **prior for cold starts** (§6.4) and as a feature in a 2026-27 correction model. | MVP (prior) |
| `popularity` | % of fantasy teams owning the player (crowd signal); from the profile endpoint, history from our snapshots | v2 |
| `price_delta_l1`, `price_delta_l3` | Recent price moves (match log `PLUS` column) | v2 |
| `app_avg_fantasy_pts` | App's own season average, used as a sanity check against our computed target | MVP (QA only) |

### H. Static

| Variable | Definition | Phase |
|---|---|---|
| `fantasy_pos` | G / F / C exactly as the app classifies him (used by the optimizer) | MVP |
| `age`, `height`, `el_seasons` | From EuroLeague player profiles | v2 |

---

## 6. Models

### 6.1 Game model (margin → win probability and coach points)

* `M_g ~ Normal(μ_g, σ_g)`, from the team's perspective.
* Own-rating estimate: `μ_own = home_court + (team_net − opp_net) × pace_factor + injury_adj + rest_adj`.
  The home-court value, the coefficients and `σ_g` are fitted on history. `injury_adj`
  uses the vacated minutes weighted by player quality.
* Market estimate: `μ_mkt = −spread`. The spread is already a margin forecast, so it
  needs **no historical training to be useful**. This matters because there is no free
  historical EuroLeague odds data.
* Blend: `μ_g = w·μ_mkt + (1−w)·μ_own`. `w` starts at a fixed ~0.7 when live odds are
  available and is re-fitted once enough rounds of stored odds exist. Without odds,
  `w = 0`.
* **Training without odds:** the player model's game-context features (`exp_margin`,
  `p_win`, `team_exp_points`) come from `μ_own` in training. At prediction time they come
  from the blend. Because the blend is better calibrated, the train/serve shift is small
  and in the safe direction.
* Outputs: `p_win = Φ(μ_g/σ_g)`, `team_exp_points`, and the coach's expected points from
  the bucket table above.
* Checks: Brier score / log-loss of `p_win`, MAE of the margin, and calibration plots.

### 6.2 Availability model `P(plays)`

The app publishes a **probability of playing** for every player (0, 0.25, 0.5 or 1) and a
projected starter/bench/out status. There is **no history** of either: past matches show
today's projection, which we verified. The only history is the snapshots we collect from
Round 14 onward. The MVP therefore combines:

| Input | Use |
|---|---|
| Manual override (OUT / DOUBTFUL / PROBABLE / FIT) | Wins over everything: 0 / 0.30 / 0.75 / model |
| App `p_play` = 0 or injured flag | 0.00 |
| App `p_play` = 0.25 / 0.5 | Used as-is, then recalibrated once snapshots accumulate |
| App `p_play` = 1 | A small logistic model on `missed_streak`, `dnp_rate_season`, `rotation_rank` and `first_game_back`, trained on box-score history (who was dressed but didn't play) |

After ~10 rounds of snapshots matched to who actually played, a learned calibration of
the app's probabilities replaces the fixed mapping (v2).

### 6.3 Player points model `E[PIR | plays]`

* **Algorithm:** LightGBM regressor (L2 loss, or Huber if outliers dominate). We also
  train p10/p90 quantile models. The objective is risk-neutral, so they are only for
  display and sanity checks.
* **Unit:** one row per (player, game) with minutes > 0, built only from features in §5.
* **Training data:** EuroLeague 2016-17 → now. That covers the single round-robin era,
  about 10 seasons and roughly 70k player-game rows. 2019-20 (cut short) and the voided
  2021-22 games get flags rather than being dropped.
* **Why not predict FP directly?** Splitting availability from production keeps the
  learned model clean (no zeros from injuries mixed in) and lets live injury news enter
  exactly.
* **v2 refinement:** split into `E[minutes] × E[PIR per minute]`. This handles role
  changes (teammate injured, new signing) more explicitly.

### 6.4 Cold start (new signings, early season)

Few EuroLeague games means the form features are noisy. We use **empirical-Bayes
shrinkage**:

```
form_shrunk = (n·form_observed + k·prior) / (n + k)
```

The prior is, in order of preference:

1. Last season in EuroLeague or EuroCup, adjusted for the league level.
2. **Price-implied PIR.** The app's price is the game-maker's own expectation, fitted as
   `PIR ≈ a + b·price` on current-season data.
3. The position average.

`k` is tuned in the backtest.

### 6.5 Baselines we must beat

1. Season-average PIR.
2. EWMA PIR (half-life ~5 games).
3. Price-implied points.

If the model does not clearly beat (2) and (3) in the backtest, it does not ship.

---

## 7. Optimizer (integer linear program)

Solver: PuLP + CBC (free) or OR-Tools. The problem is tiny (~350 players, 20 coaches) and
solves in under a second.

**Decision variables** (binary), for each player *i*: `s_i` starter, `m_i` sixth man,
`b_i` bench, `c_i` captain. For each coach *k*: `h_k`. Squad membership: `x_i = s_i + m_i + b_i`.

**Objective** (risk-neutral, points only):

```
max  Σ_i E_i · (s_i + m_i + 0.5·b_i + c_i)  +  Σ_k E_k · h_k
```

The captain term `+c_i` adds the second copy of the captain's score.

**Constraints:**

| Rule | Constraint |
|---|---|
| Squad composition | Σ_G x = 4, Σ_F x = 4, Σ_C x = 2, Σ_k h = 1 |
| Roles | Σ s = 5, Σ m = 1, Σ b = 4, s_i + m_i + b_i ≤ 1 |
| Formation | 1 ≤ Σ_G s ≤ 3, 1 ≤ Σ_F s ≤ 3, 1 ≤ Σ_C s ≤ 2 (generates exactly the 5 allowed formations) |
| Captain | c_i ≤ s_i, Σ c = 1 |
| Budget | Σ price_i·x_i + Σ price_k·h_k ≤ budget |
| Team limit | Σ_{i ∈ team t} x_i ≤ 6 for every real team t |
| User locks/excludes | x_i = 1 / x_i = 0 |

**Main mode (generic best 11):** the model above with `budget` = 100 credits by default.
Managers whose team value has grown can raise it in the notebook. The notebook also
lists:

* the top players per position by expected points and by expected points per credit
* the 5 best captain options
* all 20 coaches ranked

**Optional roster mode (any pasted roster, no account link):**

* `budget = bank + Σ current prices of the roster`.
* Trades: `Σ_{i ∉ roster} x_i + Σ_{k ≠ roster coach} h_k ≤ 4`. The limit is removed in
  unlimited windows.
* A trade is only made if it improves expected points by more than a threshold `ε`
  (default 1.0 pt), to avoid churning on noise.

**Optional lookahead (to test in the backtest):** with only 4 trades per round, a player
with three good matchups ahead can beat one with a single great matchup. A rolling
multi-round ILP over the next H rounds links squads through the trade constraints and
keeps only the first round's decisions. The default is H = 1. We will compare H = 2 and
H = 3 on simulated seasons.

---

## 8. Data sources and pipeline

| Source | What | Access (verified 24 Nov 2026) |
|---|---|---|
| EuroLeague stats API: `v2/.../games` (schedule, results, OT), `v2/.../games/{code}/stats` (box score), `v2/.../clubs/{club}/people` (rosters with position, height, birth date, tenure dates; head coaches) | Every PIR component, minutes, starter flag, ±, coach of record | **Public.** 2016-17 → 2026-27 R13 downloaded (`python -m elfantasy.data.build`). The v2 stats endpoint is 4× faster than the legacy `Boxscore` and carries roster codes. |
| Fantasy app, **public** endpoints (league 10): `config`, `schedules/{s}/matchdays/{md}`, `matches/{id}/lineups`, `basketball/leagues/10/players/{id}/match-log/rendered`, `players/{id}/trend`, `players/{id}/profile` | Fantasy position, price, projected starter/bench/out, **probability of playing**, per-round **price history** (price before the round + change), fantasy points, ownership, injured flag; coaches included | **Public, no login.** Current-season tables: `python -m elfantasy.data.fantasy_build`. Pre-deadline state: `python -m elfantasy.data.snapshot`. |
| Fantasy app, login-only: `odds/matches/{id}`, `leagues/10/injuries`, `players-lists/...` | In-app bookmaker odds, injury list, player list | **Tested with a token on 24 Nov 2026: nothing useful.** Odds are empty for every match (the league config has `odds_comparator: null`, so EuroLeague Fantasy has no odds feature). The injury list is empty. The player list is a 10-player subset of the public lineups. The token is not needed. |
| Odds history | Closing spreads/totals | **No free dataset found.** The only EuroLeague odds set on Kaggle is a 50-row sample of a paid file, and odds APIs charge for history. We start our own history with the snapshots. |
| Optional free live odds | Spread + total | The only route left is a free-tier odds API key (no payment, but needs a sign-up). Not needed for the MVP; reconsider after the backtest. |
| Manual overrides | Injury news not yet in the app, lock/exclude players | A cell in the weekly notebook |

**Storage:** raw JSON cached on disk (so we never re-download history), normalized into
DuckDB/Parquet tables:

* `games`
* `player_games`
* `teams`
* `fantasy_snapshots` (player × round × timestamp: price, status, popularity)
* `fantasy_scores`
* `odds_snapshots`
* `overrides`

**Important:** price history can be recovered from the match logs, but the app's
**probability of playing, projected lineups, injury flags and ownership cannot**. Past
matches show only today's projection. The **snapshot collector must run before every
round's deadline**. Its output is committed to `snapshots/`, because a cloud container is
temporary and these files cannot be re-downloaded. The first snapshot (Round 14) was taken
22 minutes before tip-off on 24 Nov 2026.

---

## 9. Training and evaluation

* **Walk-forward validation:** for each round *r* in the evaluation seasons, train on
  everything before *r* (or refit per season and update features per round) and predict *r*.
  * Tuning: 2023-24 and 2024-25.
  * Held-out test: 2025-26 and 2026-27 R1–R13.
* **Prediction metrics** (players who played): MAE and RMSE of FP, Spearman rank
  correlation within each round, and calibration of the p10/p90 bands.
* **Game model metrics:** Brier score and log-loss of `p_win`, margin MAE, and coach-points MAE.
* **Decision metric (the one that matters):** a **season simulation**. Start from the
  best-11 squad at R1, then for each round let the optimizer make its ≤4 trades using
  predictions, and score with actual results. We compare season totals across:
  * model vs the three baselines
  * horizon H = 1 / 2 / 3
  * trade threshold ε
* **Historical prices problem:** prices don't exist for seasons before 2026-27. For those
  simulations we use a **synthetic price model** (price after round *r* as a function of
  the round's score and the previous price), fitted on the **real 2026-27 price history**
  from the match logs (~5,000 player-rounds). 2026-27 itself is simulated with real prices.

---

## 10. Weekly run (notebook)

`notebooks/weekly_picks.ipynb`, run after the previous round ends and before the next
deadline:

1. **Refresh:** new box scores, app snapshot (prices, P(plays), projected lineups,
   ownership), odds if available.
2. **Overrides:** edit a small table (OUT / DOUBTFUL / FIT, lock, exclude), and optionally
   set the budget or paste a roster.
3. **Predict:** game model → availability → player model → `E[FP]` for all players and coaches.
4. **Optimize:** the best 11 (and trades if a roster was pasted).
5. **Report:**
   * Best 11 with lineup, sixth man, bench and captain.
   * Rankings by position (points and points per credit), captain shortlist, coach ranking.
   * Projection table with p10/p50/p90.
   * Top feature contributions per pick (SHAP), so you can sanity-check the reasoning.
   * Trades in/out with the gain for each, if a roster was pasted.
6. **Log** the recommendation, and the actual result after the round, to track live
   performance.

---

## 11. Repository layout (planned)

```
config/
  game_rules.yaml          # rules, verified against app data (done)
src/elfantasy/
  rules.py  scoring.py  paths.py                                   (done)
  data/      http.py  euroleague.py  build.py                      (done)
             fantasy_app.py  fantasy_build.py  snapshot.py         (done)
  features/  player.py  team.py  opponent.py  context.py  build.py
  models/    game.py  availability.py  player_points.py  priors.py
  optimize/  squad.py
  backtest/  walkforward.py  season_sim.py  synthetic_prices.py
notebooks/
  01_data_exploration.ipynb
  02_backtest.ipynb
  weekly_picks.ipynb
snapshots/                 # committed: pre-deadline app state per round (not recoverable later)
tests/
data/                      # gitignored: raw API cache + processed parquet (rebuildable)
```

---

## 12. Milestones (MVP this season)

| # | Milestone | Done when |
|---|---|---|
| M0 | **Fantasy snapshot collector** running every round | **Done.** First snapshot taken for R14. Automated with GitHub Actions (`.github/workflows/snapshot.yml`), Tue–Fri 14:41 UTC; the earliest 2026-27 tip-off is 16:00 UTC. |
| M1 | Data layer | **Built**: EuroLeague 2016-17 → now, current-season fantasy tables, team/player id maps, verified scoring |
| M2 | Features + baselines + backtest harness | Walk-forward metrics for the 3 baselines |
| M3 | Game model, player model, availability rules, ILP (both modes) | Model beats baselines in the walk-forward test and the season simulation |
| M4 | Weekly notebook live | First real recommendation, ideally ready for the **unlimited window after R18 (19–22 Dec)**, when anyone can rebuild their whole squad into the recommended 11 |
| v2 | Learned availability, minutes × rate split, lookahead H>1, market features, fatigue/travel, domestic leagues | Each addition kept only if it improves the season simulation |

---

## 13. Open questions

Answered in v0.2:

* Token: yes, but now optional.
* Paid data: no.
* Output: generic, not for a specific team.
* Training seasons: from 2016-17.
* Two of the three rule checks were resolved from data.

* Automatic snapshots: run by GitHub Actions.

Still open:

1. **Coach in the 6-per-club limit:** unresolved. We use the stricter rule, which is
   always safe.

---

## Sources

* Official rules (classic mode): <https://fantaking.gitbook.io/euroleague-fantasy-challenge-rules/>, in particular
  the [initial team](https://fantaking.gitbook.io/euroleague-fantasy-challenge-rules/classic-mode/initial-team),
  [managing your team](https://fantaking.gitbook.io/euroleague-fantasy-challenge-rules/classic-mode/managing-your-team),
  [players scoring](https://fantaking.gitbook.io/euroleague-fantasy-challenge-rules/classic-mode/players-scoring),
  [coach scoring](https://fantaking.gitbook.io/euroleague-fantasy-challenge-rules/classic-mode/coach-scoring) and
  [price variations](https://fantaking.gitbook.io/euroleague-fantasy-challenge-rules/classic-mode/quotations-and-price-variations) pages
* Captain ×2 change (2025-26): <https://www.eurohoops.net/en/euroleague/1863394/euroleague-fantasy-upgrades/>
* 2026-27 season overview: <https://basketballsphere.com/en/euroleague-fantasy/>,
  <https://www.dunkest.com/en/euroleague/news/fantasy-basketball/244469/euroleague-fantasy-2026-27-minutes-roles-and-schedule>
* Odds data search: [Kaggle European Basketball & FIBA Betting Odds](https://www.kaggle.com/datasets/oliviersportsdata/european-basketball-and-fiba-betting-odds)
  (50-row sample; full file paid), [odds-api.io historical](https://odds-api.io/historical-odds-api) (paid history)
