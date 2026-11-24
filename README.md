# EuroLeague Fantasy Predictor

Predicts each round's best EuroLeague Fantasy Challenge squad: 5 starters, a sixth man, 4 bench
players, a captain and a head coach. It also produces rankings by position, a captain shortlist
and a coach ranking. The recommendations are generic, not tied to any fantasy team. Optionally,
paste a roster to get the best ≤4 trades from it.

How it works:

1. Models predict each player's and coach's expected fantasy score:
   * a **game model** (margin, win probability, coach points)
   * an **availability model** (does he play?)
   * a **player points model** (PIR if he plays)
2. An **integer-programming optimizer** picks the squad under the game's rules: budget,
   positions, formations, the 6-per-team limit, trade limits and the captain.

Status: **data layer built, models next.** See [`docs/design.md`](docs/design.md) for the variables,
models, rules and weekly flow. The game rules are encoded in
[`config/game_rules.yaml`](config/game_rules.yaml) and were verified against the app's own scores.

## Setup

```bash
pip install -e ".[dev]"
```

## Data commands

All three commands use public endpoints; no login is needed.

```bash
python -m elfantasy.data.build            # EuroLeague 2016-17 -> now: games, box scores, rosters (~15 min first run, cached after)
```

```bash
python -m elfantasy.data.fantasy_build    # current-season fantasy tables: prices per round, positions, P(plays), id maps
```

```bash
python -m elfantasy.data.snapshot         # run BEFORE EACH ROUND'S DEADLINE; commit the file it writes to snapshots/
```

The app's pre-deadline state (probability of playing, projected lineups, ownership, injured
flags) can't be fetched after the round starts. Snapshots are the only way to keep it, so they
are committed to the repo.

No login token is needed. The collector can read `ELF_FANTASY_TOKEN` from a local `.env` file
(gitignored; never commit it). But as of 24 Nov 2026 the login-only endpoints add nothing: the app
has no odds for EuroLeague and its injury list is empty.

Processed tables go to `data/processed/*.parquet` (gitignored and rebuildable). Run the tests
with `pytest`.

## Backtest

```bash
pip install -e ".[model,dev]" tabulate
```

```bash
python -m elfantasy.backtest.walkforward   # baselines, round by round; writes docs/backtest_baselines.md
```

```bash
python -m elfantasy.backtest.models        # trained models vs baselines, 4-season squad simulation; writes docs/backtest_models.md
```
