# How to run the EuroLeague Fantasy predictor

This guide covers the weekly routine, two ways to run it (ask Claude, or on your own computer),
every setting, how to read the output, and what to do when something goes wrong.

---

## 1. The weekly routine (5 minutes)

| When | What | Who |
|---|---|---|
| Daily 10:41 and 14:41 UTC | The app's line-ups, play probabilities, injury flags and ownership are saved to `snapshots/` (history for later calibration) | GitHub Actions, automatic |
| As late as possible before the first tip-off | Run the weekly picks (section 2 or 3). Line-ups and injury flags are fetched live from the app, so later = fresher news. | You (or Claude) |
| Before the first tip-off | Make the trades / set the line-up and captain in the app | You |
| After the round | `python -m elfantasy.evaluate`: how the picks actually scored | You (or Claude) |

- **Deadline = the round's first tip-off.** In 2026-27 the first tip-off is never earlier than
  16:00 UTC, and rounds start on Tuesday, Wednesday or Thursday. The notebook prints the exact
  deadline.
- **Double-round weeks** are two separate rounds (e.g. Tuesday and Thursday). Run the picks once
  for each round.
- **Unlimited trade windows** come after rounds 6, 13, 18, 23, 28 and 34, and in the play-offs.
  Then you can switch your whole squad to the recommended best 11.

---

## 2. Easiest: ask Claude

In a Claude Code session on this repository, ask for example:

> Run the weekly picks for round 15 with a budget of 101.4. Mike James is out.

Claude refreshes the data, runs the notebook pipeline, shows you the squad, and saves the
recommendation to `recommendations/`.

A new cloud session starts without the downloaded data. Its first run downloads about 10 seasons
of box scores (15–60 minutes, because the EuroLeague API limits how fast we can download). The
same session is fast afterwards.

---

## 3. On your own computer

### 3.1 One-time setup (about 30–60 minutes, mostly waiting)

You need **Python 3.11 or newer** (tested on 3.11, 3.12 and 3.13), **git**, and an internet
connection. Check with `python3 --version`. On Windows, use `py` instead of `python3`.

1. Get the code:

   ```bash
   git clone https://github.com/theopalios/el-fantasy-predictor.git
   ```

   ```bash
   cd el-fantasy-predictor
   ```

2. Create and activate a virtual environment, which keeps these packages separate from the rest
   of your computer:

   ```bash
   python3 -m venv .venv
   ```

   ```bash
   source .venv/bin/activate
   ```

   On Windows: `.venv\Scripts\activate`. Activate it again each time you open a new terminal.

3. Install the project with everything it needs:

   ```bash
   pip install -e ".[model,notebook,dev]"
   ```

4. Check that it works (should end with `passed`, no failures):

   ```bash
   pytest -q
   ```

5. Download the EuroLeague history (2016-17 → now). The first run takes 15–60 minutes because of
   the API's rate limit. If you see `HTTP 429 ... retrying`, that is normal: it waits and
   continues. If you stop it, run it again; everything already downloaded is cached.

   ```bash
   python -m elfantasy.data.build
   ```

6. Download this season's fantasy data (player list, prices per round, id maps), about 3 minutes:

   ```bash
   python -m elfantasy.data.fantasy_build
   ```

7. Optional check that the data is consistent:

   ```bash
   python -m elfantasy.data.qa
   ```

   Expect `PIR ... != Valuation: 0.0000%`, coach points matching the app 100%, and players
   matching the app on about 98.7% of player-rounds. A short "mismatches" table follows: those
   are the app's stats feed differing from the official box score by a point or two. That's
   normal.

The data is stored in `data/` (not in git) and stays on your computer, so you only do this once.

### 3.2 Every round: the notebook

1. Open a terminal in the project folder, activate the environment (step 2 above), and update the
   code and snapshots:

   ```bash
   git pull
   ```

2. Start Jupyter:

   ```bash
   jupyter lab notebooks/weekly_picks.ipynb
   ```

3. In the **Settings** cell, set your budget and any injury news (section 4), then use
   **Run → Run All Cells**. The cell after Settings does the work:
   - refreshes the data (about a minute)
   - fetches the app's line-ups live
   - trains the models (30 seconds to 2 minutes)
   - picks the squad

4. Read the squad, captain and coach (section 5), and make the moves in the app.

With `SAVE = True` the recommendation is saved to `recommendations/<season>/mdNN_<time>.md`, plus
a `.json` used for tracking. Use `SAVE = False` for test runs, because `evaluate` scores every
saved file.

### 3.3 Every round: command line instead of the notebook

Next round, 100 credits, refresh data first, save the result:

```bash
python -m elfantasy.weekly --refresh --save
```

A specific round, your own budget, injury news:

```bash
python -m elfantasy.weekly --matchday 15 --budget 101.4 --out "Mike James" --doubtful "Kevin Punter" --save
```

`--out`, `--doubtful`, `--probable`, `--exclude` and `--lock` can be repeated. `--exclude` and
`--lock` also accept coaches. For roster trades, use the notebook or ask Claude.

### 3.4 After the round

```bash
python -m elfantasy.evaluate
```

This scores every saved recommendation whose round is finished against the real fantasy points
(the app's own points when available). It compares each with the expected total and the best
possible squad in hindsight, and writes `recommendations/<season>/tracking.md`.

If the round just ended, first fetch the new box scores and the app's per-round prices and points.
The hindsight comparison needs those.

```bash
python -m elfantasy.data.build
```

```bash
python -m elfantasy.data.fantasy_build
```

### 3.5 Replay a past round

Each saved snapshot lets you see what the model would have picked before that round's deadline.
It uses the snapshot's line-ups and only box scores from before the deadline:

```bash
python -m elfantasy.weekly --snapshot snapshots/2026-27/md14/20261124T165522Z.json.gz --save
```

Pick the last snapshot taken before the deadline; the file name is the UTC time it was taken.
Saved replays are scored by `python -m elfantasy.evaluate` like any other recommendation.

---

## 4. Settings (notebook Settings cell)

| Setting | Default | What it does |
|---|---|---|
| `MATCHDAY` | `None` | `None` = the next round with unplayed games. Or a round number. |
| `BUDGET` | `100.0` | Credits for the best 11. Use your team value + bank if it has grown. |
| `REFRESH` | `True` | Fetch new box scores and the current player list before predicting. |
| `OVERRIDES` | `{}` | News the app has not picked up, e.g. `{'Mike James': 'OUT', 'Nadir Hifi': 'DOUBTFUL'}`. Values: `OUT` (0%), `DOUBTFUL` (30%), `PROBABLE` (75%), `FIT` (use the model even if the app lists him as out). Anything else is an error. |
| `LOCKED` | `[]` | Players (or a coach) who must be in the squad. |
| `EXCLUDED` | `[]` | Players (or coaches) to leave out. |
| `ROSTER` | `None` | Your current 10 players + coach (names), to get the best trades instead of a fresh squad. |
| `BANK` | `0.0` | Unspent credits (roster mode). |
| `MAX_TRADES` | `4` | Trades allowed this round (the coach counts as one). Use 11 in an unlimited window. A trade is only suggested if it gains at least 1 expected point. |
| `SAVE` | `True` | Save the result to `recommendations/` (set `False` for test runs). |

Names can be full names or any unique part ("Vezenkov"). If a name matches several players, you
get an error listing them.

---

## 5. Reading the output

| Column | Meaning |
|---|---|
| `role` | Fantasy role. `starter (C)` = captain (double points). The sixth man also scores full points. Bench = half. |
| `exp_pts` | Expected fantasy points = P(plays) × points if he plays. **This is what the optimizer maximizes.** |
| `pts_if_plays` | The player model's prediction if he plays (includes the team-win bonus on average). |
| `p_play` | Probability he plays: your override > the app's probability > our availability model (details in `docs/design.md` §6.2). |
| `real_lineup` | The app's projected real-life line-up: starter / bench / out. Not the fantasy role. |
| `pts_per_credit` | Expected points per credit; high = good value. |
| Coaches `p_win`, `exp_pts` | Win probability and expected coach points from the game model (margin buckets: +25 … −20). |

The **"why"** section lists, for each pick, the features that moved the prediction most, in
fantasy points relative to an average player. For example, "longer-run form pir, weighted
(half-life 8 games) = 23.0 → +4.3" means his strong form adds 4.3 expected points.

**Games moved out of their round** (postponed to after the next round starts): under the game's
rules, those teams' players and coaches score their season average, and the captain is not
doubled. The tool applies this automatically and adds a note.

**Expectations:** a single game is noisy (an average error of about 6 points per player). The
model's edge shows over many rounds. In the backtest (4 seasons, no injury news) it scored about
149 points per round, against 136 for the best simple method.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `HTTP 429 ... retrying in 300s` | The EuroLeague API rate limit. It waits and continues on its own. Re-running later also works, because downloads are cached. |
| `FileNotFoundError: ... data/processed/...parquet` | The data hasn't been downloaded yet. Run steps 5 and 6 of section 3.1. |
| `'Name' is ambiguous or unknown (...)` | Use a longer part of the name, or the exact name from the list in the error. |
| `optimizer status Infeasible` | The budget is too low for the locked players, or locks break a rule (e.g. more than 6 from one club). |
| Note: "The deadline has passed" | You ran it after the first tip-off; the round has started. Use `MATCHDAY` for the next round. |
| `Round 99 does not exist; valid rounds are 1 to 38` | Use a round number from the season (`None` = the next round). |
| `override for '...' must be one of OUT, DOUBTFUL, PROBABLE, FIT` | Fix the spelling of the override value. |
| Note: "N players have not played a EuroLeague game this season" | Normal. They get a low play probability; override with `PROBABLE` if you know one will play. |
| Snapshots stopped appearing | Check the repo's **Actions** tab on GitHub ("Fantasy snapshot"). On public repos, scheduled workflows pause after 60 days without commits; re-enable them there. A run that saves nothing usually means the line-ups had not changed since the last snapshot. |
| Jupyter shows the wrong Python / `No module named elfantasy` | Activate the virtual environment before `jupyter lab`, and run `pip install -e ".[model,notebook,dev]"` in it. |
| You moved the data folder | Set `ELF_DATA_DIR=/path/to/data` before running. |

---

## 7. Maintenance (optional)

| Task | Command |
|---|---|
| Re-run the baseline backtest | `python -m elfantasy.backtest.walkforward` |
| Re-run the model backtest (writes `docs/backtest_models.md`) | `python -m elfantasy.backtest.models` |
| Refresh this season's price history (for evaluation and backtests) | `python -m elfantasy.data.fantasy_build` |
| Take a snapshot by hand (if the Action failed) | `python -m elfantasy.data.snapshot`, then commit the new file in `snapshots/` |
| Regenerate the notebook after editing its cells | `python scripts/make_weekly_notebook.py` |
| Run the tests | `pytest -q` |

A login token is **not** needed. The app's login-only endpoints (odds, injury list) were empty
for EuroLeague Fantasy when tested.
