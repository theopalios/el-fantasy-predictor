# EuroLeague Fantasy Predictor

Predicts each round's best EuroLeague Fantasy Challenge squad: 5 starters, a sixth man, 4 bench
players, a captain and a head coach. It also recommends the best ≤4 trades from your current
roster.

How it works:

1. Models predict each player's and coach's expected fantasy score:
   * a **game model** (margin, win probability, coach points)
   * an **availability model** (does he play?)
   * a **player points model** (PIR if he plays)
2. An **integer-programming optimizer** picks the squad under the game's rules: budget,
   positions, formations, the 6-per-team limit, trade limits and the captain.

Status: **design phase**. See [`docs/design.md`](docs/design.md) for the variables, models, rules
and weekly flow. The game rules are encoded in [`config/game_rules.yaml`](config/game_rules.yaml).
