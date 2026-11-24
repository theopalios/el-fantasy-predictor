"""Regenerate notebooks/weekly_picks.ipynb (keeps the notebook's cells in reviewable code)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

cells = [
    nbf.v4.new_markdown_cell(
        "# EuroLeague Fantasy: weekly picks\n\n"
        "Run this before each round's deadline (the first tip-off). A GitHub Action saves the app's\n"
        "line-ups at 14:41 UTC on game days; running after that uses the freshest injury news.\n\n"
        "1. Adjust the settings cell (budget, players you know are out, optional roster).\n"
        "2. **Run all**. It refreshes the data, trains the models on all history (~1 min) and\n"
        "   picks the best 11 within your budget.\n"
        "3. The result is saved as markdown under `recommendations/`."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from IPython.display import Markdown, display\n"
        "from elfantasy.weekly import recommend\n\n"
        "pd.set_option('display.max_columns', 20)\n"
        "pd.set_option('display.width', 200)"
    ),
    nbf.v4.new_markdown_cell("## Settings"),
    nbf.v4.new_code_cell(
        "MATCHDAY = None        # None = the next round; or a round number, e.g. 15\n"
        "BUDGET = 100.0         # your team value + bank if it has grown\n"
        "REFRESH = True         # fetch new box scores and the current player list first\n\n"
        "# Injury news the app has not picked up yet: 'OUT', 'DOUBTFUL' or 'PROBABLE'\n"
        "OVERRIDES = {\n"
        "    # 'Mike James': 'OUT',\n"
        "}\n"
        "LOCKED = []            # players that must be in the squad\n"
        "EXCLUDED = []          # players to leave out\n\n"
        "# Optional: your current 10 players + coach (names) to get the best trades instead\n"
        "ROSTER = None          # e.g. ['Sasha Vezenkov', ..., 'Georgios Bartzokas']\n"
        "BANK = 0.0             # unspent credits\n"
        "MAX_TRADES = 4         # 4 per round; raise it during an unlimited window"
    ),
    nbf.v4.new_code_cell(
        "rec = recommend(matchday=MATCHDAY, budget=BUDGET, overrides=OVERRIDES, locked=LOCKED,\n"
        "                excluded=EXCLUDED, roster=ROSTER, bank=BANK, max_trades=MAX_TRADES,\n"
        "                refresh=REFRESH)\n"
        "print(f'Round {rec.matchday} | deadline {rec.deadline:%a %d %b %H:%M} UTC | '\n"
        "      f'expected {rec.squad.objective:.1f} pts | cost {rec.squad.cost:.1f} / {rec.budget:.1f}')\n"
        "for n in rec.notes:\n"
        "    print('note:', n)"
    ),
    nbf.v4.new_markdown_cell("## Best 11\n\n`exp_pts` = P(plays) × points if he plays. Captain marked (C)."),
    nbf.v4.new_code_cell("display(rec.squad_table())\nrec.coach_pick()"),
    nbf.v4.new_markdown_cell("## Captain shortlist"),
    nbf.v4.new_code_cell("rec.captain_shortlist()"),
    nbf.v4.new_markdown_cell("## Rankings by position"),
    nbf.v4.new_code_cell(
        "for pos in ('G', 'F', 'C'):\n"
        "    display(Markdown(f'### {pos}'))\n"
        "    display(rec.rankings(pos, n=10))"
    ),
    nbf.v4.new_markdown_cell("## Value picks (expected points per credit)"),
    nbf.v4.new_code_cell("rec.value_picks(n=12)"),
    nbf.v4.new_markdown_cell("## Coaches"),
    nbf.v4.new_code_cell("rec.coach_table()"),
    nbf.v4.new_markdown_cell(
        "## Why these picks\n\nEach feature's contribution (fantasy points) to the prediction, "
        "relative to the average player."
    ),
    nbf.v4.new_code_cell(
        "for name in rec.squad_table()['name']:\n"
        "    display(Markdown(f'**{name}**'))\n"
        "    display(rec.why(name).head(5))"
    ),
    nbf.v4.new_markdown_cell("## Trades for your roster (if `ROSTER` is set)"),
    nbf.v4.new_code_cell(
        "if rec.trades:\n"
        "    print('Out:', ', '.join(rec.trades['out']) or 'none')\n"
        "    print('In: ', ', '.join(rec.trades['in']) or 'none')\n"
        "    print(f\"Expected gain: {rec.trades['gain']:+.1f} pts\")\n"
        "else:\n"
        "    print('No roster given.')"
    ),
    nbf.v4.new_markdown_cell("## Save"),
    nbf.v4.new_code_cell("path = rec.save()\nprint('saved to', path)"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = ROOT / "notebooks" / "weekly_picks.ipynb"
nbf.write(nb, out)
print("wrote", out)
