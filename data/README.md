# `data/` — datasets and the model's memory

The code finds these files through [`mundial_llm/paths.py`](../mundial_llm/paths.py), so they must stay in this folder.

## `historical_matches_with_xg.csv` (built locally, not in the repository)

StatsBomb's terms don't allow their data to be redistributed, so this file is listed in `.gitignore` and built on your own machine. It is created automatically the first time you run a prediction, or up front with:

```powershell
python -m mundial_llm.cache_historical_data
```

It takes about 1–2 minutes and needs internet. It holds 314 matches from six senior men's international tournaments in [StatsBomb Open Data](https://github.com/statsbomb/open-data):

| Tournament | Matches |
|------------|--------:|
| FIFA World Cup 2018 | 64 |
| UEFA Euro 2020 | 51 |
| FIFA World Cup 2022 | 64 |
| African Cup of Nations 2023 | 52 |
| UEFA Euro 2024 | 51 |
| Copa América 2024 | 32 |

| Column | Meaning |
|--------|---------|
| `match_id` | StatsBomb match ID |
| `match_date` | `YYYY-MM-DD` |
| `competition_name`, `season_name` | e.g. `FIFA World Cup`, `2022` |
| `home_team`, `away_team` | StatsBomb team names (tournament games are mostly at neutral venues, so "home" is nominal) |
| `home_score`, `away_score` | Goals after extra time; penalty shootouts are not included, so they show as draws |
| `home_team_xg`, `away_team_xg` | Sum of StatsBomb shot xG for each team, leaving out penalty-shootout kicks |

If any match fails to download after three tries, the builder stops without saving, so a partial file never stores missing data as zero xG. To rebuild, delete the file and run the command again.

## `learning_memory.json`

Lessons the model wrote after wrong predictions, added to future prompts for the same teams. A lesson saved by the current code looks like this:

```json
{
    "teams": ["Croatia", "Italy"],
    "match_date": "2024-06-24",
    "context": "Predicted a Croatia win, but the result was a draw",
    "lesson": "Overestimating the impact of home venue on goal scoring probability fails to account for ..."
}
```

Written by `auto_trainer.py`, and by `evaluate_model.py` only with `--save-lessons`. The match date lets the backtest use only lessons learned from matches played before the one it predicts; lessons with no date are left out of backtests.

The file in the repository holds seven lessons from my own runs: four from backtests and three from the 2026 World Cup. An earlier version of the code wrote them, before lessons recorded their date and before the backtest was limited to earlier matches, so they have no date: predictions use them, backtests don't. Replace the contents with `[]` to start from an empty memory.

## `trained_matches.json`

The ESPN match IDs the auto-trainer has already processed, so each finished 2026 match is learned from once. The repository ships the list from my own run. Delete the file and the next run records every finished match as a baseline instead of learning from them.

## Credits

<a href="https://github.com/statsbomb/open-data"><img src="https://raw.githubusercontent.com/statsbomb/open-data/master/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png" alt="StatsBomb" height="36"></a>

Match and event data © StatsBomb (Hudl), from StatsBomb Open Data, used under their public data user agreement.
