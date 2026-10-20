# `tests/` — automated tests

Unit tests for the logic that needs neither the LLM nor the internet, so they run in a couple of seconds and on GitHub Actions for every push ([workflow](../.github/workflows/tests.yml)).

```powershell
pytest
```

| File | Covers |
|------|--------|
| [`test_parsing.py`](test_parsing.py) | Reading the win/draw/win percentages from the model's answer (markdown bold, decimals, team names with special characters, one team name inside another, missing numbers), outcome helpers, fuzzy and alias team-name matching, and lesson extraction. |
| [`test_stats.py`](test_stats.py) | Team stats and recent form (most recent first, whatever the file order), the history filter that keeps the backtest from seeing the future, the xG baseline, and the Brier score. |
| [`test_stakes.py`](test_stakes.py) | The group-stage stakes simulator on hand-built standings (*already qualified*, *must win*, *a draw is enough*, team names that differ between ESPN feeds), and its fallbacks when the teams are in different groups or the match is no longer in the schedule. |
| [`test_memory.py`](test_memory.py) | Lessons are saved with their match date, empty lessons are skipped, and backtests only see lessons from earlier matches. |
| [`test_data.py`](test_data.py) | Penalty-shootout kicks are left out of xG, and a failed download raises instead of being stored as zero xG. |
| [`test_menu.py`](test_menu.py) | The menu explains that the tournament is over instead of reporting a network error, and reports a real network failure. |

[`conftest.py`](conftest.py) puts the repository root on the import path so the tests can import `mundial_llm`.
