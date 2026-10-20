import pandas as pd

from mundial_llm.evaluate_model import brier_score, history_before, xg_baseline
from mundial_llm.predict_match import get_team_stats

MATCHES = pd.DataFrame([
    # Deliberately out of date order, like the cache file.
    {"match_id": 3, "match_date": "2022-11-30", "competition_name": "FIFA World", "season_name": 2022,
     "home_team": "Spain", "away_team": "Japan", "home_score": 1, "away_score": 2, "home_team_xg": 1.5, "away_team_xg": 0.5},
    {"match_id": 1, "match_date": "2022-11-20", "competition_name": "FIFA World", "season_name": 2022,
     "home_team": "Spain", "away_team": "Costa Rica", "home_score": 7, "away_score": 0, "home_team_xg": 3.0, "away_team_xg": 0.0},
    {"match_id": 2, "match_date": "2022-11-25", "competition_name": "FIFA World", "season_name": 2022,
     "home_team": "Germany", "away_team": "Spain", "home_score": 1, "away_score": 1, "home_team_xg": 1.0, "away_team_xg": 1.0},
])


def test_form_is_most_recent_first_regardless_of_file_order():
    assert get_team_stats(MATCHES, "Spain")["form"] == "LDW"


def test_team_stats_totals():
    stats = get_team_stats(MATCHES, "Spain")
    assert (stats["wins"], stats["draws"], stats["losses"]) == (1, 1, 1)
    assert stats["goals_scored"] == 9
    assert stats["avg_xg_for"] == round((1.5 + 3.0 + 1.0) / 3, 2)


def test_unknown_team_has_no_stats():
    assert get_team_stats(MATCHES, "Atlantis") == {}


def test_history_before_excludes_the_match_day_and_later():
    history = history_before(MATCHES, "2022-11-25T19:00:00Z")
    assert list(history["match_id"]) == [1]


def test_xg_baseline_picks_the_better_xg_differential():
    assert xg_baseline(MATCHES, "Spain", "Costa Rica") == "Home Win"
    assert xg_baseline(MATCHES, "Costa Rica", "Spain") == "Away Win"
    assert xg_baseline(MATCHES, "Atlantis", "Narnia") == "Draw"


def test_brier_score():
    assert brier_score({"Home Win": 100, "Draw": 0, "Away Win": 0}, "Home Win") == 0
    assert brier_score({"Home Win": 0, "Draw": 0, "Away Win": 100}, "Home Win") == 2
    assert round(brier_score({"Home Win": 1, "Draw": 1, "Away Win": 1}, "Draw"), 3) == 0.667
