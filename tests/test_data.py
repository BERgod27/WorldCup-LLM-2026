import pandas as pd
import pytest

from mundial_llm import cache_historical_data
from mundial_llm.cache_historical_data import match_xg


def test_xg_leaves_out_penalty_shootout_kicks():
    events = pd.DataFrame([
        {"type": "Shot", "team": "Argentina", "period": 1, "shot_statsbomb_xg": 0.79},
        {"type": "Shot", "team": "France", "period": 4, "shot_statsbomb_xg": 0.30},
        {"type": "Pass", "team": "France", "period": 2, "shot_statsbomb_xg": None},
        {"type": "Shot", "team": "Argentina", "period": 5, "shot_statsbomb_xg": 0.78},  # shootout kick
        {"type": "Shot", "team": "France", "period": 5, "shot_statsbomb_xg": 0.78},     # shootout kick
    ])
    assert match_xg(events, "Argentina", "France") == (0.79, 0.30)


def test_a_failed_download_raises_instead_of_returning_zero_xg(monkeypatch):
    def offline(match_id):
        raise ConnectionError("offline")
    monkeypatch.setattr(cache_historical_data.statsbomb, "events", offline)
    with pytest.raises(ConnectionError):
        cache_historical_data.fetch_xg_for_match(1, "Argentina", "France", attempts=1)
