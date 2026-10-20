import json

import pytest

from mundial_llm import learning_memory


@pytest.fixture
def memory_file(tmp_path, monkeypatch):
    path = tmp_path / "learning_memory.json"
    monkeypatch.setattr(learning_memory, "MEMORY_FILE", str(path))
    return path


def test_lessons_are_saved_with_their_match_date(memory_file):
    assert learning_memory.save_lesson("Croatia", "Italy", "Draw", "Home Win", "Respect low blocks.", match_date="2024-06-24T19:00Z")
    entry = json.loads(memory_file.read_text(encoding="utf-8"))[0]
    assert entry["match_date"] == "2024-06-24"
    assert entry["context"] == "Predicted a Croatia win, but the result was a draw"


def test_empty_lessons_are_not_saved(memory_file):
    assert not learning_memory.save_lesson("Croatia", "Italy", "Draw", "Home Win", "   ")
    assert not memory_file.exists()


def test_backtests_only_see_lessons_from_earlier_matches(memory_file):
    learning_memory.save_lesson("Croatia", "Italy", "Draw", "Home Win", "Lesson from June 2024.", match_date="2024-06-24")
    learning_memory.save_lesson("Italy", "Spain", "Away Win", "Draw", "Lesson from 2021.", match_date="2021-07-06")
    learning_memory.save_lesson("Italy", "Albania", "Home Win", "Draw", "Lesson with no date.")

    before_match = learning_memory.get_lessons_for_match("Italy", "Albania", before="2024-06-15")
    assert "Lesson from 2021." in before_match
    assert "Lesson from June 2024." not in before_match  # learned after the match being predicted
    assert "Lesson with no date." not in before_match    # can't prove it came earlier

    live = learning_memory.get_lessons_for_match("Italy", "Albania")
    assert all(text in live for text in ["Lesson from June 2024.", "Lesson from 2021.", "Lesson with no date."])
