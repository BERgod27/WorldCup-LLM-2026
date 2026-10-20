import pytest

from mundial_llm.learning_memory import describe_outcome, extract_lesson
from mundial_llm.predict_match import actual_outcome, fuzzy_match_team, parse_prediction, predicted_outcome

PREDICTION = """Spain control the ball but Morocco concede very little.

PREDICTION:
- Morocco win: 25%
- Draw: 30%
- Spain win: 45%
- Most likely scoreline: 0-1
- Key tactical factor: Morocco's low block against Spain's possession."""


def test_parses_the_prediction_block():
    assert parse_prediction(PREDICTION, "Morocco", "Spain") == {"Home Win": 25, "Draw": 30, "Away Win": 45}


def test_parses_markdown_bold_and_decimals():
    text = "**PREDICTION:**\n- **Morocco win:** 25.5%\n- **Draw:** 30%\n- **Spain win:** 44.5%"
    assert parse_prediction(text, "Morocco", "Spain") == {"Home Win": 25.5, "Draw": 30, "Away Win": 44.5}


def test_uses_the_last_block_when_numbers_appear_earlier():
    text = "Last time I said Morocco win: 90% but that was wrong.\n" + PREDICTION
    assert parse_prediction(text, "Morocco", "Spain")["Home Win"] == 25


def test_team_names_with_regex_characters():
    text = "- Côte d'Ivoire win: 40%\n- Draw: 35%\n- Bosnia-Herzegovina win: 25%"
    assert parse_prediction(text, "Côte d'Ivoire", "Bosnia-Herzegovina") == {"Home Win": 40, "Draw": 35, "Away Win": 25}


def test_returns_none_when_a_number_is_missing():
    assert parse_prediction("- Morocco win: 25%\n- Draw: 30%", "Morocco", "Spain") is None


def test_predicted_and_actual_outcomes():
    assert predicted_outcome({"Home Win": 25, "Draw": 30, "Away Win": 45}) == "Away Win"
    assert actual_outcome(2, 1) == "Home Win"
    assert actual_outcome(1, 1) == "Draw"
    assert actual_outcome(0, 3) == "Away Win"


def test_one_team_name_inside_another():
    # "Guinea win" must not be read from the "Equatorial Guinea win" line, in either order.
    text = "- Equatorial Guinea win: 40%\n- Draw: 30%\n- Guinea win: 30%"
    assert parse_prediction(text, "Equatorial Guinea", "Guinea") == {"Home Win": 40, "Draw": 30, "Away Win": 30}
    text = "- Guinea win: 30%\n- Draw: 30%\n- Equatorial Guinea win: 40%"
    assert parse_prediction(text, "Guinea", "Equatorial Guinea") == {"Home Win": 30, "Draw": 30, "Away Win": 40}


DATASET_TEAMS = ["Spain", "South Korea", "Portugal", "Morocco", "Cape Verde Islands", "United States", "Turkey",
                 "Côte d'Ivoire", "Colombia", "Iceland"]


@pytest.mark.parametrize("given, expected", [
    ("spain", "Spain"),
    ("South Korea", "South Korea"),
    ("Portugall", "Portugal"),
    ("portugall", "Portugal"),          # case-insensitive fuzzy match
    ("Cape Verde", "Cape Verde Islands"),  # ESPN vs StatsBomb spellings
    ("USA", "United States"),
    ("Türkiye", "Turkey"),
    ("Ivory Coast", "Côte d'Ivoire"),
    ("Columbia", "Colombia"),
    ("Ireland", "Ireland"),    # not in the dataset, and must not become Iceland
    ("Atlantis", "Atlantis"),  # unknown teams are passed through unchanged
])
def test_fuzzy_match_team(given, expected):
    assert fuzzy_match_team(given, DATASET_TEAMS) == expected


def test_describe_outcome_uses_team_names():
    assert describe_outcome("Home Win", "Croatia", "Italy") == "a Croatia win"
    assert describe_outcome("Away Win", "Croatia", "Italy") == "an Italy win"
    assert describe_outcome("Draw", "Croatia", "Italy") == "a draw"
    assert describe_outcome("Away Win", "Egypt", "Uruguay") == "a Uruguay win"
    assert describe_outcome("Home Win", "United States", "Iran") == "a United States win"


def test_extract_lesson():
    assert extract_lesson("**Lesson:** Respect   low blocks.\n") == "Respect low blocks."
    assert extract_lesson("Respect low blocks.") == "Respect low blocks."
    assert extract_lesson("   ") == ""
