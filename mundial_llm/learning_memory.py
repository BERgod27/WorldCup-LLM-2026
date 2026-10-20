import json
import os
import re

from .paths import LEARNING_MEMORY_FILE as MEMORY_FILE

# {predicted} and {actual} are filled with describe_outcome(), e.g. "a Croatia win" or "a draw".
# Team names rather than "Home Win" matter: tournament games are mostly at neutral venues,
# and "home"/"away" wording led the model to write lessons about a home advantage that isn't there.
REFLECTION_PROMPT = (
    "You predicted {predicted}, but the actual result was {actual}. "
    "Analyze why your logic failed based on the stats. "
    "Write exactly ONE sentence starting with 'Lesson: ' that acts as a tactical rule to prevent this mistake."
)

def load_memory():
    """Load the learning memory from the JSON file."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARNING] Failed to load learning memory: {e}")
        return []

def save_memory(memory):
    """Save the learning memory to the JSON file."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        print(f"  [ERROR] Failed to save learning memory: {e}")

def describe_outcome(outcome: str, home: str, away: str) -> str:
    """'Home Win' -> 'a Croatia win', 'Draw' -> 'a draw', 'Away Win' -> 'an Italy win'."""
    if outcome == "Draw":
        return "a draw"
    team = home if outcome == "Home Win" else away
    # "a United States win", "a Uruguay win": these names start with a "yoo" sound
    yoo_sound = team.lower().startswith(("uni", "uru", "ukr", "uga"))
    article = "an" if team[:1].upper() in "AEIOU" and not yoo_sound else "a"
    return f"{article} {team} win"

def extract_lesson(reflection: str) -> str:
    """Pull the one-sentence lesson out of the model's reflection, or '' if there is none."""
    clean = reflection.replace("**", "").replace("*", "").strip()
    match = re.search(r"Lesson:\s*(.*)", clean, re.IGNORECASE | re.DOTALL)
    lesson = match.group(1) if match else clean
    return " ".join(lesson.split())

def save_lesson(team_a: str, team_b: str, actual_outcome: str, predicted_outcome: str, lesson: str,
                match_date: str | None = None) -> bool:
    """
    Save a new lesson learned from a failed prediction. Empty lessons are skipped.

    match_date (YYYY-MM-DD) is the date of the match the lesson came from, so a backtest
    can use only lessons learned before the match it is predicting.
    """
    lesson = (lesson or "").strip()
    if not lesson:
        print(f"  [LEARNING] No usable lesson for {team_a} vs {team_b}; nothing saved")
        return False

    memory = load_memory()
    memory.append({
        "teams": [team_a, team_b],
        "match_date": match_date[:10] if match_date else None,
        "context": f"Predicted {describe_outcome(predicted_outcome, team_a, team_b)}, "
                   f"but the result was {describe_outcome(actual_outcome, team_a, team_b)}",
        "lesson": lesson
    })
    save_memory(memory)
    print(f"  [LEARNING] Saved new lesson for {team_a} vs {team_b}")
    return True

def get_lessons_for_match(team_a: str, team_b: str, before: str | None = None) -> str:
    """
    Retrieve any lessons that involve either team.

    With before (YYYY-MM-DD), only lessons from matches played before that date are
    returned, and lessons with no recorded match date are left out. Backtests use this
    so a lesson can never carry the result of the match being predicted, or a later one.
    """
    def usable(entry):
        if not entry.get("lesson"):
            return False
        if team_a not in entry.get("teams", []) and team_b not in entry.get("teams", []):
            return False
        if before is not None:
            return bool(entry.get("match_date")) and entry["match_date"] < before[:10]
        return True

    relevant_lessons = [entry["lesson"] for entry in load_memory() if usable(entry)]
    if not relevant_lessons:
        return ""

    lines = ["\n=== LESSONS LEARNED FROM PAST FAILURES ==="]
    for i, lesson in enumerate(relevant_lessons, 1):
        lines.append(f"{i}. {lesson}")
    return "\n".join(lines)
