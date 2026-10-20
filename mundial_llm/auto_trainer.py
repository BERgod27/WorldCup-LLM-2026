"""
Mundial LLM - Auto Trainer
==========================
Checks ESPN for newly finished 2026 World Cup matches, predicts each one from
the historical stats alone, and when the prediction is wrong asks the model to
reflect and stores a one-sentence lesson in learning_memory.json. Those lessons
are added to future prompts involving the same teams.

The replayed prediction is blind to the result: it uses neutral stakes instead
of the live standings (which already include the match), and only lessons
learned from earlier matches.

Finished matches are tracked in trained_matches.json so each one is used once.
When that file does not exist, the first run only records the matches already
finished, as a baseline, and later runs learn from the new ones.

Usage:
    python -m mundial_llm.auto_trainer
    python -m mundial_llm.auto_trainer --model qwen3.5:9b --verbose
"""

import argparse
import json
import os
import requests
import ollama

from .cache_historical_data import load_cached_matches
from .predict_match import (
    build_prompt_vars, SYSTEM_PROMPT, USER_PROMPT, NEUTRAL_STAKES, fuzzy_match_team,
    parse_prediction, predicted_outcome, actual_outcome,
)
from .learning_memory import REFLECTION_PROMPT, describe_outcome, extract_lesson, save_lesson
from .paths import TRAINED_MATCHES_FILE as TRAINED_FILE

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    "?limit=200&dates=20260611-20260720"
)

def load_trained():
    if not os.path.exists(TRAINED_FILE):
        return []
    with open(TRAINED_FILE, "r") as f:
        return json.load(f)

def save_trained(trained):
    with open(TRAINED_FILE, "w") as f:
        json.dump(trained, f, indent=4)

def run_auto_trainer(model_name: str = "mundial-analyst", verbose: bool = False):
    print("Fetching ESPN Scoreboard...")
    try:
        r = requests.get(ESPN_SCOREBOARD, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] Could not reach the ESPN API: {e}")
        return
    
    trained_matches = load_trained()
    matches_df = load_cached_matches()
    if matches_df.empty:
        print("[ERROR] Could not load the historical data cache.")
        return
    
    # First run: record what has already been played, without training on it
    if not trained_matches and not os.path.exists(TRAINED_FILE):
        print("[INFO] Initializing trained matches database. No training will occur on this first run to establish a baseline.")
        for event in data.get("events", []):
            status = event.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("state", "")
            if status == "post":
                match_id = event.get("id")
                trained_matches.append(match_id)
        save_trained(trained_matches)
        print(f"[INFO] Initialized with {len(trained_matches)} past matches.")
        return

    new_matches = 0
    for event in data.get("events", []):
        status = event.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("state", "")
        match_id = event.get("id")
        
        if status == "post" and match_id not in trained_matches:
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) == 2:
                home = competitors[0].get("team", {}).get("displayName", "")
                away = competitors[1].get("team", {}).get("displayName", "")
                home_score = int(competitors[0].get("score", "0"))
                away_score = int(competitors[1].get("score", "0"))
                match_date = event.get("date", "")[:10]
                
                print(f"\n[NEW MATCH DETECTED] {home} {home_score} - {away_score} {away}")
                try:
                    success = evaluate_single_match(home, away, home_score, away_score, match_date,
                                                    matches_df, model_name, verbose)
                except Exception as e:
                    print(f"[ERROR] Could not get a response from Ollama: {e}")
                    print("Make sure Ollama is running: ollama serve")
                    return
                
                if success:
                    trained_matches.append(match_id)
                    save_trained(trained_matches)
                    new_matches += 1

    if new_matches == 0:
        print("No new matches found.")
    else:
        print(f"Finished training on {new_matches} new matches.")

def evaluate_single_match(home, away, home_score, away_score, match_date, matches_df,
                          model_name="mundial-analyst", verbose=False):
    actual = actual_outcome(home_score, away_score)

    all_teams = sorted(set(matches_df['home_team']) | set(matches_df['away_team']))
    db_home = fuzzy_match_team(home, all_teams)
    db_away = fuzzy_match_team(away, all_teams)
    
    # Neutral stakes and only earlier lessons, so nothing in the prompt reveals the result
    stakes_info = dict(NEUTRAL_STAKES, date=match_date or "TBD")
    vars_ = build_prompt_vars(db_home, db_away, matches_df, stakes_info, lessons_before=match_date or None)
    
    filled_user_prompt = USER_PROMPT.format(**vars_)
    filled_system_prompt = SYSTEM_PROMPT.format(team_a=db_home, team_b=db_away)
    
    messages = [
        {"role": "system", "content": filled_system_prompt},
        {"role": "user", "content": filled_user_prompt}
    ]
    
    print("  Predicting outcome...")
    content = ""
    for attempt in range(3):
        resp = ollama.chat(
            model=model_name,
            messages=messages,
            think=False,
            options={"temperature": 0.25}
        )
        content = resp.message.content.strip()
        if content:
            break
        print("  [WARNING] LLM returned empty response. Retrying...")
    
    if verbose:
        print(f"  LLM output:\n{content}\n")
    
    probs = parse_prediction(content, db_home, db_away)
    if probs is None:
        print("  [ERROR] Failed to parse prediction.")
        return False
        
    predicted = predicted_outcome(probs)
    print(f"  Predicted: {describe_outcome(predicted, db_home, db_away)} | Actual: {describe_outcome(actual, db_home, db_away)}")
    
    if predicted != actual:
        print("  Generating lesson for failure...")
        messages.append(resp.message)
        messages.append({"role": "user", "content": REFLECTION_PROMPT.format(
            predicted=describe_outcome(predicted, db_home, db_away),
            actual=describe_outcome(actual, db_home, db_away),
        )})
        
        try:
            reflection_resp = ollama.chat(
                model=model_name,
                messages=messages,
                think=False,
                options={"temperature": 0.4}
            )
            save_lesson(db_home, db_away, actual, predicted, extract_lesson(reflection_resp.message.content),
                        match_date=match_date or None)
        except Exception as ex:
            print(f"  [ERROR] Failed to generate lesson: {ex}")
            
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn lessons from finished 2026 World Cup matches.")
    parser.add_argument("--model", type=str, default="mundial-analyst", help="Ollama model to use.")
    parser.add_argument("--verbose", action="store_true", help="Print the model's full answer for each match.")
    args = parser.parse_args()
    run_auto_trainer(args.model, args.verbose)
