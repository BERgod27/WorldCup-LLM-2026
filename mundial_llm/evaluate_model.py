"""
Mundial LLM - Backtesting Evaluation
====================================
Replays past matches through the prediction prompt and scores the model.

Nothing in the prompt comes from the future: for each match, the stats are
built only from matches played BEFORE that match's date, the date itself is
not shown, and with --with-lessons only lessons learned from earlier matches
are included. Backtests never write new lessons unless --save-lessons is passed.

One thing this cannot rule out: the base model may remember famous results from
its pretraining (the 2024 tournaments happened before qwen3.5 was released).
--live-tournament on the 2026 World Cup is the cleaner out-of-sample test.

Every run also scores a simple baseline on the same matches: the team with the
better average xG differential (xG for minus xG against) wins, equal means draw.

Usage:
    python -m mundial_llm.evaluate_model --from-date 2024-06-01 --sample 0          # every Euro 2024 + Copa America 2024 match
    python -m mundial_llm.evaluate_model --to-date 2024-06-01 --sample 0 --save-lessons   # learn lessons on 2018-2023 only...
    python -m mundial_llm.evaluate_model --from-date 2024-06-01 --sample 0 --with-lessons # ...then test with them
    python -m mundial_llm.evaluate_model --sample 50 --seed 42                      # random sample of all tournaments
    python -m mundial_llm.evaluate_model --live-tournament
"""

import argparse
import sys

import ollama
import pandas as pd
import requests

from .cache_historical_data import load_cached_matches
from .learning_memory import REFLECTION_PROMPT, describe_outcome, extract_lesson, save_lesson
from .predict_match import (
    OUTCOMES, SYSTEM_PROMPT, USER_PROMPT, actual_outcome, build_prompt_vars,
    fuzzy_match_team, get_team_stats, parse_prediction, predicted_outcome,
)

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    "?limit=200&dates=20260611-20260720"
)

MOCK_STAKES = {
    "stage": "Tournament Match",
    "venue": "Historical Venue",
    "group_standings_text": "N/A (Historical Match)",
    "home_stakes": "Standard tournament stakes.",
    "home_bias": "Balanced approach.",
    "away_stakes": "Standard tournament stakes.",
    "away_bias": "Balanced approach.",
}


def history_before(matches: pd.DataFrame, match_date: str) -> pd.DataFrame:
    """Only the matches played strictly before match_date, so no future results leak in."""
    return matches[matches["match_date"] < match_date[:10]]


def xg_baseline(history: pd.DataFrame, home: str, away: str) -> str:
    """Predict the side with the better average xG differential; equal (or no data) means draw."""
    def diff(team):
        stats = get_team_stats(history, team)
        return stats["avg_xg_for"] - stats["avg_xg_against"] if stats else 0.0
    home_diff, away_diff = diff(home), diff(away)
    if home_diff > away_diff:
        return "Home Win"
    if away_diff > home_diff:
        return "Away Win"
    return "Draw"


def brier_score(probs: dict, actual: str) -> float:
    """Multi-class Brier score (0 = perfect, 2 = worst, 0.667 = always 33/33/33)."""
    total = sum(probs.values()) or 1.0
    return sum((probs[o] / total - (1.0 if o == actual else 0.0)) ** 2 for o in OUTCOMES)


def fetch_completed_tournament_matches() -> pd.DataFrame:
    """Completed 2026 World Cup matches from ESPN, in the same columns as the history cache."""
    print("  Fetching completed 2026 World Cup matches from ESPN...")
    r = requests.get(ESPN_SCOREBOARD, timeout=10)
    r.raise_for_status()

    completed = []
    for event in r.json().get("events", []):
        comp = event.get("competitions", [{}])[0]
        if comp.get("status", {}).get("type", {}).get("state", "") != "post":
            continue
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue
        completed.append({
            "match_id": event.get("id"),
            "match_date": event.get("date", "2026-06-11")[:10],
            "home_team": competitors[0].get("team", {}).get("displayName", ""),
            "away_team": competitors[1].get("team", {}).get("displayName", ""),
            "home_score": int(competitors[0].get("score", "0")),
            "away_score": int(competitors[1].get("score", "0")),
        })
    return pd.DataFrame(completed)


def evaluate_matches(num_samples: int, model_name: str = "mundial-analyst", live_tournament: bool = False,
                     seed: int | None = None, with_lessons: bool = False, save_lessons: bool = False,
                     from_date: str | None = None, to_date: str | None = None):
    print(f"\n============================================================")
    print(f"  MUNDIAL LLM - AUTOMATED BACKTESTING EVALUATION")
    print(f"  Model: {model_name}")
    if live_tournament:
        mode = "live 2026 tournament"
    else:
        mode = f"historical, {'all matches' if num_samples <= 0 else f'sample of {num_samples} (seed {seed})'}"
        mode += f" from {from_date}" if from_date else ""
        mode += f" before {to_date}" if to_date else ""
    print(f"  Mode: {mode}")
    print(f"  Lessons in prompt: {'yes' if with_lessons else 'no'} | Saving new lessons: {'yes' if save_lessons else 'no'}")
    print(f"============================================================\n")

    matches_df = load_cached_matches()
    if matches_df.empty:
        print("[ERROR] Could not load historical data cache.")
        sys.exit(1)
    valid_matches = matches_df.dropna(subset=["home_score", "away_score"])
    all_teams = sorted(set(valid_matches["home_team"]) | set(valid_matches["away_team"]))

    if live_tournament:
        try:
            samples = fetch_completed_tournament_matches()
        except Exception as e:
            print(f"  [ERROR] Failed to fetch live matches: {e}")
            sys.exit(1)
        if samples.empty:
            print("  [INFO] No completed matches found for the 2026 World Cup yet.")
            sys.exit(0)
        print(f"  Found {len(samples)} completed live matches to evaluate!\n")
    else:
        candidates = valid_matches
        if from_date:
            candidates = candidates[candidates["match_date"] >= from_date]
        if to_date:
            candidates = candidates[candidates["match_date"] < to_date]
        if 0 < num_samples < len(candidates):
            samples = candidates.sample(n=num_samples, random_state=seed)
        else:
            samples = candidates.sort_values("match_date")

    llm_correct = baseline_correct = evaluated = unparsed = 0
    brier_total = 0.0

    print(f"{'MATCH':<35} | {'ACTUAL':<8} | {'LLM':<8} | {'xG BASE':<8} | {'LLM OK?':<7}")
    print("-" * 80)

    for _, row in samples.iterrows():
        home_team = fuzzy_match_team(row["home_team"], all_teams)
        away_team = fuzzy_match_team(row["away_team"], all_teams)
        actual = actual_outcome(int(row["home_score"]), int(row["away_score"]))

        history = history_before(valid_matches, str(row["match_date"]))
        # The real date is left out, so the prompt gives the model less to recognise the match by.
        stakes = dict(MOCK_STAKES, date="TBD")
        vars_ = build_prompt_vars(home_team, away_team, history, stakes,
                                  use_lessons=with_lessons, lessons_before=str(row["match_date"]))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(team_a=home_team, team_b=away_team)},
            {"role": "user", "content": USER_PROMPT.format(**vars_)},
        ]

        try:
            response = ollama.chat(model=model_name, messages=messages, think=False, options={"temperature": 0.2})
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {home_team} vs {away_team}: {e}")
            continue

        probs = parse_prediction(response.message.content, home_team, away_team)
        if probs is None:
            # Counting an unparsed answer as a prediction would silently bias the score.
            unparsed += 1
            print(f"{home_team + ' vs ' + away_team:<35} | could not parse the model's percentages, skipped")
            continue

        predicted = predicted_outcome(probs)
        baseline = xg_baseline(history, home_team, away_team)
        evaluated += 1
        llm_correct += predicted == actual
        baseline_correct += baseline == actual
        brier_total += brier_score(probs, actual)

        match_str = f"{home_team} vs {away_team}"
        print(f"{match_str:<35} | {actual:<8} | {predicted:<8} | {baseline:<8} | {'YES' if predicted == actual else 'NO':<7}")

        if predicted != actual and save_lessons:
            messages.append(response.message)
            messages.append({"role": "user", "content": REFLECTION_PROMPT.format(
                predicted=describe_outcome(predicted, home_team, away_team),
                actual=describe_outcome(actual, home_team, away_team),
            )})
            try:
                reflection = ollama.chat(model=model_name, messages=messages, think=False, options={"temperature": 0.4})
                save_lesson(home_team, away_team, actual, predicted, extract_lesson(reflection.message.content),
                            match_date=str(row["match_date"]))
            except Exception as ex:
                print(f"  [ERROR] Failed to generate lesson: {ex}")

    print("-" * 80)
    if evaluated == 0:
        print("  No matches could be scored.")
        return
    print(f"  Matches scored:        {evaluated} (unparsed answers skipped: {unparsed})")
    print(f"  LLM accuracy:          {llm_correct}/{evaluated} ({100 * llm_correct / evaluated:.1f}%)")
    print(f"  xG baseline accuracy:  {baseline_correct}/{evaluated} ({100 * baseline_correct / evaluated:.1f}%)")
    print(f"  LLM Brier score:       {brier_total / evaluated:.3f} (lower is better; 0.667 = always 33/33/33)")
    print("============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mundial LLM against historical data.")
    parser.add_argument("--sample", type=int, default=5, help="Number of random historical matches to evaluate (0 = all).")
    parser.add_argument("--from-date", type=str, default=None, help="Only evaluate historical matches on or after this date (YYYY-MM-DD).")
    parser.add_argument("--to-date", type=str, default=None, help="Only evaluate historical matches before this date (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, so a sample can be reproduced.")
    parser.add_argument("--model", type=str, default="mundial-analyst", help="Ollama model to use.")
    parser.add_argument("--live-tournament", action="store_true", help="Evaluate on completed 2026 World Cup matches instead.")
    parser.add_argument("--with-lessons", action="store_true", help="Include saved lessons from earlier matches in the prompt.")
    parser.add_argument("--save-lessons", action="store_true", help="Write a lesson to learning_memory.json for each miss.")
    args = parser.parse_args()

    evaluate_matches(args.sample, args.model, args.live_tournament, args.seed, args.with_lessons, args.save_lessons,
                     args.from_date, args.to_date)
