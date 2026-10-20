"""
Mundial LLM — Match Prediction Pipeline
========================================
Usage:
    python -m mundial_llm.predict_match "Portugal" "France"
    python -m mundial_llm.predict_match "Brazil" "Argentina" --model mundial-analyst
    python -m mundial_llm.predict_match --list-teams

The script:
  1. Fetches historical stats for both teams across all 6 international competitions.
  2. Analyzes match situations and standings (group stage stakes) via ESPN.
  3. Builds a structured tactical prompt with tournament stats, stakes, and risk biases.
  4. Streams the prediction analysis from the local Ollama model.
"""

import argparse
import re
import sys
import warnings
from difflib import get_close_matches
import pandas as pd
import ollama

warnings.filterwarnings("ignore")

# Import our custom modules
from .cache_historical_data import load_cached_matches
from .match_stakes import analyze_match_stakes, fetch_remaining_fixtures
from .learning_memory import get_lessons_for_match

# ---------------------------------------------------------------------------
# Stats aggregators
# ---------------------------------------------------------------------------

# ESPN and StatsBomb spell some national teams differently.
TEAM_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "cape verde": "Cape Verde Islands",
    "cabo verde": "Cape Verde Islands",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "czechia": "Czech Republic",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "dr congo": "Congo DR",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "ir iran": "Iran",
    "ireland": "Ireland",  # not in the dataset; without this it fuzzy-matches Iceland
}

def fuzzy_match_team(name: str, teams: list[str]) -> str:
    """Find the closest team name in the dataset (case-insensitive, with ESPN/StatsBomb aliases)."""
    by_lower = {t.lower(): t for t in teams}
    key = name.lower().strip()
    if key in by_lower:
        return by_lower[key]
    alias = TEAM_ALIASES.get(key)
    if alias:
        return by_lower.get(alias.lower(), alias)
    close = get_close_matches(key, list(by_lower), n=1, cutoff=0.85)
    if close:
        print(f"  [INFO] '{name}' matched as '{by_lower[close[0]]}' in historical database")
        return by_lower[close[0]]
    # If not found, return original and let the LLM use general knowledge
    return name

def get_team_stats(matches: pd.DataFrame, team: str) -> dict:
    """Compute overall and tournament-level stats for a team."""
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].sort_values("match_date")

    if team_matches.empty:
        return {}

    goals_for, goals_against, xg_for, xg_against = [], [], [], []
    results = []
    tournaments = {}

    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        gf = int(row["home_score"]) if is_home else int(row["away_score"])
        ga = int(row["away_score"]) if is_home else int(row["home_score"])
        goals_for.append(gf)
        goals_against.append(ga)

        xgf = row.get("home_team_xg" if is_home else "away_team_xg", 0.0)
        xga = row.get("away_team_xg" if is_home else "home_team_xg", 0.0)
        
        xg_for.append(float(xgf) if pd.notna(xgf) else 0.0)
        xg_against.append(float(xga) if pd.notna(xga) else 0.0)

        res = "W" if gf > ga else ("D" if gf == ga else "L")
        results.append(res)
        
        # Tournament breakdown
        comp = f"{row['competition_name']} {row['season_name']}"
        if comp not in tournaments:
            tournaments[comp] = {"gp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0}
        
        tournaments[comp]["gp"] += 1
        tournaments[comp]["gf"] += gf
        tournaments[comp]["ga"] += ga
        tournaments[comp]["xgf"] += float(xgf) if pd.notna(xgf) else 0.0
        tournaments[comp]["xga"] += float(xga) if pd.notna(xga) else 0.0
        if res == "W":
            tournaments[comp]["w"] += 1
        elif res == "D":
            tournaments[comp]["d"] += 1
        else:
            tournaments[comp]["l"] += 1

    n = len(team_matches)
    stats = {
        "team": team,
        "matches_played": n,
        "wins": results.count("W"),
        "draws": results.count("D"),
        "losses": results.count("L"),
        "goals_scored": sum(goals_for),
        "goals_conceded": sum(goals_against),
        "avg_goals_scored": round(sum(goals_for) / n, 2),
        "avg_goals_conceded": round(sum(goals_against) / n, 2),
        "avg_xg_for": round(sum(xg_for) / n, 2),
        "avg_xg_against": round(sum(xg_against) / n, 2),
        "form": "".join(reversed(results[-10:])), # Last 10 matches, most recent first
        "tournaments": tournaments
    }
    return stats

def get_match_log(matches: pd.DataFrame, team: str) -> str:
    """Build a readable match-by-match log for a team (last 10 games)."""
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].copy().sort_values("match_date", ascending=False).head(10) # 10 most recent
    
    # Reverse to chronological order
    team_matches = team_matches.iloc[::-1]
    
    lines = []
    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        opponent = row["away_team"] if is_home else row["home_team"]
        gf = int(row["home_score"]) if is_home else int(row["away_score"])
        ga = int(row["away_score"]) if is_home else int(row["home_score"])
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        xgf = row.get("home_team_xg" if is_home else "away_team_xg")
        xga = row.get("away_team_xg" if is_home else "home_team_xg")
        
        xg_str = f" | xG: {xgf:.2f} vs {xga:.2f}" if (pd.notna(xgf) and (xgf > 0 or xga > 0)) else ""
        comp = f"{row['competition_name']} {row['season_name']}"
        lines.append(
            f"  [{result}] {row['match_date'][:10]} ({comp:15s}) vs {opponent:<18} {gf}-{ga}{xg_str}"
        )
    return "\n".join(lines) if lines else "  No match data available."

def get_head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> str:
    """Check if these two teams met in the database and return results."""
    h2h = matches[
        ((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) |
        ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
    ].sort_values("match_date", ascending=False)
    
    if h2h.empty:
        return "No direct head-to-head games found in the historical database."
        
    lines = []
    for _, row in h2h.iterrows():
        xg_str = ""
        if pd.notna(row.get("home_team_xg")) and (row["home_team_xg"] > 0 or row["away_team_xg"] > 0):
            xg_str = f" (xG: {row['home_team_xg']:.2f} vs {row['away_team_xg']:.2f})"
        lines.append(
            f"  - {row['match_date'][:10]} ({row['competition_name']} {row['season_name']}): "
            f"{row['home_team']} {int(row['home_score'])}-{int(row['away_score'])} {row['away_team']}{xg_str}"
        )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite football tactical analyst specialising in international football and World Cups.
Your role is to provide structured, data-driven match previews with probabilistic outcomes.
You reason from statistics, head-to-head records, and the tactical situation/stakes of the match (e.g. if a team must win, they will take high risks; if they need a draw or are already qualified, they will play safe).
Never reason from sentiment, reputation, or guesswork.
Always end your analysis with a clear probability estimate in this exact format:

PREDICTION:
- {team_a} win: XX%
- Draw: XX%
- {team_b} win: XX%
- Most likely scoreline: X-X
- Key tactical factor: [one sentence]"""

USER_PROMPT = """Analyse the following upcoming FIFA World Cup 2026 match and provide a full tactical preview.

=== MATCH ===
{team_a} vs {team_b}
Stage: {stage}
Date: {date}
Venue: {venue}

=== GROUP STANDINGS ===
{group_standings}

=== MATCH SITUATION & STAKES ===
- {team_a}: {a_stakes}
  Tactical behavior: {a_bias}
- {team_b}: {b_stakes}
  Tactical behavior: {b_bias}

=== {team_a} — HISTORICAL STATS ===
Matches played: {a_matches}
Record (W-D-L): {a_wins}-{a_draws}-{a_losses}
Goals scored: {a_gf} (avg {a_avg_gf}/match)
Goals conceded: {a_ga} (avg {a_avg_ga}/match)
xG for (avg/match): {a_xgf}
xG against (avg/match): {a_xga}
Form (recent first): {a_form}
Tournament Breakdown:
{a_breakdown}
Recent matches log:
{a_log}

=== {team_b} — HISTORICAL STATS ===
Matches played: {b_matches}
Record (W-D-L): {b_wins}-{b_draws}-{b_losses}
Goals scored: {b_gf} (avg {b_avg_gf}/match)
Goals conceded: {b_ga} (avg {b_avg_ga}/match)
xG for (avg/match): {b_xgf}
xG against (avg/match): {b_xga}
Form (recent first): {b_form}
Tournament Breakdown:
{b_breakdown}
Recent matches log:
{b_log}

=== HEAD-TO-HEAD HISTORY ===
{h2h}
{lessons_learned}

=== YOUR TASK ===
Analyze the match internally based on the stats and stakes provided. 
IMPORTANT: Some of your past 'Lessons Learned' may be from highly unexpected outlier results. Do not overreact to them. Always prioritize core statistical strength (like xG) and do not let outlier lessons override massive disparities in team quality.
You do not need to write out a full tactical explanation, but you MUST ensure your analysis strictly focuses on {team_a} and {team_b}. Do not hallucinate or mention other teams.
Skip directly to the PREDICTION block, using the exact format specified."""

# ---------------------------------------------------------------------------
# Prompt builder helper
# ---------------------------------------------------------------------------

def format_tournament_breakdown(tournaments: dict) -> str:
    """Format the tournament-by-tournament stats breakdown."""
    if not tournaments:
        return "  No tournament-specific data available."
    lines = []
    for comp, stats in sorted(tournaments.items()):
        avg_gf = round(stats["gf"] / stats["gp"], 2)
        avg_ga = round(stats["ga"] / stats["gp"], 2)
        avg_xgf = round(stats["xgf"] / stats["gp"], 2)
        avg_xga = round(stats["xga"] / stats["gp"], 2)
        lines.append(
            f"  * {comp:22s}: {stats['gp']} GP | {stats['w']}W-{stats['d']}D-{stats['l']}L | "
            f"Goals: {stats['gf']}-{stats['ga']} (avg {avg_gf}/{avg_ga}) | "
            f"xG: {avg_xgf:.2f}/{avg_xga:.2f}"
        )
    return "\n".join(lines)

def build_prompt_vars(team_a: str, team_b: str, matches: pd.DataFrame, stakes_info: dict,
                      use_lessons: bool = True, lessons_before: str | None = None) -> dict:
    """
    Assemble all template variables from StatsBomb data and stakes info.

    lessons_before (YYYY-MM-DD) limits lessons to ones learned from earlier matches (used by backtests).
    """
    stats_a = get_team_stats(matches, team_a)
    stats_b = get_team_stats(matches, team_b)

    def s(stats: dict, key: str, default: str = "N/A"):
        val = stats.get(key, default)
        return str(val) if val != default else default

    return {
        "team_a": team_a,
        "team_b": team_b,
        "stage": stakes_info.get("stage", "Group Stage"),
        "date": stakes_info.get("date", "TBD"),
        "venue": stakes_info.get("venue", "TBD"),
        "group_standings": stakes_info.get("group_standings_text", "N/A"),
        "a_stakes": stakes_info.get("home_stakes", "Standard stakes."),
        "a_bias": stakes_info.get("home_bias", "Balanced approach."),
        "b_stakes": stakes_info.get("away_stakes", "Standard stakes."),
        "b_bias": stakes_info.get("away_bias", "Balanced approach."),
        
        "a_matches": s(stats_a, "matches_played"),
        "a_wins": s(stats_a, "wins"),
        "a_draws": s(stats_a, "draws"),
        "a_losses": s(stats_a, "losses"),
        "a_gf": s(stats_a, "goals_scored"),
        "a_ga": s(stats_a, "goals_conceded"),
        "a_avg_gf": s(stats_a, "avg_goals_scored"),
        "a_avg_ga": s(stats_a, "avg_goals_conceded"),
        "a_xgf": s(stats_a, "avg_xg_for"),
        "a_xga": s(stats_a, "avg_xg_against"),
        "a_form": s(stats_a, "form"),
        "a_breakdown": format_tournament_breakdown(stats_a.get("tournaments", {})),
        "a_log": get_match_log(matches, team_a),
        
        "b_matches": s(stats_b, "matches_played"),
        "b_wins": s(stats_b, "wins"),
        "b_draws": s(stats_b, "draws"),
        "b_losses": s(stats_b, "losses"),
        "b_gf": s(stats_b, "goals_scored"),
        "b_ga": s(stats_b, "goals_conceded"),
        "b_avg_gf": s(stats_b, "avg_goals_scored"),
        "b_avg_ga": s(stats_b, "avg_goals_conceded"),
        "b_xgf": s(stats_b, "avg_xg_for"),
        "b_xga": s(stats_b, "avg_xg_against"),
        "b_form": s(stats_b, "form"),
        "b_breakdown": format_tournament_breakdown(stats_b.get("tournaments", {})),
        "b_log": get_match_log(matches, team_b),
        
        "h2h": get_head_to_head(matches, team_a, team_b),
        "lessons_learned": get_lessons_for_match(team_a, team_b, before=lessons_before) if use_lessons else "",
    }

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

OUTCOMES = ("Home Win", "Draw", "Away Win")

def parse_prediction(content: str, team_a: str, team_b: str) -> dict | None:
    """
    Extract the win/draw/win percentages from the model's PREDICTION block.

    Returns {"Home Win": float, "Draw": float, "Away Win": float}, where team_a is
    the home side, or None when the three numbers cannot be found.
    """
    text = content.replace("**", "").replace("*", "")
    # The lookbehinds stop "Guinea win" from matching inside "Equatorial Guinea win".
    not_inside_name = r"(?<![\w'’-])(?<![\w'’] )"
    patterns = {
        "Home Win": rf"{not_inside_name}{re.escape(team_a)}\s+win\s*:\s*(\d+(?:\.\d+)?)\s*%",
        "Draw": r"\bDraw\s*:\s*(\d+(?:\.\d+)?)\s*%",
        "Away Win": rf"{not_inside_name}{re.escape(team_b)}\s+win\s*:\s*(\d+(?:\.\d+)?)\s*%",
    }
    probs = {}
    for outcome, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return None
        probs[outcome] = float(matches[-1])  # the PREDICTION block comes last
    return probs

def predicted_outcome(probs: dict) -> str:
    """The outcome with the highest probability."""
    return max(OUTCOMES, key=lambda o: probs[o])

def actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "Home Win"
    if away_score > home_score:
        return "Away Win"
    return "Draw"

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Used when there is no upcoming fixture to read standings for: a pairing that is not
# on the 2026 schedule, or a match being replayed after it was played (so the live
# standings would already contain its result).
NEUTRAL_STAKES = {
    "stage": "Tournament Match",
    "date": "TBD",
    "venue": "TBD",
    "group_standings_text": "N/A (no upcoming fixture for this pairing)",
    "home_stakes": "Standard tournament stakes.",
    "home_bias": "Balanced approach.",
    "away_stakes": "Standard tournament stakes.",
    "away_bias": "Balanced approach.",
}

def predict_match(team_a: str, team_b: str, model_name: str = "mundial-analyst", fixture: dict = None):
    """Fetch data, analyze stakes, build prompt, and run the LLM."""
    print(f"\n{'='*65}")
    print(f"  MUNDIAL LLM - Match Prediction & Stakes Reasoning")
    print(f"  {team_a.upper()} vs {team_b.upper()}")
    print(f"  Model: {model_name}")
    print(f"{'='*65}\n")

    # 1. Load historical database cache
    print("[1/3] Loading historical international match database cache...")
    matches = load_cached_matches()
    if matches.empty:
        print("[ERROR] Could not load historical data. Make sure cache script is compiled.")
        sys.exit(1)
        
    all_teams = list(set(matches["home_team"].tolist() + matches["away_team"].tolist()))

    # Match team names to DB names
    db_home = fuzzy_match_team(team_a, all_teams)
    db_away = fuzzy_match_team(team_b, all_teams)

    # 2. Get group stage stakes and upcoming match metadata
    print("[2/3] Fetching standings and analyzing tactical stakes...")
    
    stage = None  # stays None when the pairing is not an upcoming 2026 fixture
    date = "TBD"
    venue = "TBD"
    
    if fixture:
        stage = fixture.get("stage", "Group Stage")
        date = fixture.get("date", "TBD")
        venue = fixture.get("venue", "TBD")
    else:
        # Check if this match is in the upcoming ESPN schedule
        fixtures = fetch_remaining_fixtures()
        # Try to find match metadata in upcoming fixtures
        home_clean = team_a.lower().strip()
        away_clean = team_b.lower().strip()
        for fix in fixtures:
            state = fix.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("state", "")
            if state not in ("pre", "in"):
                continue  # finished matches: live standings would already include the result
            f_home = fix.get("competitions", [{}])[0].get("competitors", [{}])[0].get("team", {}).get("displayName", "").lower().strip()
            f_away = fix.get("competitions", [{}])[0].get("competitors", [{}])[1].get("team", {}).get("displayName", "").lower().strip()
            if (home_clean in f_home and away_clean in f_away) or (away_clean in f_home and home_clean in f_away):
                # Found metadata in ESPN!
                stage = fix.get("season", {}).get("slug", "group-stage")
                
                raw_date = fix.get("date", "")
                try:
                    from datetime import datetime
                    match_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    date = match_dt.strftime("%a %d %b  %H:%M UTC")
                except Exception:
                    date = raw_date
                    
                comp = fix.get("competitions", [{}])[0]
                venue_obj = comp.get("venue", {})
                venue_name = venue_obj.get("fullName", "")
                city = venue_obj.get("address", {}).get("city", "")
                venue = f"{venue_name}, {city}" if city else venue_name
                break

    # Analyze stakes
    if stage is None:
        print("  [INFO] No upcoming 2026 fixture found for this pairing, so neutral stakes are used.")
        stakes_info = dict(NEUTRAL_STAKES)
    else:
        stakes_info = analyze_match_stakes(team_a, team_b, stage)
    stakes_info["date"] = date
    stakes_info["venue"] = venue

    # 3. Build prompt variables
    print(f"[3/3] Querying {model_name}...")
    vars_ = build_prompt_vars(db_home, db_away, matches, stakes_info)
    
    # Build prompts
    filled_user_prompt = USER_PROMPT.format(**vars_)
    filled_system_prompt = SYSTEM_PROMPT.format(team_a=db_home, team_b=db_away)
    
    messages = [
        {"role": "system", "content": filled_system_prompt},
        {"role": "user",   "content": filled_user_prompt},
    ]

    print("\nStreaming analysis...\n")
    print("-" * 65)

    try:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            think=False,       # disable qwen3 extended reasoning mode
            options={"temperature": 0.25},
            stream=True,
        )
        for chunk in stream:
            content = chunk.message.content
            if content:
                print(content, end="", flush=True)
        print(f"\n{'-'*65}\n")
    except Exception as e:
        print(f"\n[ERROR] Could not get response from Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Mundial LLM - Match Prediction & Stakes Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("team_a", nargs="?", help="First team name")
    parser.add_argument("team_b", nargs="?", help="Second team name")
    parser.add_argument(
        "--model", type=str, default="mundial-analyst",
        help="Ollama model name (default: mundial-analyst)"
    )
    parser.add_argument(
        "--list-teams", action="store_true",
        help="List all teams available in the database"
    )
    args = parser.parse_args()

    if args.list_teams:
        print("\nTeams available in historical database:")
        matches = load_cached_matches()
        if not matches.empty:
            teams = sorted(set(matches["home_team"].tolist() + matches["away_team"].tolist()))
            for t in teams:
                print(f"  - {t}")
        return

    if not args.team_a or not args.team_b:
        parser.print_help()
        sys.exit(1)

    predict_match(args.team_a, args.team_b, model_name=args.model)

if __name__ == "__main__":
    main()
