"""
Mundial LLM — Interactive Match Selector & Prediction Menu
===========================================================
Fetches all remaining 2026 FIFA World Cup fixtures from the ESPN API,
lets the user pick one from a numbered menu, then runs the LLM prediction.

Usage:
    python -m mundial_llm
    python -m mundial_llm --model qwen3.5:9b
"""

import argparse
import sys
import warnings
import io
from datetime import datetime

# Force UTF-8 output on Windows terminals (fixes Turkiye, Curacao etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    "?limit=200&dates=20260611-20260720"
)

# ---------------------------------------------------------------------------
# ESPN schedule fetcher
# ---------------------------------------------------------------------------

def fetch_remaining_fixtures() -> list[dict] | None:
    """
    Fetch all scheduled (not yet played) 2026 WC fixtures from the ESPN API.
    Returns a list of fixture dicts with keys: date, home, away, venue, stage,
    an empty list when every game has been played, or None if ESPN can't be reached.
    """
    try:
        r = requests.get(ESPN_SCOREBOARD, timeout=10)
        r.raise_for_status()
        events = r.json().get("events", [])
    except (requests.RequestException, ValueError) as e:
        print(f"[ERROR] Cannot reach ESPN API: {e}")
        return None

    fixtures = []

    for event in events:
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("state", "")

        # Only include games not yet started
        if status not in ("pre", "in"):
            continue

        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue

        # ESPN orders: index 0 = home, index 1 = away
        home = teams[0].get("team", {}).get("displayName", "TBD")
        away = teams[1].get("team", {}).get("displayName", "TBD")

        # Skip placeholder knockout games (e.g. "Group A Winner vs ...")
        if "Winner" in home or "Place" in home or "Winner" in away or "Place" in away:
            continue

        raw_date = event.get("date", "")
        try:
            match_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            date_str = match_dt.strftime("%a %d %b  %H:%M UTC")
        except ValueError:
            date_str = raw_date[:10]

        venue_obj = comp.get("venue", {})
        venue = venue_obj.get("fullName", "")
        city = venue_obj.get("address", {}).get("city", "")
        venue_str = f"{venue}, {city}" if city else venue

        stage = event.get("season", {}).get("slug", "").replace("-", " ").title()

        fixtures.append({
            "date": date_str,
            "home": home,
            "away": away,
            "venue": venue_str,
            "stage": stage,
        })

    return fixtures


# ---------------------------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------------------------
from .predict_match import predict_match


# ---------------------------------------------------------------------------
# Menu display
# ---------------------------------------------------------------------------

def print_menu(fixtures: list[dict], page: int, page_size: int = 15):
    """Print a paginated menu of upcoming fixtures."""
    total = len(fixtures)
    total_pages = (total + page_size - 1) // page_size
    start = page * page_size
    end = min(start + page_size, total)
    page_fixtures = fixtures[start:end]

    print(f"\n{'='*62}")
    print(f"  MUNDIAL LLM 2026  --  Upcoming Fixtures")
    print(f"  Page {page + 1}/{total_pages}  |  {total} games remaining")
    print(f"{'='*62}")

    # Group by date for readability
    current_date = None
    for i, fix in enumerate(page_fixtures, start=start + 1):
        date_part = fix["date"].split("  ")[0].strip()  # e.g. "Tue 17 Jun"
        if date_part != current_date:
            print(f"  {date_part}")
            print(f"  {'-'*56}")
            current_date = date_part
        time_part = fix["date"].split("  ")[-1] if "  " in fix["date"] else ""
        print(f"  [{i:>2}] {fix['home']:<22} vs  {fix['away']:<22}  {time_part}")

    print(f"\n{'-'*62}")
    nav = []
    if page > 0:
        nav.append("[P] Previous page")
    if end < total:
        nav.append("[N] Next page")
    nav.append("[Q] Quit")
    print("  " + "   ".join(nav))
    print(f"{'-'*62}")


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mundial LLM - pick an upcoming 2026 World Cup fixture and predict it.")
    parser.add_argument("--model", type=str, default="mundial-analyst", help="Ollama model to use.")
    model = parser.parse_args().model

    # Pre-load data
    print("\n  Loading 2026 WC schedule from ESPN API...")
    fixtures = fetch_remaining_fixtures()
    if fixtures is None:
        print("[ERROR] Could not load the schedule. Check your internet connection.")
        sys.exit(1)
    if not fixtures:
        print("\n  The 2026 World Cup is over, so there are no upcoming fixtures to pick from.")
        print("  You can still predict any pairing, or score the model on the finished tournament:")
        print('    python -m mundial_llm.predict_match "Spain" "Brazil"')
        print("    python -m mundial_llm.evaluate_model --live-tournament")
        sys.exit(0)
    print(f"  {len(fixtures)} upcoming fixtures found.\n")

    print("  Loading historical international matches database cache...")
    from .cache_historical_data import load_cached_matches
    matches_cache = load_cached_matches()
    print(f"  {len(matches_cache)} historical matches loaded.")

    page = 0
    page_size = 15

    while True:
        print_menu(fixtures, page, page_size)

        try:
            raw = input("\n  Enter game number or command: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if raw == "Q":
            print("\n  Goodbye!")
            break
        elif raw == "N":
            total_pages = (len(fixtures) + page_size - 1) // page_size
            if (page + 1) < total_pages:
                page += 1
        elif raw == "P":
            if page > 0:
                page -= 1
        elif raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(fixtures):
                chosen = fixtures[idx]
                predict_match(chosen["home"], chosen["away"], model_name=model, fixture=chosen)
                input("  Press Enter to return to the menu...")
            else:
                print(f"  [!] Invalid number. Choose between 1 and {len(fixtures)}.")
        else:
            print("  [!] Unknown command. Enter a number, N, P, or Q.")


if __name__ == "__main__":
    main()
