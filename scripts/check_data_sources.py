"""
Smoke test for all data sources used by the Mundial LLM project.

Sources tested (in priority order):
  1. StatsBomb Open Data (statsbombpy) — free, no scraping, no API key.
     Full 2022 FIFA World Cup data including match events and shot maps.
  2. FBref (via soccerdata) — richest xG/shooting data, but often blocked
     by Cloudflare. Capped at 1 retry for speed.
  3. WhoScored (via soccerdata) — secondary scraping source.

Note: ESPN via soccerdata only supports domestic leagues (Premier League,
La Liga, etc.) and does NOT carry INT-World Cup data.

FBref and WhoScored need the optional `soccerdata` package
(pip install soccerdata); without it those two checks are skipped.

FBref and WhoScored were only explored early on and are not used by the app;
if you try them, respect those sites' terms of use.

Usage (from the repository root):
    python scripts/check_data_sources.py
"""

import sys
import traceback
import warnings

# Suppress the StatsBomb "open data access only" warning (expected behaviour)
warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

from statsbombpy import sb as statsbomb

try:
    import soccerdata as sd
except ImportError:
    sd = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try(name, fn):
    """Run fn() and return its result, or None + a printed reason on failure."""
    print(f"\n--- {name} ---")
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

# 2022 World Cup identifiers in StatsBomb Open Data
WC2022_COMPETITION_ID = 43
WC2022_SEASON_ID = 106


def fetch_statsbomb_schedule():
    """
    StatsBomb Open Data — competition list and 2022 WC match schedule.
    Free, zero scraping, data served directly from GitHub.
    """
    comps = statsbomb.competitions()
    wc_comps = comps[comps["competition_name"] == "FIFA World Cup"]
    print(f"[OK] StatsBomb competitions: found {len(wc_comps)} FIFA World Cup seasons available.")

    matches = statsbomb.matches(
        competition_id=WC2022_COMPETITION_ID,
        season_id=WC2022_SEASON_ID,
    )
    print(f"[OK] StatsBomb 2022 WC matches: {len(matches)} games.")
    cols = ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]
    print(matches[cols].head().to_string(index=False))
    return matches


def fetch_statsbomb_events(matches_df):
    """
    StatsBomb Open Data — detailed event stream for one match (shots, passes…).
    Picks the first match from the schedule.
    """
    first_match_id = int(matches_df["match_id"].iloc[0])
    events = statsbomb.events(match_id=first_match_id)
    shot_events = events[events["type"] == "Shot"]
    print(f"[OK] StatsBomb events for match {first_match_id}: {len(events)} events total, "
          f"{len(shot_events)} shots (xG data available).")
    if not shot_events.empty and "shot_statsbomb_xg" in shot_events.columns:
        print(shot_events[["player", "team", "shot_statsbomb_xg", "shot_outcome"]].head().to_string(index=False))
    return events


def fetch_statsbomb_lineups(matches_df):
    """
    StatsBomb Open Data — starting XIs and player data for one match.
    """
    first_match_id = int(matches_df["match_id"].iloc[0])
    lineups = statsbomb.lineups(match_id=first_match_id)
    teams = list(lineups.keys())
    print(f"[OK] StatsBomb lineups for match {first_match_id}: teams -> {teams}")
    for team, lineup_df in lineups.items():
        print(f"  {team}: {len(lineup_df)} players listed")
    return lineups


def fetch_fbref_shooting():
    """
    FBref (via soccerdata) — richest xG source but often Cloudflare-blocked.
    We attempt it once with no_store so we don't poison the cache.
    """
    fbref = sd.FBref(leagues="INT-World Cup", seasons="2022", no_store=True)
    stats = fbref.read_team_match_stats(stat_type="shooting")
    print(f"[OK] FBref team shooting stats: {len(stats)} rows.")
    print(stats.head().to_string())
    return stats


def fetch_whoscored_schedule():
    """WhoScored (via soccerdata) — secondary scraping source for the schedule."""
    ws = sd.WhoScored(leagues="INT-World Cup", seasons="2022")
    schedule = ws.read_schedule()
    print(f"[OK] WhoScored schedule: {len(schedule)} matches.")
    print(schedule.head().to_string())
    return schedule


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def main():
    results = {}

    # 1. StatsBomb schedule (primary — guaranteed for 2022 WC)
    results["statsbomb_schedule"] = _try(
        "StatsBomb Open Data - 2022 WC schedule", fetch_statsbomb_schedule
    )

    # 2. StatsBomb events (only if schedule loaded)
    if results["statsbomb_schedule"] is not None:
        results["statsbomb_events"] = _try(
            "StatsBomb Open Data - match events + xG", 
            lambda: fetch_statsbomb_events(results["statsbomb_schedule"])
        )
        results["statsbomb_lineups"] = _try(
            "StatsBomb Open Data - lineups",
            lambda: fetch_statsbomb_lineups(results["statsbomb_schedule"])
        )
    else:
        results["statsbomb_events"] = None
        results["statsbomb_lineups"] = None

    # 3-4. FBref and WhoScored — optional, need soccerdata (may be Cloudflare-blocked)
    if sd is None:
        print("\n--- FBref / WhoScored ---\n[SKIP] soccerdata is not installed (pip install soccerdata).")
    else:
        results["fbref_shooting"] = _try(
            "FBref team shooting stats (2022 WC)", fetch_fbref_shooting
        )
        results["whoscored_schedule"] = _try(
            "WhoScored schedule (2022 WC)", fetch_whoscored_schedule
        )

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATA SOURCE SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        status = "OK" if val is not None else "FAILED"
        print(f"  {key:35s} -> {status}")

    schedule_ok = results.get("statsbomb_schedule") is not None
    print("-" * 60)
    if schedule_ok:
        print("RESULT: Core data layer is functional (StatsBomb OK).")
        return 0
    print("RESULT: CRITICAL - Could not load any match data.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
