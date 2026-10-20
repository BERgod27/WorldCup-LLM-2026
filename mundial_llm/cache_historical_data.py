"""
Mundial LLM - Historical Data Cache Compiler
=============================================
Fetches all matches from 6 major modern international competitions
and compiles/saves their xG data using thread pool concurrency.
This creates a local cache to avoid making slow API calls during prediction.

The cache is built from StatsBomb Open Data on first run and is not shipped
with the repository, because StatsBomb's user agreement does not allow the
data to be redistributed.
"""

import os
import time
import warnings
import pandas as pd
from statsbombpy import sb as statsbomb
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress the StatsBomb warning
warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

# Targets: senior men's international tournaments in modern era
TARGET_COMPETITIONS = [
    {"name": "FIFA World Cup 2022", "competition_id": 43, "season_id": 106},
    {"name": "FIFA World Cup 2018", "competition_id": 43, "season_id": 3},
    {"name": "UEFA Euro 2024", "competition_id": 55, "season_id": 282},
    {"name": "UEFA Euro 2020", "competition_id": 55, "season_id": 43},
    {"name": "Copa America 2024", "competition_id": 223, "season_id": 282},
    {"name": "African Cup of Nations 2023", "competition_id": 1267, "season_id": 107},
]

from .paths import HISTORICAL_MATCHES_FILE as CACHE_FILE

SHOOTOUT_PERIOD = 5  # StatsBomb puts penalty-shootout kicks in period 5

def match_xg(events: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float]:
    """Sum shot xG per team from a match's events, leaving out penalty-shootout kicks."""
    if events.empty or "type" not in events.columns:
        return 0.0, 0.0

    shots = events[events["type"] == "Shot"]
    if "period" in shots.columns:
        shots = shots[shots["period"] < SHOOTOUT_PERIOD]
    if shots.empty or "shot_statsbomb_xg" not in shots.columns:
        return 0.0, 0.0

    xg_by_team = shots.groupby("team")["shot_statsbomb_xg"].sum().to_dict()
    home_xg = round(float(xg_by_team.get(home_team, 0.0)), 2)
    away_xg = round(float(xg_by_team.get(away_team, 0.0)), 2)
    return home_xg, away_xg

def fetch_xg_for_match(match_id, home_team, away_team, attempts=3):
    """Fetch a match's events and sum xG per team. Retries, then raises, so a failed download never becomes 0.0 xG."""
    for attempt in range(1, attempts + 1):
        try:
            events = statsbomb.events(match_id=match_id)
            return match_xg(events, home_team, away_team)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)

def build_cache(max_workers=15):
    """Fetch all matches and events in parallel, then save to CSV."""
    print("\n============================================================")
    print("  COMPILING HISTORICAL INTERNATIONAL MATCHES DATA CACHE")
    print("  This takes about 1-2 minutes and only happens once...")
    print("============================================================\n")
    
    all_matches = []
    
    # 1. Fetch matches for all competitions
    for comp in TARGET_COMPETITIONS:
        name = comp["name"]
        cid = comp["competition_id"]
        sid = comp["season_id"]
        print(f"  Fetching schedule for {name}...")
        try:
            m = statsbomb.matches(competition_id=cid, season_id=sid)
            m["competition_name"] = " ".join(name.split(" ")[:-1]) # e.g. FIFA World Cup
            m["season_name"] = name.split(" ")[-1] # e.g. 2022
            all_matches.append(m)
        except Exception as e:
            print(f"  [ERROR] Failed to load schedule for {name}: {e}")
            print("  Cache not saved, so an incomplete dataset is never used. Check your connection and run again.")
            return False
        
    df = pd.concat(all_matches, ignore_index=True)
    total_matches = len(df)
    print(f"\n  Found {total_matches} matches across all tournaments.")
    print("  Fetching match-level shot event data & calculating xG...")
    
    # Pre-allocate xG columns
    df["home_team_xg"] = 0.0
    df["away_team_xg"] = 0.0
    
    # Fetch in parallel
    matches_to_fetch = []
    for idx, row in df.iterrows():
        matches_to_fetch.append((idx, row["match_id"], row["home_team"], row["away_team"]))
        
    completed_count = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create map of future to index
        future_to_idx = {
            executor.submit(fetch_xg_for_match, mid, home, away): idx 
            for idx, mid, home, away in matches_to_fetch
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                home_xg, away_xg = future.result()
                df.at[idx, "home_team_xg"] = home_xg
                df.at[idx, "away_team_xg"] = away_xg
            except Exception as e:
                failed.append((df.at[idx, "match_id"], e))
            
            completed_count += 1
            if completed_count % 30 == 0 or completed_count == total_matches:
                print(f"    Progress: {completed_count}/{total_matches} matches processed ({int(completed_count/total_matches*100)}%)")
                
    if failed:
        match_id, error = failed[0]
        print(f"\n  [ERROR] Could not download events for {len(failed)} match(es), e.g. match {match_id}: {error}")
        print("  Cache not saved, so missing xG is never stored as 0.0. Check your connection and run again.")
        return False

    # Select columns to save
    cols_to_save = [
        "match_id", "match_date", "competition_name", "season_name",
        "home_team", "away_team", "home_score", "away_score",
        "home_team_xg", "away_team_xg"
    ]
    
    # Filter to only existing columns
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    cache_df = df[cols_to_save].copy()
    
    # Save cache
    cache_df.to_csv(CACHE_FILE, index=False)
    print(f"\n  [SUCCESS] Compiled {len(cache_df)} matches and saved cache to '{os.path.basename(CACHE_FILE)}'!\n")
    return True

def load_cached_matches():
    """Load matches from CSV cache, or build it if missing."""
    if not os.path.exists(CACHE_FILE):
        success = build_cache()
        if not success:
            return pd.DataFrame()
            
    try:
        return pd.read_csv(CACHE_FILE)
    except Exception as e:
        print(f"  [ERROR] Failed to load cache file: {e}. Rebuilding...")
        success = build_cache()
        if success:
            return pd.read_csv(CACHE_FILE)
        return pd.DataFrame()

if __name__ == "__main__":
    raise SystemExit(0 if build_cache() else 1)
