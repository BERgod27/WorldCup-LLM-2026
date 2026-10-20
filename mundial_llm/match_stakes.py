"""
Mundial LLM - Match Stakes & Standings Analyzer
================================================
Fetches live group standings from the ESPN API, parses match situations,
and simulates mathematical group stage outcomes to determine tactical stakes
and risk biases for each team.
"""

import requests
import itertools

ESPN_STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings"
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    "?limit=200&dates=20260611-20260720"
)

def fetch_standings():
    """Fetch live standings from ESPN API."""
    try:
        r = requests.get(ESPN_STANDINGS_URL, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARNING] Failed to fetch live standings from ESPN: {e}")
        return None

def fetch_remaining_fixtures():
    """Fetch every 2026 World Cup event from the ESPN scoreboard; callers keep the ones not yet played."""
    try:
        r = requests.get(ESPN_SCOREBOARD, timeout=10)
        r.raise_for_status()
        return r.json().get("events", [])
    except Exception as e:
        print(f"  [WARNING] Failed to fetch upcoming fixtures from ESPN: {e}")
        return []

def clean_team_name(name):
    """Clean team name for comparison."""
    return name.lower().strip().replace(" ", "")

def same_team(a, b):
    """Loose name match, since ESPN and StatsBomb spell some teams differently."""
    a, b = clean_team_name(a), clean_team_name(b)
    return a == b or a in b or b in a

def find_group_and_entries(standings_data, team_name):
    """Find the group name and standings entries for a given team."""
    if not standings_data or 'children' not in standings_data:
        return None, []
    
    for group in standings_data['children']:
        group_name = group.get("name")
        entries = group.get("standings", {}).get("entries", [])
        for entry in entries:
            curr_name = entry.get("team", {}).get("displayName", "")
            if same_team(team_name, curr_name):
                return group_name, entries
    return None, []

def get_team_stats(entry):
    """Convert raw stats entry to a dictionary."""
    stats = {}
    for s in entry.get("stats", []):
        stats[s.get("name")] = s.get("value")
    return {
        'points': int(stats.get('points', 0)),
        'gf': int(stats.get('pointsFor', 0)),
        'ga': int(stats.get('pointsAgainst', 0)),
        'gd': int(stats.get('pointDifferential', 0)),
        'gp': int(stats.get('gamesPlayed', 0)),
        'rank': int(stats.get('rank', 0))
    }

def format_standings_table(entries):
    """Format group standings as a clean ASCII table."""
    lines = []
    lines.append(f"{'Rank':<5} {'Team':<20} {'GP':<4} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<4} {'Pts':<4}")
    lines.append("-" * 60)
    for entry in entries:
        t_name = entry.get("team", {}).get("displayName")
        stats = {}
        for s in entry.get("stats", []):
            stats[s.get("name")] = s.get("value")
            
        rank = int(stats.get("rank", 0))
        gp = int(stats.get("gamesPlayed", 0))
        w = int(stats.get("wins", 0))
        d = int(stats.get("ties", 0))
        l = int(stats.get("losses", 0))
        gf = int(stats.get("pointsFor", 0))
        ga = int(stats.get("pointsAgainst", 0))
        gd = int(stats.get("pointDifferential", 0))
        pts = int(stats.get("points", 0))
        
        gd_str = f"{gd:+}" if gd != 0 else "0"
        lines.append(f"{rank:<5} {t_name:<20} {gp:<4} {w:<3} {d:<3} {l:<3} {gf:<3} {ga:<3} {gd_str:<4} {pts:<4}")
    return "\n".join(lines)

def simulate_outcomes(group_teams, group_standings_dict, remaining_group_fixtures):
    """
    Simulates all combinations of the remaining matches in the group.
    Returns list of dicts with outcome details and sorted team lists.
    """
    outcomes = []
    
    # We simulate simple wins (1-0), draws (0-0), and losses (0-1)
    match_options = [
        ('home_win', 1, 0),
        ('draw', 0, 0),
        ('away_win', 0, 1)
    ]
    
    num_matches = len(remaining_group_fixtures)
    if num_matches == 0:
        sorted_teams = sorted(
            group_teams,
            key=lambda t: (group_standings_dict[t]['points'], group_standings_dict[t]['gd'], group_standings_dict[t]['gf']),
            reverse=True
        )
        return [{'outcomes': {}, 'standings': sorted_teams}]
        
    for combo in itertools.product(match_options, repeat=num_matches):
        sim_standings = {t: dict(stats) for t, stats in group_standings_dict.items()}
        fixture_outcomes = {}
        
        for idx, (outcome_type, home_gf, away_gf) in enumerate(combo):
            fix = remaining_group_fixtures[idx]
            home = fix['home']
            away = fix['away']
            
            fixture_outcomes[f"{home} vs {away}"] = outcome_type
            
            # Find closest matching team names in group_teams
            home_key = next((t for t in group_teams if same_team(t, home)), None)
            away_key = next((t for t in group_teams if same_team(t, away)), None)
            
            if not home_key or not away_key:
                continue
                
            sim_standings[home_key]['gp'] += 1
            sim_standings[away_key]['gp'] += 1
            sim_standings[home_key]['gf'] += home_gf
            sim_standings[home_key]['ga'] += away_gf
            sim_standings[home_key]['gd'] = sim_standings[home_key]['gf'] - sim_standings[home_key]['ga']
            sim_standings[away_key]['gf'] += away_gf
            sim_standings[away_key]['ga'] += home_gf
            sim_standings[away_key]['gd'] = sim_standings[away_key]['gf'] - sim_standings[away_key]['ga']
            
            if outcome_type == 'home_win':
                sim_standings[home_key]['points'] += 3
            elif outcome_type == 'draw':
                sim_standings[home_key]['points'] += 1
                sim_standings[away_key]['points'] += 1
            elif outcome_type == 'away_win':
                sim_standings[away_key]['points'] += 3
                
        sorted_teams = sorted(
            group_teams,
            key=lambda t: (sim_standings[t]['points'], sim_standings[t]['gd'], sim_standings[t]['gf']),
            reverse=True
        )
        
        outcomes.append({
            'outcomes': fixture_outcomes,
            'standings': sorted_teams,
            'standings_details': sim_standings
        })
        
    return outcomes

def classify_stakes(team, outcomes):
    """
    Analyze the set of possible final ranks for the team under different results.
    Returns a dict with stakes and tactical bias description.
    """
    # Group simulated final ranks of 'team' by their match result (Win, Draw, Loss)
    ranks_if_win = []
    ranks_if_draw = []
    ranks_if_loss = []
    
    for out in outcomes:
        # Find which result occurred for the team's match
        match_result = None
        for match_str, res in out['outcomes'].items():
            home_name, away_name = match_str.split(" vs ")
            if same_team(team, home_name):
                is_home = True
            elif same_team(team, away_name):
                is_home = False
            else:
                continue
            if res == 'draw':
                match_result = 'draw'
            elif (res == 'home_win' and is_home) or (res == 'away_win' and not is_home):
                match_result = 'win'
            else:
                match_result = 'loss'
            break
                
        if not match_result:
            # If no matches left, it defaults to the current rank
            final_rank = out['standings'].index(team) + 1
            ranks_if_win.append(final_rank)
            ranks_if_draw.append(final_rank)
            ranks_if_loss.append(final_rank)
        else:
            final_rank = out['standings'].index(team) + 1
            if match_result == 'win':
                ranks_if_win.append(final_rank)
            elif match_result == 'draw':
                ranks_if_draw.append(final_rank)
            elif match_result == 'loss':
                ranks_if_loss.append(final_rank)
                
    # Unique ranks
    r_win = sorted(list(set(ranks_if_win)))
    r_draw = sorted(list(set(ranks_if_draw)))
    r_loss = sorted(list(set(ranks_if_loss)))
    
    # Check general properties
    all_ranks = sorted(list(set(r_win + r_draw + r_loss)))
    
    # 1. Already Qualified (Top 2 guaranteed in all outcomes)
    if all_ranks and max(all_ranks) <= 2:
        return {
            "stakes": "Already Qualified. Guaranteed to finish in the top 2 of the group regardless of the result.",
            "bias": "No pressure. Expected to play safe, conserve energy, rotate key players to avoid injuries or cards, and minimize defensive risks."
        }
        
    # 2. Already Eliminated (Always finishes 4th, or 3rd with low/no points)
    if all_ranks and min(all_ranks) >= 4:
        return {
            "stakes": "Already Mathematically Eliminated. Cannot reach the knockout stages.",
            "bias": "No tournament pressure. Playing for pride. Expected to play with freedom, testing new tactics or younger players, but without taking extreme risks."
        }
        
    # 3. Must Win (Top 2/3 only possible if we win)
    can_qualify_win = min(r_win) <= 2 if r_win else False
    can_qualify_draw = min(r_draw) <= 2 if r_draw else False
    
    if can_qualify_win and not can_qualify_draw:
        return {
            "stakes": "MUST WIN to guarantee qualification. A draw or loss is highly likely to eliminate them.",
            "bias": "Must take maximum risks. Expected to press high, commit extra players to attack, play direct soccer, and play extremely aggressively, leaving themselves open to counter-attacks."
        }
        
    # 4. Draw is Sufficient (Draw guarantees top 2, Loss is risky)
    guaranteed_qualify_draw = max(r_draw) <= 2 if r_draw else False
    if guaranteed_qualify_draw:
        return {
            "stakes": "A Draw is sufficient to qualify. A loss puts their progression at serious risk.",
            "bias": "Priority on safety. Expected to focus heavily on defensive solidity, play a low block, maintain a compact structure, and avoid taking unnecessary risks."
        }
        
    # 5. Default: Standard Group Stage Stakes
    return {
        "stakes": "Needs a positive result to secure or maximize chances of qualification.",
        "bias": "Balanced stakes. Expected to play standard tactics, taking calculated risks as the match flow dictates."
    }

def standard_group_stakes(result, standings_text):
    """Fallback when the group situation can't be worked out: neutral group-stage stakes."""
    result["group_standings_text"] = standings_text
    result["home_stakes"] = "Group Stage match. Standard stakes: aim to win to progress."
    result["away_stakes"] = "Group Stage match. Standard stakes: aim to win to progress."
    result["home_bias"] = "Standard balanced approach."
    result["away_bias"] = "Standard balanced approach."
    return result

def analyze_match_stakes(home_team, away_team, stage_str):
    """
    Run the full stakes analysis for an upcoming fixture.
    """
    # 1. Check if knockout stage
    is_group = "group" in stage_str.lower()
    
    result = {
        "stage": "Group Stage" if is_group else "Knockout Stage",
        "group_name": "N/A",
        "group_standings_text": "N/A (Knockout Stage)",
        "home_stakes": "Single-elimination match. Must win to progress (via normal time, extra time, or penalty shootout).",
        "away_stakes": "Single-elimination match. Must win to progress (via normal time, extra time, or penalty shootout).",
        "home_bias": "Must take risks if trailing, but expected to play with extreme defensive discipline to avoid conceding first.",
        "away_bias": "Must take risks if trailing, but expected to play with extreme defensive discipline to avoid conceding first."
    }
    
    if not is_group:
        return result
        
    # 2. Get standings
    standings_data = fetch_standings()
    if not standings_data:
        # Fallback if API is offline
        return standard_group_stakes(result, "Standings currently unavailable.")
        
    group_name, entries = find_group_and_entries(standings_data, home_team)
    if not group_name or not entries:
        # Fallback if team not found in standings
        return standard_group_stakes(result, "Team standings not found.")
        
    result["group_name"] = group_name
    result["group_standings_text"] = format_standings_table(entries)
    
    # 3. Check games played
    group_stats = {}
    group_teams = []
    for entry in entries:
        name = entry.get("team", {}).get("displayName")
        stats = get_team_stats(entry)
        group_stats[name] = stats
        group_teams.append(name)
        
    # Find home and away entries in group_stats (ESPN might have slightly different names)
    home_key = next((t for t in group_teams if same_team(t, home_team)), None)
    away_key = next((t for t in group_teams if same_team(t, away_team)), None)
    
    if not home_key or not away_key:
        # The two teams are not in the same group, so the group table can't be used
        return standard_group_stakes(result, "N/A (teams are not in the same group)")
        
    gp_home = group_stats[home_key]['gp']
    gp_away = group_stats[away_key]['gp']
    
    # If it's matchday 1 or 2 (GP < 2)
    if gp_home < 2 or gp_away < 2:
        result["home_stakes"] = f"Early Group Stage match (Matchday {gp_home + 1}). Standard stakes: aim to secure points to build a solid base."
        result["away_stakes"] = f"Early Group Stage match (Matchday {gp_away + 1}). Standard stakes: aim to secure points to build a solid base."
        result["home_bias"] = "Balanced approach, taking standard calculated tactical risks."
        result["away_bias"] = "Balanced approach, taking standard calculated tactical risks."
        return result
        
    # 4. Matchday 3: Simulate outcomes
    # Find remaining group fixtures for these 4 teams
    fixtures = fetch_remaining_fixtures()
    group_clean_teams = [clean_team_name(t) for t in group_teams]
    
    remaining_group_fixtures = []
    for event in fixtures:
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("state", "")
        if status not in ("pre", "in"):
            continue
            
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
            
        home = competitors[0].get("team", {}).get("displayName", "")
        away = competitors[1].get("team", {}).get("displayName", "")
        
        # Check if BOTH teams belong to this group
        if clean_team_name(home) in group_clean_teams and clean_team_name(away) in group_clean_teams:
            remaining_group_fixtures.append({'home': home, 'away': away})

    # Without this match in the remaining schedule (schedule unavailable, or the match is
    # already over), the current table would be treated as final, so don't simulate.
    this_match_pending = any(
        (same_team(f['home'], home_key) and same_team(f['away'], away_key)) or
        (same_team(f['home'], away_key) and same_team(f['away'], home_key))
        for f in remaining_group_fixtures
    )
    if not this_match_pending:
        return standard_group_stakes(result, result["group_standings_text"])
            
    # Run simulation
    outcomes = simulate_outcomes(group_teams, group_stats, remaining_group_fixtures)
    
    # Classify stakes for both teams
    home_stakes_dict = classify_stakes(home_key, outcomes)
    away_stakes_dict = classify_stakes(away_key, outcomes)
    
    result["home_stakes"] = home_stakes_dict["stakes"]
    result["home_bias"] = home_stakes_dict["bias"]
    result["away_stakes"] = away_stakes_dict["stakes"]
    result["away_bias"] = away_stakes_dict["bias"]
    
    return result

if __name__ == "__main__":
    # Test script with mock team names
    import argparse
    parser = argparse.ArgumentParser(description="Test Match Stakes Analyzer")
    parser.add_argument("--home", type=str, default="France")
    parser.add_argument("--away", type=str, default="Senegal")
    parser.add_argument("--stage", type=str, default="group-stage")
    args = parser.parse_args()
    
    print(f"\nAnalyzing stakes for: {args.home} vs {args.away} ({args.stage})...")
    analysis = analyze_match_stakes(args.home, args.away, args.stage)
    for k, v in analysis.items():
        print(f"\n[{k.upper()}]:\n{v}")
