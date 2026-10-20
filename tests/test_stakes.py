from mundial_llm import match_stakes
from mundial_llm.match_stakes import analyze_match_stakes, classify_stakes, simulate_outcomes


def standing(points, gf, ga):
    return {"points": points, "gf": gf, "ga": ga, "gd": gf - ga, "gp": 2, "rank": 0}


def stakes_for(standings, fixtures):
    teams = list(standings)
    outcomes = simulate_outcomes(teams, standings, fixtures)
    return {team: classify_stakes(team, outcomes)["stakes"] for team in teams}


def test_matchday_three_stakes():
    # France have 6 points; Senegal's goal difference is so poor that only a win can lift them into the top 2.
    standings = {
        "France": standing(6, 5, 0),
        "Peru": standing(4, 3, 2),
        "Senegal": standing(3, 1, 4),
        "Japan": standing(1, 1, 4),
    }
    fixtures = [{"home": "France", "away": "Senegal"}, {"home": "Peru", "away": "Japan"}]
    stakes = stakes_for(standings, fixtures)

    assert stakes["France"].startswith("Already Qualified")
    assert stakes["Senegal"].startswith("MUST WIN")


def test_draw_is_enough():
    # A draw puts Brazil and Serbia on 5, out of reach; a loss lets a Swiss win overtake on goal difference.
    standings = {
        "Brazil": standing(4, 2, 1),
        "Serbia": standing(4, 2, 1),
        "Cameroon": standing(1, 1, 3),
        "Switzerland": standing(1, 1, 1),
    }
    fixtures = [{"home": "Brazil", "away": "Serbia"}, {"home": "Cameroon", "away": "Switzerland"}]
    stakes = stakes_for(standings, fixtures)

    assert stakes["Brazil"].startswith("A Draw is sufficient")
    assert stakes["Serbia"].startswith("A Draw is sufficient")


def test_name_spelling_can_differ_between_feeds():
    # The scoreboard and the standings feed don't always spell or capitalise a team the same way.
    standings = {
        "United States": standing(6, 4, 0),
        "Wales": standing(3, 2, 2),
        "Iran": standing(3, 2, 3),
        "England": standing(0, 0, 3),
    }
    fixtures = [{"home": "united states", "away": "Wales"}, {"home": "Iran", "away": "England"}]
    stakes = stakes_for(standings, fixtures)
    assert stakes["United States"].startswith("Already Qualified")


def espn_entry(name, points, gp):
    stats = {"points": points, "pointsFor": 0, "pointsAgainst": 0, "pointDifferential": 0, "gamesPlayed": gp, "rank": 0}
    return {"team": {"displayName": name}, "stats": [{"name": k, "value": v} for k, v in stats.items()]}


def espn_event(home, away, state):
    return {"competitions": [{"status": {"type": {"state": state}},
                              "competitors": [{"team": {"displayName": home}}, {"team": {"displayName": away}}]}]}


GROUP_K = {"children": [{"name": "Group K", "standings": {"entries": [
    espn_entry("Portugal", 4, 2), espn_entry("Colombia", 4, 2), espn_entry("Uzbekistan", 3, 2), espn_entry("Jamaica", 0, 2),
]}}]}


def test_teams_from_different_groups_get_group_stakes_not_knockout_text(monkeypatch):
    monkeypatch.setattr(match_stakes, "fetch_standings", lambda: GROUP_K)
    result = analyze_match_stakes("Portugal", "France", "group-stage")
    assert result["stage"] == "Group Stage"
    assert "Single-elimination" not in result["home_stakes"]


def test_no_simulation_when_the_match_is_not_in_the_remaining_schedule(monkeypatch):
    # Schedule unavailable (or the match already played): the current table must not be treated as final.
    monkeypatch.setattr(match_stakes, "fetch_standings", lambda: GROUP_K)
    monkeypatch.setattr(match_stakes, "fetch_remaining_fixtures", lambda: [])
    result = analyze_match_stakes("Colombia", "Jamaica", "group-stage")
    assert result["home_stakes"].startswith("Group Stage match. Standard stakes")
    assert result["away_stakes"].startswith("Group Stage match. Standard stakes")


def test_matchday_three_is_simulated_when_the_match_is_still_to_play(monkeypatch):
    monkeypatch.setattr(match_stakes, "fetch_standings", lambda: GROUP_K)
    monkeypatch.setattr(match_stakes, "fetch_remaining_fixtures", lambda: [
        espn_event("Colombia", "Jamaica", "pre"), espn_event("Portugal", "Uzbekistan", "pre"),
    ])
    result = analyze_match_stakes("Colombia", "Jamaica", "group-stage")
    assert not result["home_stakes"].startswith("Group Stage match. Standard stakes")
