import pytest

from mundial_llm import mundial_menu


def test_finished_tournament_is_explained_not_blamed_on_the_network(monkeypatch, capsys):
    monkeypatch.setattr(mundial_menu, "fetch_remaining_fixtures", lambda: [])
    monkeypatch.setattr("sys.argv", ["mundial_llm"])
    with pytest.raises(SystemExit) as exit_info:
        mundial_menu.main()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "World Cup is over" in out
    assert "internet" not in out


def test_network_failure_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(mundial_menu, "fetch_remaining_fixtures", lambda: None)
    monkeypatch.setattr("sys.argv", ["mundial_llm"])
    with pytest.raises(SystemExit) as exit_info:
        mundial_menu.main()
    assert exit_info.value.code == 1
    assert "internet" in capsys.readouterr().out
