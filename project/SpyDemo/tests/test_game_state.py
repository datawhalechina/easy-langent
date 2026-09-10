from game.state import AGENTS, FALLBACK_WORD_PAIRS, init_game_state


def test_init_game_state_defaults_are_complete():
    state = init_game_state()

    assert state["civilian_word"] == ""
    assert state["undercover_word"] == ""
    assert state["role_assignment"] == {}
    assert state["speeches"] == {}
    assert state["history_speeches"] == []
    assert state["speech_reasoning"] == {}
    assert state["votes"] == {}
    assert state["vote_reasoning"] == {}
    assert state["game_status"] == "running"
    assert state["winner"] == ""
    assert state["eliminated"] == []
    assert state["round"] == 1


def test_constants_are_prepared_for_game():
    assert AGENTS == ["agent1", "agent2", "agent3", "agent4"]
    assert len(FALLBACK_WORD_PAIRS) >= 1
