import json
import random

from game.state import AGENTS, init_game_state
from game.model import FakeLLMAdapter
from game.logic import (
    assign_roles,
    assign_roles_to_agents,
    build_history_context,
    count_votes,
    decide_game_outcome,
    generate_speeches,
    generate_words,
    judge_result,
    pick_eliminated_agent,
    validate_vote,
    vote_undercover,
)


def test_assign_roles_to_agents_has_one_undercover():
    assignments = assign_roles_to_agents("奶茶", "果汁", rng=random.Random(0))

    undercover_count = sum(1 for role, _ in assignments.values() if role == "卧底")
    civilian_count = sum(1 for role, _ in assignments.values() if role == "平民")

    assert undercover_count == 1
    assert civilian_count == 3
    assert set(assignments.keys()) == set(AGENTS)


def test_build_history_context_skips_eliminated_agents():
    history = [{"agent1": "a", "agent2": "b"}, {"agent1": "c", "agent3": "d"}]
    context = build_history_context(history, ["agent2"])

    assert "agent1：a" in context
    assert "agent3：d" in context
    assert "agent2：b" not in context


def test_validate_vote_rewrites_invalid_target():
    result = validate_vote("agent1", "agent1", ["agent1", "agent2", "agent3"], rng=random.Random(1))
    assert result in {"agent2", "agent3"}


def test_count_votes_and_pick_eliminated_agent():
    vote_count = count_votes({"agent1": "agent2", "agent2": "agent3", "agent3": "agent2"})
    eliminated = pick_eliminated_agent(vote_count, rng=random.Random(2))

    assert vote_count == {"agent2": 2, "agent3": 1}
    assert eliminated == "agent2"


def test_generate_words_uses_llm_json_result():
    state = init_game_state()
    fake_llm = FakeLLMAdapter([json.dumps({"civilian": "牙刷", "undercover": "牙膏"})])

    updated = generate_words(state, fake_llm)

    assert updated["civilian_word"] == "牙刷"
    assert updated["undercover_word"] == "牙膏"


def test_generate_words_falls_back_when_output_is_invalid():
    state = init_game_state()
    fake_llm = FakeLLMAdapter(["not-json"])

    updated = generate_words(state, fake_llm, rng=random.Random(0))

    assert (updated["civilian_word"], updated["undercover_word"]) == ("手机", "平板")


def test_assign_roles_writes_assignment_to_state():
    state = init_game_state()
    state["civilian_word"] = "奶茶"
    state["undercover_word"] = "果汁"

    updated = assign_roles(state, rng=random.Random(0))

    assert len(updated["role_assignment"]) == 4


def test_generate_speeches_skips_eliminated_and_uses_fallback():
    state = init_game_state()
    state["round"] = 1
    state["civilian_word"] = "奶茶"
    state["undercover_word"] = "果汁"
    state["role_assignment"] = {
        "agent1": ("平民", "奶茶"),
        "agent2": ("平民", "奶茶"),
        "agent3": ("平民", "奶茶"),
        "agent4": ("卧底", "果汁"),
    }
    state["eliminated"] = ["agent2"]
    fake_llm = FakeLLMAdapter(
        [
            json.dumps({"speech": "这是一种常见饮品。", "reason": "正常输出"}),
            "broken-output",
            json.dumps({"speech": "大家很熟悉。", "reason": "正常输出"}),
        ]
    )

    updated = generate_speeches(state, fake_llm)

    assert "agent2" not in updated["speeches"]
    assert updated["speeches"]["agent1"] == "这是一种常见饮品。"
    assert "日常生活中很常见" in updated["speeches"]["agent3"]
    assert len(updated["history_speeches"]) == 1


def test_vote_undercover_rewrites_invalid_votes():
    state = init_game_state()
    state["round"] = 1
    state["role_assignment"] = {
        "agent1": ("平民", "奶茶"),
        "agent2": ("平民", "奶茶"),
        "agent3": ("平民", "奶茶"),
        "agent4": ("卧底", "果汁"),
    }
    state["speeches"] = {
        "agent1": "描述1",
        "agent2": "描述2",
        "agent3": "描述3",
        "agent4": "描述4",
    }
    fake_llm = FakeLLMAdapter(
        [
            json.dumps({"vote": "agent1", "reason": "投自己，非法"}),
            json.dumps({"vote": "nobody", "reason": "不存在"}),
            "broken-output",
            json.dumps({"vote": "agent2", "reason": "正常"}),
        ]
    )

    updated = vote_undercover(state, fake_llm, rng=random.Random(0))

    for voter, target in updated["votes"].items():
        assert target in AGENTS
        assert target != voter


def test_judge_result_civilian_wins_when_undercover_out():
    state = init_game_state()
    state["role_assignment"] = {
        "agent1": ("平民", "奶茶"),
        "agent2": ("平民", "奶茶"),
        "agent3": ("平民", "奶茶"),
        "agent4": ("卧底", "果汁"),
    }
    state["votes"] = {"agent1": "agent4", "agent2": "agent4", "agent3": "agent4", "agent4": "agent1"}

    updated = judge_result(state, rng=random.Random(0))

    assert updated["game_status"] == "end"
    assert updated["winner"] == "civilian"
    assert updated["eliminated"] == ["agent4"]


def test_decide_game_outcome_undercover_wins_at_final_two():
    state = init_game_state()
    state["round"] = 2
    state["role_assignment"] = {
        "agent1": ("平民", "奶茶"),
        "agent2": ("平民", "奶茶"),
        "agent3": ("平民", "奶茶"),
        "agent4": ("卧底", "果汁"),
    }
    state["eliminated"] = ["agent2", "agent3"]

    status, winner, next_round = decide_game_outcome(state, "agent1")

    assert status == "end"
    assert winner == "undercover"
    assert next_round == 2
