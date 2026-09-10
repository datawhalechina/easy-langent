import json
import random

from game.state import init_game_state
from game.graph import build_game_graph, route_game
from game.model import FakeLLMAdapter


def test_route_game_switches_by_status():
    state = init_game_state()
    assert route_game(state) == "generate_speeches"

    state["game_status"] = "end"
    assert route_game(state) == "show_final_result"


def test_graph_can_finish_one_complete_game():
    fake_llm = FakeLLMAdapter(
        [
            json.dumps({"civilian": "牙刷", "undercover": "牙膏"}),
            json.dumps({"speech": "这是清洁牙齿的工具。", "reason": "平民描述"}),
            json.dumps({"speech": "每天都会接触到。", "reason": "平民描述"}),
            json.dumps({"speech": "和口腔卫生有关。", "reason": "平民描述"}),
            json.dumps({"speech": "它有不同的味道。", "reason": "卧底描述"}),
            json.dumps({"vote": "agent4", "reason": "可疑"}),
            json.dumps({"vote": "agent4", "reason": "可疑"}),
            json.dumps({"vote": "agent4", "reason": "可疑"}),
            json.dumps({"vote": "agent1", "reason": "卧底自保"}),
        ]
    )

    compiled_graph = build_game_graph(fake_llm, rng=random.Random(0)).compile()
    final_state = compiled_graph.invoke(init_game_state())

    assert final_state["game_status"] == "end"
    assert final_state["winner"] == "civilian"
    assert final_state["eliminated"] == ["agent4"]
    assert final_state["civilian_word"] == "牙刷"
    assert final_state["undercover_word"] == "牙膏"
