"""构建 LangGraph 执行图。"""

from __future__ import annotations

import random

from langgraph.graph import END, StateGraph

from game.state import GameState
from game.model import LLMAdapter
from game.logic import (
    assign_roles,
    generate_speeches,
    generate_words,
    judge_result,
    show_final_result,
    vote_undercover,
)


def route_game(state: GameState) -> str:
    """根据游戏状态决定继续下一轮还是结束。"""

    return "generate_speeches" if state["game_status"] == "running" else "show_final_result"


def build_game_graph(llm_adapter: LLMAdapter, rng: random.Random | None = None):
    """把 6 个模块组织为一张可执行状态图。"""

    graph = StateGraph(GameState)
    graph.add_node("generate_words", lambda state: generate_words(state, llm_adapter, rng=rng))
    graph.add_node("assign_roles", lambda state: assign_roles(state, rng=rng))
    graph.add_node("generate_speeches", lambda state: generate_speeches(state, llm_adapter))
    graph.add_node("vote_undercover", lambda state: vote_undercover(state, llm_adapter, rng=rng))
    graph.add_node("judge_result", lambda state: judge_result(state, rng=rng))
    graph.add_node("show_final_result", show_final_result)

    graph.set_entry_point("generate_words")
    graph.add_edge("generate_words", "assign_roles")
    graph.add_edge("assign_roles", "generate_speeches")
    graph.add_edge("generate_speeches", "vote_undercover")
    graph.add_edge("vote_undercover", "judge_result")
    graph.add_conditional_edges("judge_result", route_game)
    graph.add_edge("show_final_result", END)
    return graph
