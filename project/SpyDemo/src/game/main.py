"""程序入口：加载模型并运行一局完整游戏。"""

from __future__ import annotations

from game.state import init_game_state
from game.graph import build_game_graph
from game.model import build_default_llm


def print_summary(state: dict) -> None:
    """用较友好的方式展示最终结果，方便答辩演示。"""

    print("=" * 50)
    print("谁是卧底 · 多智能体多轮策略版")
    print("=" * 50)
    print(f"平民词：{state['civilian_word']}")
    print(f"卧底词：{state['undercover_word']}")
    print(f"总轮次：{state['round']}")
    print(f"淘汰顺序：{state['eliminated']}")
    print(f"胜利方：{'平民' if state['winner'] == 'civilian' else '卧底'}")

    print("\n本轮发言记录：")
    for agent, speech in state["speeches"].items():
        print(f"- {agent}: {speech}")

    print("\n本轮投票记录：")
    for agent, vote in state["votes"].items():
        print(f"- {agent} -> {vote}")


def main() -> None:
    """运行完整游戏流程。"""

    llm_adapter = build_default_llm()
    game_graph = build_game_graph(llm_adapter).compile()
    final_state = game_graph.invoke(init_game_state())
    print_summary(final_state)


if __name__ == "__main__":
    main()
