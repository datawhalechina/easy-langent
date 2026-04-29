"""实现游戏节点与可测试的核心逻辑。"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from game.state import AGENTS, FALLBACK_WORD_PAIRS, GameState
from game.model import LLMAdapter


def build_history_context(history_speeches: List[Dict[str, str]], eliminated: List[str]) -> str:
    """把历史发言整理成文本，供后续提示词引用。"""

    if not history_speeches:
        return ""

    lines = ["【历史发言记录】"]
    for index, round_speeches in enumerate(history_speeches, start=1):
        lines.append(f"第{index}轮发言：")
        for agent, speech in round_speeches.items():
            if agent not in eliminated:
                lines.append(f"- {agent}：{speech}")
    lines.append("")
    return "\n".join(lines)


def assign_roles_to_agents(
    civilian_word: str,
    undercover_word: str,
    rng: random.Random | None = None,
) -> Dict[str, Tuple[str, str]]:
    """随机分配 1 个卧底和 3 个平民。"""

    rng = rng or random
    undercover = rng.choice(AGENTS)
    assignments: Dict[str, Tuple[str, str]] = {}
    for agent in AGENTS:
        if agent == undercover:
            assignments[agent] = ("卧底", undercover_word)
        else:
            assignments[agent] = ("平民", civilian_word)
    return assignments


def validate_vote(
    voter: str,
    vote: str,
    current_agents: List[str],
    rng: random.Random | None = None,
) -> str:
    """保证投票目标合法，避免投自己或投给不存在的人。"""

    if vote != voter and vote in current_agents:
        return vote

    rng = rng or random
    candidates = [agent for agent in current_agents if agent != voter]
    return rng.choice(candidates)


def count_votes(votes: Dict[str, str]) -> Dict[str, int]:
    """统计每位玩家获得的票数。"""

    result: Dict[str, int] = {}
    for target in votes.values():
        result[target] = result.get(target, 0) + 1
    return result


def pick_eliminated_agent(
    vote_count: Dict[str, int],
    rng: random.Random | None = None,
) -> str:
    """平票时随机淘汰一人，保持游戏能继续推进。"""

    rng = rng or random
    max_vote = max(vote_count.values())
    candidates = [agent for agent, count in vote_count.items() if count == max_vote]
    return rng.choice(candidates)


def decide_game_outcome(state: GameState, eliminated_agent: str) -> tuple[str, str, int]:
    """根据淘汰结果判断游戏是否结束，并给出下一轮轮次。"""

    eliminated_role = state["role_assignment"][eliminated_agent][0]
    remaining_agents = [agent for agent in AGENTS if agent not in state["eliminated"]]
    civilian_count = sum(
        1 for agent in remaining_agents if state["role_assignment"][agent][0] == "平民"
    )
    undercover_count = sum(
        1 for agent in remaining_agents if state["role_assignment"][agent][0] == "卧底"
    )

    if eliminated_role == "卧底":
        return "end", "civilian", state["round"]
    if civilian_count == 1 and undercover_count == 1:
        return "end", "undercover", state["round"]
    return "running", "", state["round"] + 1


def generate_words(
    state: GameState,
    llm_adapter: LLMAdapter,
    rng: random.Random | None = None,
) -> GameState:
    """调用模型生成词语对；若输出异常，则回退到本地词库。"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是专业的「谁是卧底」游戏出题人，需生成一组高质量的词语对。
核心要求：
1. 词语类型：日常物品/食品/场景，避免生僻词
2. 语义关系：平民词与卧底词高度相似但核心特征不同
3. 输出格式：必须严格返回 JSON，例如 {{\"civilian\": \"奶茶\", \"undercover\": \"果汁\"}}
禁止输出任何额外文字。""",
            ),
            ("user", "生成一组符合要求的谁是卧底词语对"),
        ]
    )
    raw_result = llm_adapter.invoke(prompt.format_messages())

    try:
        word_data = json.loads(raw_result.strip())
        state["civilian_word"] = word_data["civilian"]
        state["undercover_word"] = word_data["undercover"]
    except (json.JSONDecodeError, KeyError, TypeError):
        rng = rng or random
        civilian_word, undercover_word = rng.choice(FALLBACK_WORD_PAIRS)
        state["civilian_word"] = civilian_word
        state["undercover_word"] = undercover_word
    return state


def assign_roles(
    state: GameState,
    rng: random.Random | None = None,
) -> GameState:
    """把生成好的词语发给 4 个智能体。"""

    state["role_assignment"] = assign_roles_to_agents(
        state["civilian_word"],
        state["undercover_word"],
        rng=rng,
    )
    return state


def generate_speeches(state: GameState, llm_adapter: LLMAdapter) -> GameState:
    """为当前存活玩家生成发言，并保留策略说明。"""

    speeches: Dict[str, str] = {}
    reasoning: Dict[str, str] = {}
    history_context = build_history_context(state["history_speeches"], state["eliminated"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是「谁是卧底」游戏的资深玩家，需要生成一段合规发言。
要求：
1. 发言必须是完整自然句子。
2. 不能直接说出词语本身。
3. 输出必须严格是 JSON，格式：{{\"speech\": \"...\", \"reason\": \"...\"}}
4. 禁止输出任何 JSON 之外的文字。""",
            ),
            (
                "user",
                """当前轮次：{round}
历史上下文：
{history_context}
你的角色：{role}
你的词语：{word}""",
            ),
        ]
    )

    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue

        raw_result = llm_adapter.invoke(
            prompt.format_messages(
                round=state["round"],
                history_context=history_context or "无",
                role=role,
                word=word,
            )
        )

        try:
            speech_data = json.loads(raw_result.strip())
            speech = speech_data["speech"]
            reason = speech_data["reason"]
        except (json.JSONDecodeError, KeyError, TypeError):
            if role == "平民":
                speech = "这是日常生活中很常见的东西，使用频率很高，很多人每天都会接触到。"
                reason = "模型输出异常，改用平民兜底发言保证流程继续。"
            else:
                speech = "这是大家都很熟悉的事物，平时经常会遇到，但细节上不太容易一下说清楚。"
                reason = "模型输出异常，改用卧底兜底发言避免流程中断。"

        speeches[agent] = speech
        reasoning[agent] = reason

    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    state["history_speeches"].append(speeches.copy())
    return state


def vote_undercover(
    state: GameState,
    llm_adapter: LLMAdapter,
    rng: random.Random | None = None,
) -> GameState:
    """根据发言进行投票，并保证投票结果有效。"""

    current_agents = [agent for agent in AGENTS if agent not in state["eliminated"]]
    speech_context = "\n".join(
        f"{agent}：{speech}" for agent, speech in state["speeches"].items()
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是「谁是卧底」游戏中的理性玩家，请根据发言进行投票。
输出必须严格是 JSON，格式：{{\"vote\": \"agent2\", \"reason\": \"...\"}}
禁止输出任何额外文字。""",
            ),
            (
                "user",
                """当前轮次：{round}
全部发言：
{speech_context}
你的角色：{role}
你的词语：{word}""",
            ),
        ]
    )

    votes: Dict[str, str] = {}
    reasons: Dict[str, str] = {}

    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue

        raw_result = llm_adapter.invoke(
            prompt.format_messages(
                round=state["round"],
                speech_context=speech_context,
                role=role,
                word=word,
            )
        )

        try:
            vote_data = json.loads(raw_result.strip())
            vote = validate_vote(agent, vote_data["vote"].strip(), current_agents, rng=rng)
            reason = vote_data["reason"]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            vote = validate_vote(agent, "", current_agents, rng=rng)
            reason = "模型输出异常，改用兜底投票结果保证流程继续。"

        votes[agent] = vote
        reasons[agent] = reason

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state


def judge_result(
    state: GameState,
    rng: random.Random | None = None,
) -> GameState:
    """统计本轮结果，更新淘汰列表、胜负状态和轮次。"""

    vote_count = count_votes(state["votes"])
    eliminated_agent = pick_eliminated_agent(vote_count, rng=rng)
    state["eliminated"].append(eliminated_agent)

    game_status, winner, next_round = decide_game_outcome(state, eliminated_agent)
    state["game_status"] = game_status
    state["winner"] = winner
    state["round"] = next_round
    return state


def show_final_result(state: GameState) -> GameState:
    """结果展示节点本身不改变状态，只作为图的终点输出前汇总。"""

    return state
