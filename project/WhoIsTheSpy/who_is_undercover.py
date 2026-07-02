# ================== 导入核心依赖 ==================
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# ================== 初始化环境与模型 ==================
# 文件位置：D:\ai\easy-langent\project\WhoIsTheSpy\who_is_undercover.py
# .env 默认放在 D:\ai\easy-langent\.env，避免把密钥提交到作业目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

# 固定随机种子，方便复现实验结果；如果想每次随机，可删除这一行。
random.seed(20260701)

llm = None
if API_KEY:
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.7,
        max_tokens=500,
    )
else:
    print("[WARN] 未检测到 API_KEY，将使用本地兜底逻辑运行。")

parser = StrOutputParser()


# ================== 1. 定义游戏状态 ==================
class GameState(TypedDict):
    """
    游戏状态字典，存储整个游戏的关键数据。
    TypedDict 的作用：给字典里的键提供类型提示，降低键名写错的概率。
    """

    civilian_word: str
    undercover_word: str
    role_assignment: dict
    speeches: dict
    history_speeches: List[Dict[str, str]]
    speech_reasoning: dict
    votes: dict
    vote_reasoning: dict
    game_status: str
    winner: str
    eliminated: List[str]
    round: int


def init_game_state() -> GameState:
    """创建一局新游戏的初始状态。"""
    return {
        "civilian_word": "",
        "undercover_word": "",
        "role_assignment": {},
        "speeches": {},
        "history_speeches": [],
        "speech_reasoning": {},
        "votes": {},
        "vote_reasoning": {},
        "game_status": "running",
        "winner": "",
        "eliminated": [],
        "round": 1,
    }


def extract_json(text: str) -> dict:
    """
    尽量从大模型输出中提取 JSON。
    有些模型会返回 ```json ... ```，这里做一次清洗，提升稳定性。
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def invoke_json(prompt: ChatPromptTemplate, inputs: dict, fallback: dict) -> dict:
    """
    调用模型并解析 JSON。
    如果模型不可用、网络失败、返回格式不符合要求，就走 fallback，保证游戏流程不中断。
    """
    if llm is None:
        return fallback

    chain = prompt | llm | parser
    try:
        output = chain.invoke(inputs)
        return extract_json(output)
    except Exception as exc:
        print(f"[WARN] 模型调用或 JSON 解析失败，已使用兜底逻辑：{exc}")
        return fallback


# ================== 2. 节点函数 ==================
def generate_words(state: GameState) -> GameState:
    """节点1：生成平民词和卧底词。"""
    fallback_pairs = [
        ("奶茶", "果汁"),
        ("牙刷", "牙膏"),
        ("米饭", "面条"),
        ("手机", "平板"),
        ("篮球", "足球"),
        ("咖啡", "红茶"),
    ]
    fallback_civilian, fallback_undercover = random.choice(fallback_pairs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是专业的「谁是卧底」游戏出题人，需要生成一组高质量的词语对。
要求：
1. 词语类型：日常物品、食品或场景，避免生僻词。
2. 语义关系：平民词与卧底词相似但核心特征不同。
3. 难度适合4人游戏。
4. 只输出 JSON，格式示例：{{"civilian": "奶茶", "undercover": "果汁"}}。
禁止输出任何额外文字。""",
            ),
            ("user", "生成一组符合要求的谁是卧底词语对。"),
        ]
    )
    word_data = invoke_json(
        prompt,
        {},
        {"civilian": fallback_civilian, "undercover": fallback_undercover},
    )

    state["civilian_word"] = str(word_data.get("civilian", fallback_civilian))
    state["undercover_word"] = str(word_data.get("undercover", fallback_undercover))
    print(f"\n[OK] 词语生成完成：平民词={state['civilian_word']} ｜ 卧底词={state['undercover_word']}")
    return state


def assign_roles(state: GameState) -> GameState:
    """节点2：随机给4个智能体分配角色。"""
    agents = ["agent1", "agent2", "agent3", "agent4"]
    undercover = random.choice(agents)
    for agent in agents:
        if agent == undercover:
            state["role_assignment"][agent] = ("卧底", state["undercover_word"])
        else:
            state["role_assignment"][agent] = ("平民", state["civilian_word"])

    print("\n[OK] 角色分配完成：")
    for agent, (role, word) in state["role_assignment"].items():
        print(f"  {agent}：{role}（词语：{word}）")
    return state


def build_history_context(state: GameState) -> str:
    """把历史发言整理成提示词上下文，让智能体能参考前几轮发言。"""
    if not state["history_speeches"]:
        return "暂无历史发言。"

    history_context = "〖历史发言记录〗\n"
    for idx, round_speeches in enumerate(state["history_speeches"], 1):
        history_context += f"第{idx}轮发言：\n"
        for agent, speech in round_speeches.items():
            if agent not in state["eliminated"]:
                history_context += f"- {agent}：{speech}\n"
    return history_context


def fallback_speech(role: str, current_round: int) -> dict:
    """模型失败时的兜底发言。"""
    if role == "平民":
        return {
            "speech": f"第{current_round}轮发言：这是生活中常见的东西，使用频率较高，大家应该都接触过，具体特征比较容易观察。",
            "reason": "平民兜底发言，尽量描述常见特征但不直接暴露词语。",
        }
    return {
        "speech": f"第{current_round}轮发言：这个东西在日常生活中也经常出现，使用场景比较广泛，和大家描述的方向有不少相似处。",
        "reason": "卧底兜底发言，模仿平民描述并避免暴露身份。",
    }


def generate_speeches(state: GameState) -> GameState:
    """节点3：让未淘汰的智能体依次发言。"""
    speeches = {}
    reasoning = {}
    current_round = state["round"]
    history_context = build_history_context(state)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是「谁是卧底」游戏的资深玩家，现在是第{current_round}轮发言。
核心规则：
1. 发言必须描述词语特征，但不能直接说出词语本身。
2. 发言控制在10到100个汉字。
3. 平民要真实描述核心特征，帮助找出卧底。
4. 卧底要模仿平民风格，模糊差异，不暴露身份。
5. 结合历史发言，避免重复或前后矛盾。
6. 只输出 JSON，格式示例：{{"speech": "这是一种日常饮品，口味很多，外出时经常购买。", "reason": "描述常见特征，避免直接说词语。"}}。
禁止输出任何额外文字。

{history_context}""",
            ),
            ("user", "你的角色是{role}，拿到的词语是{word}。"),
        ]
    )

    print(f"\n[SPEECH] 第{current_round}轮发言阶段：")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue

        data = invoke_json(
            prompt,
            {
                "current_round": current_round,
                "history_context": history_context,
                "role": role,
                "word": word,
            },
            fallback_speech(role, current_round),
        )
        speech = str(data.get("speech", fallback_speech(role, current_round)["speech"]))
        reason = str(data.get("reason", fallback_speech(role, current_round)["reason"]))

        speeches[agent] = speech
        reasoning[agent] = reason
        print(f"\n{agent}（{role}）")
        print(f"  发言：{speech}")
        print(f"  策略：{reason}")

    state["history_speeches"].append(speeches.copy())
    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    return state


def fallback_vote(agent: str, current_agents: List[str], current_round: int) -> dict:
    """模型失败时的兜底投票，保证不会投给自己。"""
    candidates = [name for name in current_agents if name != agent]
    return {
        "vote": random.choice(candidates),
        "reason": f"第{current_round}轮模型分析不可用，使用兜底策略随机选择可疑玩家。",
    }


def vote_undercover(state: GameState) -> GameState:
    """节点4：根据当前发言和历史发言进行投票。"""
    votes = {}
    reasons = {}
    current_round = state["round"]
    current_agents = [agent for agent in state["role_assignment"] if agent not in state["eliminated"]]

    speech_context = f"〖第{current_round}轮发言〗\n"
    speech_context += "\n".join([f"{agent}：{speech}" for agent, speech in state["speeches"].items()])
    if len(state["history_speeches"]) > 1:
        speech_context += "\n\n〖历史发言参考〗\n"
        for idx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{idx}轮：\n"
            for agent, speech in round_speeches.items():
                if agent in current_agents:
                    speech_context += f"- {agent}：{speech}\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是「谁是卧底」游戏的理性玩家，需要基于当前轮和历史发言分析并投票。
投票规则：
1. 不能投自己。
2. 不能投已淘汰玩家。
3. 平民重点找描述偏离或前后矛盾的人。
4. 卧底要隐藏自己，选择看起来像平民的玩家投票。
5. 只输出 JSON，格式示例：{{"vote": "agent2", "reason": "agent2的描述与其他人明显不同。"}}。
禁止输出任何额外文字。

{speech_context}""",
            ),
            ("user", "你的身份：{agent}；你的角色：{role}；你的词语：{word}。请选择你要投票的玩家。"),
        ]
    )

    print(f"\n[VOTE] 第{current_round}轮投票阶段：")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue

        data = invoke_json(
            prompt,
            {
                "speech_context": speech_context,
                "agent": agent,
                "role": role,
                "word": word,
            },
            fallback_vote(agent, current_agents, current_round),
        )
        vote = str(data.get("vote", "")).strip()
        reason = str(data.get("reason", "未给出理由。"))

        if vote == agent or vote not in current_agents:
            fixed = fallback_vote(agent, current_agents, current_round)
            vote = fixed["vote"]
            reason = f"原投票无效，已修正。{fixed['reason']}"

        votes[agent] = vote
        reasons[agent] = reason
        print(f"\n{agent}（{role}）")
        print(f"  投票给：{vote}")
        print(f"  理由：{reason}")

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state


def judge_result(state: GameState) -> GameState:
    """节点5：统计投票并判断游戏是否结束。"""
    vote_count = {}
    for vote in state["votes"].values():
        vote_count[vote] = vote_count.get(vote, 0) + 1

    max_vote = max(vote_count.values())
    eliminated_candidates = [agent for agent, count in vote_count.items() if count == max_vote]
    eliminated = random.choice(eliminated_candidates)
    state["eliminated"].append(eliminated)

    role = state["role_assignment"][eliminated][0]
    current_round = state["round"]
    print(f"\n[OUT] 第{current_round}轮淘汰结果：{eliminated}（{role}）")

    remaining = [agent for agent in state["role_assignment"] if agent not in state["eliminated"]]
    civilian_count = sum(1 for agent in remaining if state["role_assignment"][agent][0] == "平民")
    undercover_count = sum(1 for agent in remaining if state["role_assignment"][agent][0] == "卧底")

    if role == "卧底":
        state["game_status"] = "end"
        state["winner"] = "civilian"
        print("[OK] 平民胜利！")
    elif civilian_count == 1 and undercover_count == 1:
        state["game_status"] = "end"
        state["winner"] = "undercover"
        print("[OK] 卧底胜利！")
    else:
        state["game_status"] = "running"
        state["round"] += 1
        print(f"[NEXT] 游戏继续，进入第{state['round']}轮。")

    return state


def show_final_result(state: GameState) -> GameState:
    """节点6：展示最终结果。"""
    print("\n" + "=" * 50)
    print("谁是卧底 · 游戏结束总结")
    print(f"胜利方：{'平民' if state['winner'] == 'civilian' else '卧底'}")
    print(f"平民词：{state['civilian_word']} | 卧底词：{state['undercover_word']}")
    print(f"总轮次：{state['round']}")
    print(f"淘汰顺序：{state['eliminated']}")
    print("=" * 50)
    return state


# ================== 3. 构建 LangGraph ==================
def build_game_graph():
    """
    构建游戏工作流。
    固定流程：词语生成 -> 角色分配 -> 发言 -> 投票 -> 裁决。
    条件流程：如果游戏未结束，回到发言；如果结束，进入总结。
    """
    graph = StateGraph(GameState)
    graph.add_node("generate_words", generate_words)
    graph.add_node("assign_roles", assign_roles)
    graph.add_node("generate_speeches", generate_speeches)
    graph.add_node("vote_undercover", vote_undercover)
    graph.add_node("judge_result", judge_result)
    graph.add_node("show_final_result", show_final_result)

    graph.set_entry_point("generate_words")
    graph.add_edge("generate_words", "assign_roles")
    graph.add_edge("assign_roles", "generate_speeches")
    graph.add_edge("generate_speeches", "vote_undercover")
    graph.add_edge("vote_undercover", "judge_result")

    def route(state: GameState) -> str:
        return "generate_speeches" if state["game_status"] == "running" else "show_final_result"

    graph.add_conditional_edges("judge_result", route)
    graph.add_edge("show_final_result", END)
    return graph


# ================== 4. 入口 ==================
if __name__ == "__main__":
    game_graph = build_game_graph()
    game = game_graph.compile()
    print("=" * 50)
    print("谁是卧底 · 多智能体多轮策略版 启动")
    print(f"模型配置：BASE_URL={BASE_URL} | MODEL={MODEL}")
    print("=" * 50)
    game.invoke(init_game_state())

