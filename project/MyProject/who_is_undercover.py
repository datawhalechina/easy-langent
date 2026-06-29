# ================== 导入核心依赖 ==================
import random
import os
import json
import textwrap
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ================== 初始化大模型 ==================
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://qianfan.baidubce.com/v2",
    model="deepseek-r1-distill-qwen-32b",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()


# ================== 1. 定义游戏状态 ==================
class GameState(TypedDict):
    """
    🧠 全局游戏状态容器
    用于在各个智能体节点间传递数据，实现状态共享
    """
    civilian_word: str  # 🟢 平民词：多数人持有的相同词语
    undercover_word: str  # 🔴 卧底词：少数人持有的相似但不同的词语
    role_assignment: dict  # 👥 角色分配映射表
    speeches: dict  # 💬 当前轮次发言内容
    history_speeches: List[Dict[str, str]]  # 📜 历史发言记录（用于记忆）
    speech_reasoning: dict  # 🤔 发言策略思考过程
    votes: dict  # 🗳️ 当前轮次投票结果
    vote_reasoning: dict  # 🧐 投票决策思考过程
    game_status: str  # 🚦 游戏状态 (running/end)
    winner: str  # 🏆 获胜方
    eliminated: List[str]  # 🚫 已淘汰玩家列表
    round: int  # 🔢 当前游戏轮次


def init_game_state() -> GameState:
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
        "round": 1
    }


# ================== 2. 节点函数 ==================
def generate_words(state: GameState) -> GameState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """🌍 **游戏环境**：2026年4月28日，江西省赣州市，天气晴朗，一群好友正在聚会。

🎯 **角色：资深出题官**
请严格遵循以下规则生成词语对：
1. **语义关联性**：生成一组“形似神不似”的词语对（如：苹果-华为、饺子-包子）。
2. **博弈平衡**：词语必须足够相似以混淆视听，又要有细微差别以便推理。
3. **本地化**：优先考虑赣州本地常见或大众熟悉的日常词汇。
4. **格式**：仅输出JSON，禁止任何解释。

📝 **输出规范**：
{{"civilian": "词语1", "undercover": "词语2"}}
⚠️ 注意：这里的双大括号是固定格式，直接输出标准JSON字符串。"""),
        ("user", "请生成一组适合‘谁是卧底’的高难度词语对")
    ])
    chain = prompt | llm | parser
    result = chain.invoke({})

    try:
        word_data = json.loads(result.strip())
        civilian_word = word_data["civilian"]
        undercover_word = word_data["undercover"]
    except (json.JSONDecodeError, KeyError):
        # 增加本地化色彩的兜底词库
        fallback_pairs = [
            ("脐橙", "橘子"), ("米粉", "面条"), ("赣州塔", "电视塔"),
            ("电动车", "摩托车"), ("奶茶", "果茶"), ("电影", "电视剧")
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)

    state["civilian_word"] = civilian_word
    state["undercover_word"] = undercover_word
    print(f"\n🎯 **词语生成**：平民词『{civilian_word}』｜ 卧底词『{undercover_word}』")
    return state


# ---- 节点2：分配角色 ----
def assign_roles(state: GameState) -> GameState:
    agents = ["agent1", "agent2", "agent3", "agent4"]
    undercover = random.choice(agents)
    for agent in agents:
        if agent == undercover:
            state["role_assignment"][agent] = ("卧底", state["undercover_word"])
        else:
            state["role_assignment"][agent] = ("平民", state["civilian_word"])

    print("\n🎭 **角色分配完毕**")
    return state


# ---- 节点3：发言 ----
def generate_speeches(state: GameState) -> GameState:
    speeches = {}
    reasoning = {}
    current_round = state["round"]

    # 构建历史记忆上下文
    history_context = ""
    if state["history_speeches"]:
        history_context = "🔍 **历史线索回顾**（请避免重复描述）：\n"
        for idx, round_speeches in enumerate(state["history_speeches"], 1):
            history_context += f"  第{idx}轮："
            for agent, speech in round_speeches.items():
                if agent not in state["eliminated"]:
                    history_context += f" {agent}说'{speech}' |"
            history_context += "\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""⏰ **时间**：2026-04-28 | 📍 **地点**：江西赣州
🎯 **任务**：第{current_round}轮发言
📝 **规则**：
1. **身份保密**：绝对不能直接说出词语！
2. **长度控制**：15-30字左右的自然口语。
3. **策略差异**：
   - 🟢 平民：描述具体特征，寻找与众不同的细节。
   - 🔴 卧底：模糊描述，观察他人用词，随机应变。
4. **记忆**：参考历史发言，不要前后矛盾。
5. **格式**：JSON

{history_context}"""),
        ("user", "你的身份是：{role}，词语是：{word}。请生成发言和内心策略。")
    ])
    chain = prompt | llm | parser

    print(f"\n🗣️ **第{current_round}轮 · 发言阶段**")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue
        output = chain.invoke({"role": role, "word": word})

        try:
            speech_data = json.loads(output.strip())
            speech = speech_data.get("speech", "正在思考...")
            reason = speech_data.get("reason", "无")
        except json.JSONDecodeError:
            speech = f"（{role}）这是一个很难猜的游戏词，大家要小心。"
            reason = "JSON解析失败回退"

        speeches[agent] = speech
        reasoning[agent] = reason
        print(f"  ⚪ {agent} ({role})：{speech}")

    state["history_speeches"].append(speeches.copy())
    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    return state


def vote_undercover(state: GameState) -> GameState:
    votes = {}
    reasons = {}
    current_agents = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    current_round = state["round"]

    # 汇总当前轮发言
    speech_context = f"📋 **第{current_round}轮发言实录**：\n"
    for agent, speech in state["speeches"].items():
        speech_context += f"  - {agent}：{speech}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""🕵️ **任务**：第{current_round}轮 · 投票决策
📝 **分析指南**：
1. **逻辑排查**：对比当前发言与历史记录，寻找逻辑断层或矛盾点。
2. **心理博弈**：
   - 平民需揪出描述模糊、试图混淆视听的玩家。
   - 卧底需伪装成平民，投票给无辜者以保全自己。
3. **输出**：JSON"""),
        ("user", "当前局势：{speech_context}\n\n你的身份：{role}，词语：{word}\n请投票并简述理由（20字内）。")
    ])
    chain = prompt | llm | parser

    print(f"\n🗳️ **第{current_round}轮 · 投票阶段**")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue
        output = chain.invoke({
            "role": role,
            "word": word,
            "speech_context": speech_context
        })

        try:
            vote_data = json.loads(output.strip())
            vote = vote_data["vote"].strip()
            reason = vote_data["reason"]
        except (json.JSONDecodeError, KeyError):
            vote = random.choice([a for a in current_agents if a != agent])
            reason = "随机投票"

        # 简单校验
        if vote not in current_agents or vote == agent:
            vote = random.choice([a for a in current_agents if a != agent])

        votes[agent] = vote
        reasons[agent] = reason
        print(f"  ⚪ {agent} 投票给：{vote} (理由：{reason})")

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state


# ---- 节点5：裁决 ----
def judge_result(state: GameState) -> GameState:
    vote_count = {}
    for v in state["votes"].values():
        vote_count[v] = vote_count.get(v, 0) + 1
    max_vote = max(vote_count.values())
    # 如果平票，随机选一个
    eliminated_candidates = [a for a, c in vote_count.items() if c == max_vote]
    eliminated = random.choice(eliminated_candidates)
    state["eliminated"].append(eliminated)
    role = state["role_assignment"][eliminated][0]
    current_round = state["round"]

    print(f"\n❌ **第{current_round}轮结果**：{eliminated} ({role}) 被淘汰")

    remaining = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    civ_count = sum(1 for a in remaining if state["role_assignment"][a][0] == "平民")
    uc_count = sum(1 for a in remaining if state["role_assignment"][a][0] == "卧底")

    # 胜负判定逻辑
    if role == "卧底":
        state["game_status"] = "end"
        state["winner"] = "civilian"
        print("🎉 **平民胜利**：成功揪出了卧底！")
    elif civ_count == 1 and uc_count == 1:
        state["game_status"] = "end"
        state["winner"] = "undercover"
        print("🎉 **卧底胜利**：最后的幸存者！")
    else:
        state["game_status"] = "running"
        state["round"] += 1
        print(f"🔄 游戏继续，进入第{state['round']}轮")
    return state


# ---- 节点6：总结 ----
def show_final_result(state: GameState) -> GameState:
    # 使用 textwrap.dedent 去除多行字符串的公共缩进（虽然这里没缩进，但为了展示功能）
    # 使用 textwrap.fill 对长文本进行自动换行，防止终端显示溢出
    summary_lines = [
        f"🏁 **游戏终局 · 复盘报告**",
        f"🏆 获胜阵营：{'平民' if state['winner'] == 'civilian' else '卧底'}",
        f"🔑 词库揭秘：平民词『{state['civilian_word']}』 VS 卧底词『{state['undercover_word']}』",
        f"⏳ 总耗时：{state['round']} 轮",
        # 核心应用：使用 textwrap.fill 限制淘汰顺序的每行长度
        f"📉 淘汰详情：{textwrap.fill(' -> '.join(state['eliminated']), width=40, subsequent_indent='            ')}",
        f"📍 地点：江西赣州 | 📅 时间：2026年4月28日"
    ]

    # 打印分隔线和内容
    print(f"\n{'=' * 50}")
    for line in summary_lines:
        print(line)
    print(f"{'=' * 50}")

    return state


# ================== 3. 构建 LangGraph ==================
def build_game_graph():
    graph = StateGraph(GameState)
    # 注册所有节点
    graph.add_node("generate_words", generate_words)
    graph.add_node("assign_roles", assign_roles)
    graph.add_node("generate_speeches", generate_speeches)
    graph.add_node("vote_undercover", vote_undercover)
    graph.add_node("judge_result", judge_result)
    graph.add_node("show_final_result", show_final_result)

    graph.set_entry_point("generate_words")

    # 定义线性流程
    graph.add_edge("generate_words", "assign_roles")
    graph.add_edge("assign_roles", "generate_speeches")
    graph.add_edge("generate_speeches", "vote_undercover")
    graph.add_edge("vote_undercover", "judge_result")

    # 定义条件分支
    def should_continue(state: GameState):
        return "generate_speeches" if state["game_status"] == "running" else "show_final_result"

    graph.add_conditional_edges("judge_result", should_continue)
    graph.add_edge("show_final_result", END)

    return graph


# ================== 4. 入口 ==================
if __name__ == "__main__":
    game_graph = build_game_graph()
    game = game_graph.compile()

    print(f"{'=' * 60}")
    print("🎮 **谁是卧底** · 多智能体推理版")
    print("📍 当前场景：江西赣州 · 2026-04-28")
    print(f"{'=' * 60}")

    game.invoke(init_game_state())