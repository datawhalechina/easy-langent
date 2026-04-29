import json
import os
from random import choices, choice
from typing import TypedDict

import faker
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# ========== 1. 环境配置 ==========
load_dotenv()

checkpointer = MemorySaver()
thread_id = "multi_agent_task_001"
config = {"configurable": {"thread_id": thread_id}}

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=500
).with_retry(
    stop_after_attempt=3,  # 最多重试 3 次
    wait_exponential_jitter=True,  # 指数退避 + 抖动（推荐）
    retry_if_exception_type=(
        ConnectionError,
        TimeoutError,
    ),
)

parser = StrOutputParser()


# ========== 2. 定义状态 ==========
class DebateState(TypedDict):
    topic: str  # 辩题
    user_input: str  # 根据用户主题生成辩题
    sides: dict  # 角色分配：{agent1: ("正方"/"反方", 角色名), ...}
    opening_statements: dict  # 开篇陈词：{agent1: "陈词内容", ...}
    debate_round1: dict  # 第1轮自由辩论：{agent1: "发言", ...}
    debate_round2: dict  # 第2轮自由辩论：{agent1: "发言", ...}
    closing_statements: dict  # 总结陈词：{agent1: "陈词", ...}
    judge_result: str  # 裁判评判结果
    winner: str  # 获胜方：正方/反方/平局


# ========== 3. 节点函数 ==========
def generate_topic(state: DebateState) -> DebateState:
    """生成辩题"""
    print(f"\n{'=' * 50}")
    print("📜 生成辩题")
    print(f"{'=' * 50}\n")
    prompt = ChatPromptTemplate([
        ("system", """你是辩论赛出题人。根据用户给出的主题，生成一个具有可辩性的辩题，并按以下格式输出：

    辩题：[辩论陈述]
        难度等级：（初级/中级/高级）
        所属领域：（如：社会文化/政策制定/道德哲学）
        核心讨论：一句话概括争议点。
        正方立场：一句话。
        反方立场：一句话。"""),
        ("user", "{user_input}")
    ])
    chain = prompt | llm | parser
    result = chain.invoke(state)
    if not result:
        topic = "AI是否会取代人类工作"  # 保底
    else:
        topic = result

    state["topic"] = topic
    print(topic)
    return state


def assign_roles(state: DebateState) -> DebateState:
    """分配正方/反方"""
    print(f"\n{'=' * 50}")
    print("📜 角色分配")
    print(f"{'=' * 50}\n")
    agents = ["agent1", "agent2", "agent3", "agent4"]
    affirmative = choices(agents, k=2)  # 抽2名正方，2名反方
    sides = {}
    for agent in agents:
        name = faker.Faker("zh_CN").name()
        side = "正方" if agent in affirmative else "反方"
        sides[agent] = (side, name)
        print(side, name, agent)
    state["sides"] = sides
    return state


def opening_statement(state: DebateState) -> DebateState:
    """开篇陈词"""
    print(f"\n{'=' * 50}")
    print("📜 开篇陈词")
    print(f"{'=' * 50}\n")
    prompt = ChatPromptTemplate([
        ("system",
         "你是辩论赛的参赛选手,辩论赛主题{topic}。\n你是：{agent},姓名：{name},立场：{standpoint},\n参赛选手：{sides}。"),
        ("user", "现在开篇陈词。输出立场相关的50-100字")
    ])
    chain = prompt | llm | parser
    opening_statements = {}
    for agent, (standpoint, name) in state["sides"].items():
        result = chain.invoke({**state, "agent": agent, "standpoint": standpoint, "name": name})
        opening_statements[agent] = result
        print(name + f"({standpoint}):", result.strip(), "\n")
    state["opening_statements"] = opening_statements
    return state


def debate_round1(state: DebateState) -> DebateState:
    """第1轮自由辩论"""
    print(f"\n{'=' * 50}")
    print("📜 第1轮自由辩论")
    print(f"{'=' * 50}\n")
    prompt = ChatPromptTemplate([
        ("system",
         "你是辩论赛的参赛选手,辩论赛主题{topic}。\n你是：{agent},姓名：{name},立场：{standpoint},\n参赛选手：{sides}。历史信息：{history}"),
        ("user", "当前是第1轮自由辩论，你需要针对对方刚才的开篇陈词进行反驳")
    ])
    chain = prompt | llm | parser
    debate_round1 = {}
    for agent, (standpoint, name) in state["sides"].items():
        result = chain.invoke({**state, "agent": agent, "standpoint": standpoint, "name": name,
                               "history": json.dumps(state.get("opening_statements", {}))})
        debate_round1[agent] = result
        print(name + f"({standpoint}):", result.strip(), "\n")
    state["debate_round1"] = debate_round1
    return state


def debate_round2(state: DebateState) -> DebateState:
    """第2轮自由辩论"""
    print(f"\n{'=' * 50}")
    print("📜 第2轮自由辩论")
    print(f"{'=' * 50}\n")
    prompt = ChatPromptTemplate([
        ("system",
         "你是辩论赛的参赛选手,辩论赛主题{topic}。\n你是：{agent},姓名：{name},立场：{standpoint},\n参赛选手：{sides}。历史信息：{history}"),
        ("user", "当前是第2轮自由辩论，你需要针对对方第1轮的发言进行反驳，并深化自己的论点,50-100字")
    ])
    chain = prompt | llm | parser
    debate_round2 = {}
    for agent, (standpoint, name) in state["sides"].items():
        result = chain.invoke({**state, "agent": agent, "standpoint": standpoint, "name": name,
                               "history": json.dumps(state.get("debate_round1", {}))})
        debate_round2[agent] = result
        print(name + f"({standpoint}):", result.strip(), "\n")
    state["debate_round2"] = debate_round2
    return state


def closing_statement(state: DebateState) -> DebateState:
    """总结陈词"""
    print(f"\n{'=' * 50}")
    print("📝 总结陈词阶段")
    print(f"{'=' * 50}\n")

    # 选正反方代表
    positive_agents = [a for a, (side, name) in state["sides"].items() if side == "正方"]
    negative_agents = [a for a, (side, name) in state["sides"].items() if side == "反方"]

    # 随机选代表
    positive_spokesperson = choice(positive_agents)
    positive_name = state["sides"][positive_spokesperson][-1]
    negative_spokesperson = choice(negative_agents)
    negative_name = state["sides"][negative_spokesperson][-1]

    speeches = {}

    # 正方总结陈词
    prompt_positive = ChatPromptTemplate.from_messages([
        ("system", f"""你是辩论赛的正方总结陈词辩手。

当前辩题：{state['topic']}
你的论点：{state["sides"][positive_spokesperson][1]}

【辩论全程回顾】
【开篇陈词】
{positive_name}：{state["opening_statements"][positive_spokesperson]}
【第1轮辩论】
{positive_name}：{state["debate_round1"][positive_spokesperson]}
【第2轮辩论】
{positive_name}：{state["debate_round2"][positive_spokesperson]}

【反方开篇陈词】
{negative_name}：{state["opening_statements"][negative_spokesperson]}

请总结正方核心论点，指出反方主要漏洞，给出最终结论（100-150字）"""),
        ("user", "请生成总结陈词")
    ])

    chain = prompt_positive | llm | parser
    positive_speech = chain.invoke({}).strip()
    speeches[positive_spokesperson] = positive_speech
    print(f"{positive_name}（正方总结）：{positive_speech}\n")

    # 反方总结陈词
    prompt_negative = ChatPromptTemplate.from_messages([
        ("system", f"""你是辩论赛的反方总结陈词辩手。

当前辩题：{state['topic']}
你的论点：{state["sides"][negative_spokesperson][1]}

【辩论全程回顾】
【开篇陈词】
{negative_name}：{state["opening_statements"][negative_spokesperson]}
【第1轮辩论】
{negative_name}：{state["debate_round1"][negative_spokesperson]}
【第2轮辩论】
{negative_name}：{state["debate_round2"][negative_spokesperson]}

【正方总结陈词（刚听完】
{positive_name}：{positive_speech}

请总结反方核心论点，指出正方主要漏洞，给出最终结论（100-150字）"""),
        ("user", "请生成总结陈词")
    ])

    chain = prompt_negative | llm | parser
    negative_speech = chain.invoke({}).strip()
    speeches[negative_spokesperson] = negative_speech
    print(f"{negative_name}（反方总结）：{negative_speech}\n")

    state["closing_statements"] = speeches
    return state


def judge_result(state: DebateState) -> DebateState:
    print(f"\n{'=' * 50}")
    print("📜 裁判评判")
    print(f"{'=' * 50}\n")

    # 1. 收集辩论全程记录
    debate_history = f"【辩题】{state['topic']}\n\n"

    debate_history += "【开篇陈词】\n"
    for agent, (side, name) in state["sides"].items():
        debate_history += f"{side} {name}：{state['opening_statements'][agent]}\n"

    debate_history += "\n【第1轮辩论】\n"
    for agent, (side, name) in state["sides"].items():
        debate_history += f"{side} {name}：{state['debate_round1'][agent]}\n"

    debate_history += "\n【第2轮辩论】\n"
    for agent, (side, name) in state["sides"].items():
        debate_history += f"{side} {name}：{state['debate_round2'][agent]}\n"

    debate_history += "\n【总结陈词】\n"
    for agent, speech in state["closing_statements"].items():
        side = state["sides"][agent][0]
        debate_history += f"{side} {agent}：{speech}\n"

    # print(debate_history)
    # 2. 构建评判提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位专业的辩论赛裁判，具有丰富的评判经验。
请根据以下评判标准对辩论进行评估：

【评判标准及权重】
1. 论点鲜明度（25%）：核心观点是否清晰明确，立场是否坚定
2. 论证逻辑性（25%）：论据是否充分，推理是否严密，是否存在逻辑漏洞
3. 应辩能力（25%）：对对方反驳的回应是否有力，是否能化解质疑
4. 说服力（25%）：整体论证是否令人信服，是否有感染力

【输出格式要求】
请按照以下JSON格式输出：
{{
    "winner": "正方/反方/平局",
    "reason": "详细评判理由（100-200字）",
    "positive_score": 正方总分（0-100）,
    "negative_score": 反方总分（0-100）,
    "breakdown": {{
        "论点鲜明度": {{"正方": 得分, "反方": 得分}},
        "论证逻辑性": {{"正方": 得分, "反方": 得分}},
        "应辩能力": {{"正方": 得分, "反方": 得分}},
        "说服力": {{"正方": 得分, "反方": 得分}}
    }}
}}


注意：请确保输出是有效的JSON格式，不要有任何额外文字。"""),
        ("user", "请评判以下辩论赛：\n\n{debate_history}")
    ])

    # 3. 调用LLM进行评判
    chain = prompt | llm | parser
    result = chain.invoke({"debate_history": debate_history})
    # 4. 解析结果（处理可能的格式问题）
    try:
        import json
        result_dict = json.loads(result)
        winner = result_dict["winner"]
        reason = result_dict["reason"]
        scores = result_dict.get("breakdown", {})

        # 打印评判详情
        print("📊 评分详情：")
        for dimension, scores_dict in scores.items():
            print(f"  {dimension}：正方 {scores_dict['正方']}分 | 反方 {scores_dict['反方']}分")

        print(f"\n🏆 获胜方：{winner}")
        print(f"📝 评判理由：{reason}")

        state["judge_result"] = result_dict
        state["winner"] = winner

    except Exception as e:
        # 格式解析失败，使用简单评判逻辑
        print(f"⚠️ 自动评判异常，使用简易模式")
        # 简单规则：总结陈词中提到"对方漏洞"次数 + 论点数量
        positive_count = result.count("正方") + result.count("支持")
        negative_count = result.count("反方") + result.count("反对")

        if positive_count > negative_count:
            winner = "正方"
        elif negative_count > positive_count:
            winner = "反方"
        else:
            winner = "平局"

        state["judge_result"] = {"winner": winner, "reason": "自动评判"}
        state["winner"] = winner
        print(f"🏆 获胜方：{winner}")

    return state


# ========== 4. 构建图 ==========
def build_debate_graph():
    graph = StateGraph(DebateState)
    graph.add_node("generate_topic", generate_topic)  # 生成赛题
    graph.add_node("assign_roles", assign_roles)  # 分配角色，2正2反
    graph.add_node("opening_statement", opening_statement)  # 开篇陈词
    graph.add_node("debate_round1", debate_round1)  # 第1轮辩论
    graph.add_node("debate_round2", debate_round2)  # 第2轮辩论
    graph.add_node("closing_statement", closing_statement)  # 总结陈词阶段
    graph.add_node("judge_result", judge_result)  # 裁判评判

    graph.set_entry_point("generate_topic")
    graph.add_edge("generate_topic", "assign_roles")
    graph.add_edge("assign_roles", "opening_statement")
    graph.add_edge("opening_statement", "debate_round1")
    graph.add_edge("debate_round1", "debate_round2")
    graph.add_edge("debate_round2", "closing_statement")
    graph.add_edge("closing_statement", "judge_result")
    graph.add_edge("judge_result", END)
    return graph.compile(checkpointer=checkpointer)


# ========== 5. 运行 ==========
if __name__ == "__main__":
    app = build_debate_graph()
    init = DebateState(
        topic="",
        user_input="AI是否会取代人类工作",
        sides={},
        opening_statements={},
        debate_round1={},
        debate_round2={},
        closing_statements={},
        judge_result="",
        winner="",
    )
    print(f"\n{'=' * 50}")
    print("📝 输入辩题方向（示例：AI是否会取代人类工作）")
    print(f"{'=' * 50}\n")

    user_input = input("").strip()
    if user_input:
        init["user_input"] = user_input
    app.invoke(init,config=config)
