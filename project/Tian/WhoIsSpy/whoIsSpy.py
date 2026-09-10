import os
import random
import textwrap
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,END

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = os.getenv("ZHIPU_BASE_URL")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="glm-5.2",
    temperature=0.7
)

# 定义游戏状态
class GameState(TypedDict):
    civilian_word: str
    undercover_word: str

    role_assignment: dict # 角色分配：{agent1:("平民"/"卧底",词语),...}

    speeches: dict # 当前轮发言：{agent1:"发言内容",...}
    history_speeches: List[Dict[str, str]] # 历史发言列表：[第1轮发言:{agent1:"发言内容",...}，第2轮发言,...]

    votes: dict # 当前轮投票：{agent1:"投给agent2",...}
    vote_reasoning: dict # 投票理由：{agent1:"理由",...}

    game_status: str # 游戏状态：running(游戏中)/end(结束)
    winner: str # 获胜方：civilian(平民) / undercover(卧底)

    eliminated: List[str] # 被淘汰的玩家列表
    round: int # 当前游戏轮次

# 游戏初始状态
def init_game_state() -> GameState:
    return {
        "civilian_word" : "",
        "undercover_word" : "",
        "role_assignment" : {},
        "speeches" : {},
        "history_speeches" : [],
        "votes" : {},
        "vote_reasoning" : {},
        "game_status" : "running",
        "winner" : "",
        "eliminated" : [],
        "round" : 1,
    }

# 节点1:词语生成模块
def generate_words(state: GameState) -> GameState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的「谁是卧底」游戏出题人，需生成一组高质量的词语对。
    核心要求：
    1. 词语类型：日常物品/食品/场景（如：奶茶-果汁、牙刷-牙膏），避免生僻词
    2. 语义关系：平民词与卧底词高度相似但核心特征不同，有足够博弈空间
    3. 难度适配：适合4人游戏，既不轻易暴露也能通过描述区分
    4. 输出格式：必须严格按照 JSON 格式输出，示例：{{"civilian": "奶茶", "undercover": "果汁"}}
    禁止输出任何额外文字，只返回JSON字符串！"""),
        ("user", "生成一组符合要求的谁是卧底词语对")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({})

    try:
        civilian_word = result["civilian"]
        undercover_word = result["undercover"]
    except KeyError:
        fallback_pairs = [
            ("奶茶", "果汁"), ("牙刷", "牙膏"), ("米饭", "面条"),
            ("手机", "平板"), ("篮球", "足球"), ("咖啡", "红茶")
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)

    # 状态更新：推荐只返回更新的状态，不返回完整的状态
    return {
        "civilian_word": civilian_word,
        "undercover_word": undercover_word,
    }

# 节点2:角色分配模块
def assign_roles(state: GameState) -> GameState:
    agents = ["agent1", "agent2", "user", "agent3"]
    undercover = random.choice(agents)

    role_assignment = {}
    for agent in agents:
        if agent == undercover:
            role_assignment[agent] = ("卧底", state["undercover_word"])
        else:
            role_assignment[agent] = ("平民", state["civilian_word"])

    print("\n角色分配完成：")
    user_role, user_word = role_assignment["user"]
    print(f"\n🎭 你的角色是【{user_role}】，你拿到的词语是：【{user_word}】")
    print("（其他玩家角色保密）\n")

    return {
        "role_assignment": role_assignment
    }

# 节点3:发言生成模块
def generate_speeches(state: GameState) -> GameState:
    speeches = {}
    current_round = state["round"]
    history_speeches = state["history_speeches"]

    # 格式化历史发言（多轮记忆核心：让智能体参考前轮发言）
    history_context = ""
    if state["history_speeches"]:
        history_context += "【历史发言记录】\n"
        for idx, round_speeches in enumerate(state["history_speeches"],1):
            history_context += f"第{idx}轮发言：\n"
            for agent,speech in round_speeches.items():
                if agent not in state["eliminated"]:
                    history_context += f"- {agent}：{speech}\n"
        history_context += "\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是「谁是卧底」游戏的资深玩家，当前是第{current_round}轮发言，需结合历史发言制定策略。
    【核心规则】
    1. 发言要求：
       - 字数：必须严格控制在10-100个汉字（不含标点），无需截断，直接生成符合长度的完整内容
       - 内容：描述词语特征，但绝对不能直接说出词语；结合历史发言调整策略，避免重复自己/他人的描述
       - 风格：自然口语化，句子完整通顺，逻辑清晰
       - 完整性：确保发言是完整的句子，语义完整不截断
    2. 角色策略：
       - 平民：描述核心特征，帮助其他平民识别卧底；避免重复前轮发言，找出发言矛盾的玩家
       - 卧底：模仿平民的描述风格，模糊核心差异；避免与前轮自己的发言矛盾，同时不暴露身份
    3. 输出格式：必须严格按照JSON格式输出，示例：
       {{{{"speech": "这是一种日常饮用的饮品，有多种口味可选，不同品牌的口感差异不大，平时在家或外出都经常能喝到"}}}}
    禁止输出任何额外文字，只返回JSON字符串！
    {history_context}"""),
        ("user", "你的角色是{role}，拿到的词语是{word}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    print(f"\n第{current_round}轮发言阶段（建议发言长度：10-100字）：")
    for agent,(role,word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue

        # 当用户发言时，不需要LLM生成
        if agent == "user":
            print(f"\n【你的回合】你的角色：{role}，你的词语：{word}")
            speech = input("请输入你的发言：").strip()
            if not speech:
                speech = "这是一种很常见的东西，大家应该都接触过。"
            speeches[agent] = speech
            print(f" 你的发言：{speech}")
            continue

        # 其他Agent需要LLM生成发言内容
        output = chain.invoke({
            "role": role,
            "word": word,
            "history_context": history_context,
            "current_round": current_round,
        })

        try:
            speech = output["speech"]
            if len(speech) > 100:
                print(f"{agent}发言超过100字（实际{len(speech)}字）,内容完整保留")
            elif len(speech) < 10:
                print(f"{agent}发言不足10字（实际{len(speech)}字），内容完整保留")
                # 兜底补充
                if role == "平民":
                    speech = f"{speech},是日常生活中很常见的物品，使用场景非常广泛，几乎每个人都接触过"
                else:
                    speech = f"{speech},大家在生活中经常能见到或用到，不同场景下的用法基本一致，不容易区分"
                print(f"{agent}发言补充后：{speech}(长度{len(speech)}字)")
        except KeyError:
            if role == "平民":
                speech = f"第{current_round}轮发言：这是日常能用到的东西，使用频率很高，不同品牌的款式略有差异，但核心功能是一样的，几乎每个家庭都有这类物品，是生活中不可或缺的常用品"
            else:
                speech = f"第{current_round}轮发言：这是大家都熟悉的物品，平时使用场景很多，外观和功能都比较相似，很难快速区分不同类型，生活中随处可见，几乎每个人都使用过这类物品"

        speeches[agent] = speech
        print(f"\n>>> {agent}")
        print(f" 发言：{speech}")

    history_speeches.append(speeches.copy())
    return {
        "speeches": speeches,
        "history_speeches": history_speeches
    }

# 节点4:投票模块
def vote_undercover(state: GameState) -> GameState:
    votes = {}
    reasons = {}
    current_agents = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    current_round = state["round"]

    # 格式化发言上下文
    speech_context = f"【第{current_round}轮发言】\n"
    speech_context += "\n".join([f"{agent}:{speech}" for agent,speech in state["speeches"].items()])

    if state["history_speeches"]:
        speech_context += "\n\n【历史发言参考】\n"
        for idx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{idx}轮：\n"
            for agent,speech in round_speeches.items():
                if agent in current_agents:
                    speech_context += f"- {agent}:{speech}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是「谁是卧底」游戏的理性玩家，需基于当前轮+历史发言分析并投票。
    【分析规则】
    1. 投票依据：
       - 对比玩家当前轮和历史发言，找出矛盾/异常的描述（卧底常出现前后矛盾）
       - 平民：重点关注发言前后不一致、描述偏离词语特征的玩家
       - 卧底：找出看起来像平民的玩家投票，避免自己被怀疑，保持投票理由连贯
    2. 输出格式：必须严格按照JSON格式输出，示例：
       {{{{"vote": "agent2", "reason": "agent2本轮和上轮发言矛盾，描述不符合平民词特征"}}}}
    禁止输出任何额外文字，只返回JSON字符串！
    {speech_context}"""),
        ("user", """你的角色：{role}
    你的词语：{word}
    请选择你要投票的玩家并说明理由（理由控制在50字内）""")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    print(f"\n第{current_round}轮投票阶段：")
    for agent,(role,word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue

        if agent == "user":
            print(f"\n【你的投票回合】当前存活玩家：{[a for a in current_agents if a != 'user']}")
            while True:
                vote = input("请输入你要投票淘汰的玩家名：").strip()
                if vote in current_agents and vote != "user":
                    break
                print(f"输入无效，请从以下玩家中选择：{[a for a in current_agents if a != 'user']}")
            reason = input("请输入你的投票理由（可直接回车跳过）：").strip() or "用户未填写理由"
            votes[agent] = vote
            reasons[agent] = reason
            print(f"  你投票给：{vote}，理由：{reason}")
            continue
            
        output = chain.invoke({
            "role": role,
            "word": word,
            "speech_context": speech_context
        })

        try:
            vote = output["vote"]
            reason = output["reason"]
        except KeyError:
            vote = random.choice([a for a in current_agents if a != agent])
            reason = textwrap.shorten(
                f"第{current_round}轮无有效分析，基于随机策略投票",
                width=50,
            )

        # 校验投票有效性
        if vote == agent or vote not in current_agents:
            vote = random.choice([a for a in current_agents if a != agent])

        votes[agent] = vote
        reasons[agent] = reason
        print(f"\n>>> {agent}：")
        print(f"  投票给：{vote}")
        print(f"  理由：{reason}")

    # 将本轮投票结果追加到历史发言的最后一轮记录中
    history_speeches = list(state["history_speeches"])
    if history_speeches:
        last_round = dict(history_speeches[-1])
        for agent, vote in votes.items():
            last_round[f"{agent}_vote"] = f"投票给{vote}，理由：{reasons[agent]}" # 在当前轮发言记录中，追加了投票字段来保存每个agent到投票记录和投票理由
        history_speeches[-1] = last_round

    return {
        "votes": votes,
        "vote_reasoning": reasons,
        "history_speeches": history_speeches,
    }

# 节点5:胜负判断模块
def judge_result(state: GameState) -> GameState:
    vote_count = {}
    eliminated = list(state["eliminated"])
    current_round = state["round"]
    round = current_round
    winner = state["winner"]  # 默认继承当前winner，游戏未结束时保持原值
    game_status = "running"
    for v in state["votes"].values():
        vote_count[v] = vote_count.get(v, 0) + 1

    max_vote = max(vote_count.values()) # 最多票数
    voted_agent = [a for a, c in vote_count.items() if c == max_vote] # 所有得票最多的
    if len(voted_agent) == 1:
        eliminated_agent = voted_agent[0]
        eliminated.append(eliminated_agent)
        role = state["role_assignment"][eliminated_agent][0]
        print(f"\n第{current_round}轮淘汰结果：{eliminated_agent}({role})")

    remaining = [a for a in state["role_assignment"] if a not in eliminated]
    civ = sum(1 for a in remaining if state["role_assignment"][a][0] == "平民")
    uc = sum(1 for a in remaining if state["role_assignment"][a][0] == "卧底")

    if civ > uc and len(voted_agent) > 1:
        print(f"\n第{current_round}轮存在平票，本轮无人出局")
        round = state["round"] + 1
        print(f"游戏继续，进入第{round}轮")
    elif len(voted_agent) == 1 and state["role_assignment"][voted_agent[0]][0] == "卧底":
        game_status = "end"
        winner = "civilian"
        print("平民胜利！")
    elif civ == 1 and uc == 1:
        game_status = "end"
        winner = "undercover"
        print("卧底胜利！")
    else:
        round = state["round"] + 1
        print(f"游戏继续，进入第{round}轮")

    return {
        "eliminated": eliminated,
        "game_status": game_status,
        "winner": winner,
        "round": round,
    }

# 节点6:结果展示模块
def show_final_result(state: GameState) -> GameState:
    print("\n" + "=" * 50)
    print("游戏结束：")
    print(f"胜利方：{'平民' if state["winner"] == 'civilian' else '卧底'}")
    print(f"平民词：{state["civilian_word"]} | 卧底词：{state["undercover_word"]}")
    print(f"总轮次：{state["round"]}")
    print(f"淘汰顺序：{state['eliminated']}")
    print("=" * 50)
    return state

# 构建图
def build_game_graph():
    graph = StateGraph(GameState)
    graph.add_node("generate_words",generate_words)
    graph.add_node("assign_roles",assign_roles)
    graph.add_node("generate_speeches",generate_speeches)
    graph.add_node("vote_undercover",vote_undercover)
    graph.add_node("judge_result",judge_result)
    graph.add_node("show_final_result",show_final_result)

    graph.set_entry_point("generate_words")
    graph.add_edge("generate_words","assign_roles")
    graph.add_edge("assign_roles","generate_speeches")
    graph.add_edge("generate_speeches","vote_undercover")
    graph.add_edge("vote_undercover","judge_result")

    def route(state: GameState):
        return "generate_speeches" if state["game_status"] == "running" else "show_final_result"

    graph.add_conditional_edges("judge_result",route)
    graph.add_edge("show_final_result",END)

    return graph

if __name__ == "__main__":
    graph = build_game_graph()

    init_state = init_game_state()
    checkpointer = MemorySaver()
    config = {
        "configurable":{
            "thread_id": "spy_001"
        }
    }

    app = graph.compile(checkpointer=checkpointer)
    result = app.invoke(init_state,config=config)