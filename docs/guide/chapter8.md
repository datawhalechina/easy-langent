# 第八章 综合实践：构建“谁是卧底”游戏智能体

## 前言

经过前面的学习，我们已经掌握了LangChain、LangGraph 构建智能体运行逻辑，以及智能体的调试与优化方法。本章将通过“谁是卧底”游戏智能体引擎的综合实战，帮大家串联所有知识点，实现从“理论学习”到“工程实践”的落地。

本章核心目标：

- 理解“谁是卧底”游戏的核心逻辑，拆解智能体引擎的核心模块；
- 熟练运用LangChain + LangGraph 框架，搭建可运行的游戏智能体；
- 掌握实战中常见问题的调试方法，能独立完成智能体的基础优化；
- 完成实战作品提交，掌握项目文档（Readme）的撰写规范。

> 说明：本章实战提供两种实现路径，适配不同基础的同学——基础薄弱的同学可直接复制本章提供的“案例代码”，完成配置后即可运行；基础较好的同学可结合前面知识，自主设计完成个性化创作。

## 8.1项目需求拆解：“谁是卧底”游戏智能体引擎

### 8.1.1 游戏规则简化

为了降低实战难度，我们简化“谁是卧底”游戏规则，核心逻辑如下（所有智能体将遵循此规则运行）：

**视角1：上帝视角（默认适配实战开发，用户旁观）**核心设定：用户作为上帝/旁观者，不参与游戏操作，所有游戏参与方均为智能体，全程由智能体自主完成游戏流程，用户仅查看游戏输出结果。

1. 游戏参与方：4个智能体（1个卧底，3个平民）；
2. 词语分配：平民获得相同的“平民词”，卧底获得与平民词相关但不同的“卧底词”（如平民词“奶茶”，卧底词“果汁”）；
3. 游戏流程：发言阶段：4个智能体依次发言，描述自己拿到的词语（不能直接说词语本身）；
4. 投票阶段：每个智能体根据所有发言，投票选出自己怀疑的“卧底”；
5. 胜负判断：得票最多的智能体被淘汰，若淘汰的是卧底，平民获胜；若淘汰的是平民，游戏继续（重复发言-投票流程）；若剩余1个平民和1个卧底，卧底获胜。

**视角2：玩家视角（可选优化，用户参与）**核心设定：用户作为一名玩家，参与到游戏中，与智能体共同进行“谁是卧底”游戏，用户需自主完成发言、投票操作，智能体作为其他玩家配合完成游戏。

1. 游戏参与方：1名用户（玩家）+ 3个智能体（共4人，1个卧底，3个平民，用户随机分配角色）；
2. 词语分配：系统随机生成平民词和卧底词，用户将收到自己的角色（平民/卧底）和对应词语（仅自己可见），2个智能体分别分配剩余角色和词语；
3. 游戏流程：发言阶段：用户与3个智能体依次发言，用户自主输入符合规则的发言（不能直接说词语本身），智能体自动生成发言；
4. 投票阶段：用户根据所有发言，自主选择怀疑的卧底并投票，3个智能体根据发言自动完成投票；
5. 胜负判断：与上帝视角一致——得票最多的参与者被淘汰，淘汰卧底则平民获胜，淘汰平民则游戏继续，剩余1人平民1人卧底则卧底获胜（用户若被淘汰，可旁观剩余流程）。

说明：本章基础实战默认适配「上帝视角」，无需修改代码即可运行；基础较好的同学可基于玩家视角优化代码，添加用户输入交互逻辑（如接收用户发言、投票输入）。

### 8.1.2 智能体引擎核心模块拆解

要实现上述游戏流程，智能体引擎需要包含哪些核心模块？

核心模块：

1. 词语生成模块：随机生成1组平民词和对应的卧底词（确保相关性，如“手机-电话”“米饭-面条”）；
2. 角色分配模块：将词语分配给3个智能体，随机指定1个为卧底，2个为平民；
3. 发言生成模块：智能体根据自己的角色（平民/卧底）和拿到的词语，生成符合规则的发言；
4. 投票模块：智能体根据所有发言，分析并投票选出怀疑的卧底；
5. 游戏控制模块：串联所有模块，控制游戏流程（发言→投票→胜负判断），记录游戏状态；
6. 结果展示模块：输出每一轮的发言、投票结果，以及最终的游戏胜负。

### 8.1.3 API密钥配置

本章实战需要调用LLM（请按照以下步骤配置API密钥：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=500
)
```

> 说明：.env文件的作用是隐藏敏感信息，避免直接将API密钥写在代码中（后续提交作品时，务必确保.env文件不提交，Readme中需说明配置方法）。

## 8.2 实战开发：逐步构建智能体引擎

本节将分步骤实现所有模块，每一步提供“教学引导+可复制代码+代码说明”，基础薄弱的同学可直接复制代码，替换API密钥后即可运行；基础较好的同学可修改提示词模板、优化投票逻辑等。

### 8.2.1 定义游戏状态

状态（State）用于记录游戏的所有信息（如角色分配、发言记录、投票记录等），后续所有节点都会读取/修改这个状态。我们使用TypedDict定义状态结构，清晰明了。

```python
class GameState(TypedDict):
    """
    游戏状态字典，存储整个游戏的所有关键数据
    TypedDict：提供类型提示，避免键名错误
    """
    civilian_word: str  # 平民词语
    undercover_word: str  # 卧底词语
    role_assignment: dict  # 角色分配：{agent1: ("平民"/"卧底", 词语), ...}
    speeches: dict  # 当前轮发言：{agent1: "发言内容", ...}
    history_speeches: List[Dict[str, str]]  # 历史发言列表：[第1轮发言, 第2轮发言, ...]
    speech_reasoning: dict  # 发言策略理由：{agent1: "理由", ...}
    votes: dict  # 当前轮投票：{agent1: "投给agent2", ...}
    vote_reasoning: dict  # 投票理由：{agent1: "理由", ...}
    game_status: str  # 游戏状态：running（进行中）/end（结束）
    winner: str  # 获胜方：civilian（平民）/undercover（卧底）
    eliminated: List[str]  # 被淘汰的玩家列表
    round: int  # 当前游戏轮次

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
```

### 8.2.2 实现核心模块

每个核心模块对应一个“节点函数”，节点函数接收当前游戏状态，执行对应逻辑，返回修改后的游戏状态。我们依次实现6个核心模块，每个模块都提供提示词模板。

#### 8.2.3.1 节点1：词语生成模块

```python
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
    chain = prompt | llm | parser
    result = chain.invoke({})

    try:
        word_data = json.loads(result.strip())
        civilian_word = word_data["civilian"]
        undercover_word = word_data["undercover"]
    except (json.JSONDecodeError, KeyError):
        fallback_pairs = [
            ("奶茶", "果汁"), ("牙刷", "牙膏"), ("米饭", "面条"),
            ("手机", "平板"), ("篮球", "足球"), ("咖啡", "红茶")
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)

    state["civilian_word"] = civilian_word
    state["undercover_word"] = undercover_word
    print(f"\n🎯 词语生成完成：平民词={civilian_word} ｜ 卧底词={undercover_word}")
    return state
```

在这个节点中我们增加了容错，如果模型调用失败，则从备用库随机生成

#### 8.2.3.2 节点2：角色分配模块

```python
def assign_roles(state: GameState) -> GameState:
    agents = ["agent1", "agent2", "agent3", "agent4"]
    undercover = random.choice(agents)
    for agent in agents:
        if agent == undercover:
            state["role_assignment"][agent] = ("卧底", state["undercover_word"])
        else:
            state["role_assignment"][agent] = ("平民", state["civilian_word"])

    print("\n🎭 角色分配完成：")
    for a, (r, w) in state["role_assignment"].items():
        print(f"  {a}：{r}（词语：{w}）")
    return state
```

这里就非常简单了，随机分配角色即可

#### 8.2.3.3 模块3：发言生成模块

```python
def generate_speeches(state: GameState) -> GameState:
    """
    节点3：生成智能体发言（发言/策略均不截断，仅Prompt引导10-100字）
    核心逻辑：
    1. 结合历史发言制定本轮发言策略（避免重复/矛盾）
    2. Prompt层面引导发言长度10-100字，不做强制截断
    3. 不同角色（平民/卧底）采用差异化发言策略
    4. 发言和策略理由完全保留原始内容，不做任何截断处理
    """
    speeches = {}
    reasoning = {}
    current_round = state["round"]
    
    # 格式化历史发言（多轮记忆核心：让智能体参考前轮发言）
    history_context = ""
    if state["history_speeches"]:
        history_context = "【历史发言记录】\n"
        for idx, round_speeches in enumerate(state["history_speeches"], 1):
            history_context += f"第{idx}轮发言：\n"
            for agent, speech in round_speeches.items():
                if agent not in state["eliminated"]:
                    history_context += f"- {agent}：{speech}\n"
        history_context += "\n"

    # 强化Prompt字数引导（不做后续截断，全靠LLM遵守）
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是「谁是卧底」游戏的资深玩家，当前是第{current_round}轮发言，需结合历史发言制定策略。
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
   {{{{"speech": "这是一种日常饮用的饮品，有多种口味可选，不同品牌的口感差异不大，平时在家或外出都经常能喝到", "reason": "作为平民，详细描述饮品特征，避免重复前轮发言，帮助其他平民识别卧底"}}}}
禁止输出任何额外文字，只返回JSON字符串！
{history_context}"""),
        ("user", "你的角色是{role}，拿到的词语是{word}")
    ])
    chain = prompt | llm | parser

    print(f"\n🗣 第{current_round}轮发言阶段（建议发言长度：10-100字）：")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue
        # 调用LLM生成符合角色策略的发言
        output = chain.invoke({"role": role, "word": word})
        
        try:
            # 解析LLM输出的JSON格式数据
            speech_data = json.loads(output.strip())
            raw_speech = speech_data["speech"]
            raw_reason = speech_data["reason"]
            
            # 核心修改1：移除发言截断，仅保留长度提示（不修改内容）
            speech = raw_speech
            # 长度提示（友好提醒，不强制修改）
            if len(speech) > 100:
                print(f"⚠️  {agent}（{role}）发言超过100字（实际{len(speech)}字），内容完整保留")
            elif len(speech) < 10:
                print(f"⚠️  {agent}（{role}）发言不足10字（实际{len(speech)}字），内容完整保留")
                
            # 兜底补充逻辑：仅补充内容，不截断（若仍需补充）
            if len(speech) < 10:
                if role == "平民":
                    speech = f"{speech}，是日常生活中很常见的物品，使用场景非常广泛，几乎每个人都接触过"
                else:
                    speech = f"{speech}，大家在生活中经常能见到或用到，不同场景下的用法基本一致，不容易区分"
                print(f"🔧 {agent}（{role}）发言补充后：{speech}（长度{len(speech)}字）")
                
        except (json.JSONDecodeError, KeyError):
            # LLM输出解析失败时的兜底发言（完整内容，不截断）
            if role == "平民":
                speech = f"第{current_round}轮发言：这是日常能用到的东西，使用频率很高，不同品牌的款式略有差异，但核心功能是一样的，几乎每个家庭都有这类物品，是生活中不可或缺的常用品"
                raw_reason = f"平民兜底发言，第{current_round}轮避免重复前轮，完整描述物品核心特征，不做截断处理"
            else:
                speech = f"第{current_round}轮发言：这是大家都熟悉的物品，平时使用场景很多，外观和功能都比较相似，很难快速区分不同类型，生活中随处可见，几乎每个人都使用过这类物品"
                raw_reason = f"卧底兜底发言，第{current_round}轮伪装平民，完整模糊描述特征避免暴露身份，不截断"
        
        reason = raw_reason

        # 保存当前智能体的发言和策略理由（完整内容）
        speeches[agent] = speech
        reasoning[agent] = reason
        # 打印发言结果（清晰展示角色和完整内容）
        print(f"\n{agent}（{role}）")
        print(f"  发言：{speech}")
        print(f"  策略：{reason}")

    # 将本轮发言存入历史（完整内容，供下一轮参考）
    state["history_speeches"].append(speeches.copy())
    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    return state
```

发言生成的核心是提示词模板，我们明确要求发言的长度、风格，以及平民和卧底的发言差异（平民真实描述，卧底伪装）；同时排除已淘汰的智能体，确保多轮游戏的合理性。

#### 8.2.3.4 模块4：投票模块

```python
def vote_undercover(state: GameState) -> GameState:
    votes = {}
    reasons = {}
    current_agents = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    current_round = state["round"]
    
    # 格式化发言上下文
    speech_context = f"【第{current_round}轮发言】\n"
    speech_context += "\n".join([f"{agent}：{speech}" for agent, speech in state["speeches"].items()])
    
    if state["history_speeches"]:
        speech_context += "\n\n【历史发言参考】\n"
        for idx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{idx}轮：\n"
            for agent, speech in round_speeches.items():
                if agent in current_agents:
                    speech_context += f"- {agent}：{speech}\n"

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
    chain = prompt | llm | parser

    print(f"\n🗳 第{current_round}轮投票阶段：")
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
            raw_reason = vote_data["reason"]
            
            reason = raw_reason
            
        except (json.JSONDecodeError, KeyError):
            vote = random.choice([a for a in current_agents if a != agent])
            reason = textwrap.shorten(
                f"第{current_round}轮无有效分析，基于随机策略投票",
                width=50
            )
        
        # 校验投票有效性
        if vote == agent or vote not in current_agents:
            vote = random.choice([a for a in current_agents if a != agent])
        
        votes[agent] = vote
        reasons[agent] = reason
        print(f"\n{agent}（{role}）")
        print(f"  投票给：{vote}")
        print(f"  理由：{reason}")

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state
```

投票模块是核心难点，我们添加了多重兜底逻辑——避免投自己、避免投淘汰的玩家、解析失败时随机投票，确保程序稳定运行；同时引导LLM根据发言分析投票，体现智能体的“决策”能力。

#### 8.2.3.5 模块5：胜负判断模块

```python
def judge_result(state: GameState) -> GameState:
    vote_count = {}
    for v in state["votes"].values():
        vote_count[v] = vote_count.get(v, 0) + 1
    max_vote = max(vote_count.values())
    eliminated = random.choice([a for a, c in vote_count.items() if c == max_vote])
    state["eliminated"].append(eliminated)
    role = state["role_assignment"][eliminated][0]
    current_round = state["round"]
    
    print(f"\n❌ 第{current_round}轮淘汰结果：{eliminated}（{role}）")

    remaining = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    civ = sum(1 for a in remaining if state["role_assignment"][a][0] == "平民")
    uc = sum(1 for a in remaining if state["role_assignment"][a][0] == "卧底")

    if role == "卧底":
        state["game_status"] = "end"
        state["winner"] = "civilian"
        print("🎉 平民胜利！")
    elif civ == 1 and uc == 1:
        state["game_status"] = "end"
        state["winner"] = "undercover"
        print("🎉 卧底胜利！")
    else:
        state["game_status"] = "running"
        state["round"] += 1
        print(f"➡ 游戏继续，进入第{state['round']}轮")
    return state
```

#### 8.2.3.6 模块6：结果展示模块

```python
def show_final_result(state: GameState) -> GameState:
    print("\n" + "="*50)
    print("📜 游戏结束 · 总结")
    print(f"胜利方：{'平民' if state['winner'] == 'civilian' else '卧底'}")
    print(f"平民词：{state['civilian_word']} | 卧底词：{state['undercover_word']}")
    print(f"总轮次：{state['round']}")
    print(f"淘汰顺序：{state['eliminated']}")
    print("="*50)
    return state
```

### 8.2.3 构建LangGraph图结构

```python
def build_game_graph():
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

    def route(state: GameState):
        return "generate_speeches" if state["game_status"] == "running" else "show_final_result"
    graph.add_conditional_edges("judge_result", route)
    graph.add_edge("show_final_result", END)
    return graph
```

图结构的核心是“节点+边”，我们通过add_node添加所有模块，通过add_edge定义固定跳转，通过add_conditional_edges定义条件跳转（游戏继续/结束），set_entry_point定义游戏入口，完美串联整个游戏流程。

## 8.3 完整运行代码

以下是完整的可运行代码，基础薄弱的同学可直接复制到Python文件中（命名为who_is_undercover.py），确保.env文件配置正确（API密钥），运行后即可看到完整的游戏流程。

```python
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
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()

# ================== 1. 定义游戏状态 ==================
class GameState(TypedDict):
    """
    游戏状态字典，存储整个游戏的所有关键数据
    TypedDict：提供类型提示，避免键名错误
    """
    civilian_word: str  # 平民词语
    undercover_word: str  # 卧底词语
    role_assignment: dict  # 角色分配：{agent1: ("平民"/"卧底", 词语), ...}
    speeches: dict  # 当前轮发言：{agent1: "发言内容", ...}
    history_speeches: List[Dict[str, str]]  # 历史发言列表：[第1轮发言, 第2轮发言, ...]
    speech_reasoning: dict  # 发言策略理由：{agent1: "理由", ...}
    votes: dict  # 当前轮投票：{agent1: "投给agent2", ...}
    vote_reasoning: dict  # 投票理由：{agent1: "理由", ...}
    game_status: str  # 游戏状态：running（进行中）/end（结束）
    winner: str  # 获胜方：civilian（平民）/undercover（卧底）
    eliminated: List[str]  # 被淘汰的玩家列表
    round: int  # 当前游戏轮次

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
        ("system", """你是专业的「谁是卧底」游戏出题人，需生成一组高质量的词语对。
核心要求：
1. 词语类型：日常物品/食品/场景（如：奶茶-果汁、牙刷-牙膏），避免生僻词
2. 语义关系：平民词与卧底词高度相似但核心特征不同，有足够博弈空间
3. 难度适配：适合4人游戏，既不轻易暴露也能通过描述区分
4. 输出格式：必须严格按照 JSON 格式输出，示例：{{"civilian": "奶茶", "undercover": "果汁"}}
禁止输出任何额外文字，只返回JSON字符串！"""),
        ("user", "生成一组符合要求的谁是卧底词语对")
    ])
    chain = prompt | llm | parser
    result = chain.invoke({})

    try:
        word_data = json.loads(result.strip())
        civilian_word = word_data["civilian"]
        undercover_word = word_data["undercover"]
    except (json.JSONDecodeError, KeyError):
        fallback_pairs = [
            ("奶茶", "果汁"), ("牙刷", "牙膏"), ("米饭", "面条"),
            ("手机", "平板"), ("篮球", "足球"), ("咖啡", "红茶")
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)

    state["civilian_word"] = civilian_word
    state["undercover_word"] = undercover_word
    print(f"\n🎯 词语生成完成：平民词={civilian_word} ｜ 卧底词={undercover_word}")
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

    print("\n🎭 角色分配完成：")
    for a, (r, w) in state["role_assignment"].items():
        print(f"  {a}：{r}（词语：{w}）")
    return state

# ---- 节点3：发言----
def generate_speeches(state: GameState) -> GameState:
    """
    节点3：生成智能体发言（发言/策略均不截断，仅Prompt引导10-100字）
    核心逻辑：
    1. 结合历史发言制定本轮发言策略（避免重复/矛盾）
    2. Prompt层面引导发言长度10-100字，不做强制截断
    3. 不同角色（平民/卧底）采用差异化发言策略
    4. 发言和策略理由完全保留原始内容，不做任何截断处理
    """
    speeches = {}
    reasoning = {}
    current_round = state["round"]
    
    # 格式化历史发言（多轮记忆核心：让智能体参考前轮发言）
    history_context = ""
    if state["history_speeches"]:
        history_context = "【历史发言记录】\n"
        for idx, round_speeches in enumerate(state["history_speeches"], 1):
            history_context += f"第{idx}轮发言：\n"
            for agent, speech in round_speeches.items():
                if agent not in state["eliminated"]:
                    history_context += f"- {agent}：{speech}\n"
        history_context += "\n"

    # 强化Prompt字数引导（不做后续截断，全靠LLM遵守）
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是「谁是卧底」游戏的资深玩家，当前是第{current_round}轮发言，需结合历史发言制定策略。
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
   {{{{"speech": "这是一种日常饮用的饮品，有多种口味可选，不同品牌的口感差异不大，平时在家或外出都经常能喝到", "reason": "作为平民，详细描述饮品特征，避免重复前轮发言，帮助其他平民识别卧底"}}}}
禁止输出任何额外文字，只返回JSON字符串！
{history_context}"""),
        ("user", "你的角色是{role}，拿到的词语是{word}")
    ])
    chain = prompt | llm | parser

    print(f"\n🗣 第{current_round}轮发言阶段（建议发言长度：10-100字）：")
    for agent, (role, word) in state["role_assignment"].items():
        if agent in state["eliminated"]:
            continue
        # 调用LLM生成符合角色策略的发言
        output = chain.invoke({"role": role, "word": word})
        
        try:
            # 解析LLM输出的JSON格式数据
            speech_data = json.loads(output.strip())
            raw_speech = speech_data["speech"]
            raw_reason = speech_data["reason"]
            
            # 核心修改1：移除发言截断，仅保留长度提示（不修改内容）
            speech = raw_speech
            # 长度提示（友好提醒，不强制修改）
            if len(speech) > 100:
                print(f"⚠️  {agent}（{role}）发言超过100字（实际{len(speech)}字），内容完整保留")
            elif len(speech) < 10:
                print(f"⚠️  {agent}（{role}）发言不足10字（实际{len(speech)}字），内容完整保留")
                
            # 兜底补充逻辑：仅补充内容，不截断（若仍需补充）
            if len(speech) < 10:
                if role == "平民":
                    speech = f"{speech}，是日常生活中很常见的物品，使用场景非常广泛，几乎每个人都接触过"
                else:
                    speech = f"{speech}，大家在生活中经常能见到或用到，不同场景下的用法基本一致，不容易区分"
                print(f"🔧 {agent}（{role}）发言补充后：{speech}（长度{len(speech)}字）")
                
        except (json.JSONDecodeError, KeyError):
            # LLM输出解析失败时的兜底发言（完整内容，不截断）
            if role == "平民":
                speech = f"第{current_round}轮发言：这是日常能用到的东西，使用频率很高，不同品牌的款式略有差异，但核心功能是一样的，几乎每个家庭都有这类物品，是生活中不可或缺的常用品"
                raw_reason = f"平民兜底发言，第{current_round}轮避免重复前轮，完整描述物品核心特征，不做截断处理"
            else:
                speech = f"第{current_round}轮发言：这是大家都熟悉的物品，平时使用场景很多，外观和功能都比较相似，很难快速区分不同类型，生活中随处可见，几乎每个人都使用过这类物品"
                raw_reason = f"卧底兜底发言，第{current_round}轮伪装平民，完整模糊描述特征避免暴露身份，不截断"
        
        reason = raw_reason

        # 保存当前智能体的发言和策略理由（完整内容）
        speeches[agent] = speech
        reasoning[agent] = reason
        # 打印发言结果（清晰展示角色和完整内容）
        print(f"\n{agent}（{role}）")
        print(f"  发言：{speech}")
        print(f"  策略：{reason}")

    # 将本轮发言存入历史（完整内容，供下一轮参考）
    state["history_speeches"].append(speeches.copy())
    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    return state

def vote_undercover(state: GameState) -> GameState:
    votes = {}
    reasons = {}
    current_agents = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    current_round = state["round"]
    
    # 格式化发言上下文
    speech_context = f"【第{current_round}轮发言】\n"
    speech_context += "\n".join([f"{agent}：{speech}" for agent, speech in state["speeches"].items()])
    
    if state["history_speeches"]:
        speech_context += "\n\n【历史发言参考】\n"
        for idx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{idx}轮：\n"
            for agent, speech in round_speeches.items():
                if agent in current_agents:
                    speech_context += f"- {agent}：{speech}\n"

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
    chain = prompt | llm | parser

    print(f"\n🗳 第{current_round}轮投票阶段：")
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
            raw_reason = vote_data["reason"]
            
            reason = raw_reason
            
        except (json.JSONDecodeError, KeyError):
            vote = random.choice([a for a in current_agents if a != agent])
            reason = textwrap.shorten(
                f"第{current_round}轮无有效分析，基于随机策略投票",
                width=50
            )
        
        # 校验投票有效性
        if vote == agent or vote not in current_agents:
            vote = random.choice([a for a in current_agents if a != agent])
        
        votes[agent] = vote
        reasons[agent] = reason
        print(f"\n{agent}（{role}）")
        print(f"  投票给：{vote}")
        print(f"  理由：{reason}")

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state

# ---- 节点5：裁决 ----
def judge_result(state: GameState) -> GameState:
    vote_count = {}
    for v in state["votes"].values():
        vote_count[v] = vote_count.get(v, 0) + 1
    max_vote = max(vote_count.values())
    eliminated = random.choice([a for a, c in vote_count.items() if c == max_vote])
    state["eliminated"].append(eliminated)
    role = state["role_assignment"][eliminated][0]
    current_round = state["round"]
    
    print(f"\n❌ 第{current_round}轮淘汰结果：{eliminated}（{role}）")

    remaining = [a for a in state["role_assignment"] if a not in state["eliminated"]]
    civ = sum(1 for a in remaining if state["role_assignment"][a][0] == "平民")
    uc = sum(1 for a in remaining if state["role_assignment"][a][0] == "卧底")

    if role == "卧底":
        state["game_status"] = "end"
        state["winner"] = "civilian"
        print("🎉 平民胜利！")
    elif civ == 1 and uc == 1:
        state["game_status"] = "end"
        state["winner"] = "undercover"
        print("🎉 卧底胜利！")
    else:
        state["game_status"] = "running"
        state["round"] += 1
        print(f"➡ 游戏继续，进入第{state['round']}轮")
    return state

# ---- 节点6：总结 ----
def show_final_result(state: GameState) -> GameState:
    print("\n" + "="*50)
    print("📜 游戏结束 · 总结")
    print(f"胜利方：{'平民' if state['winner'] == 'civilian' else '卧底'}")
    print(f"平民词：{state['civilian_word']} | 卧底词：{state['undercover_word']}")
    print(f"总轮次：{state['round']}")
    print(f"淘汰顺序：{state['eliminated']}")
    print("="*50)
    return state

# ================== 3. 构建 LangGraph ==================
def build_game_graph():
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

    def route(state: GameState):
        return "generate_speeches" if state["game_status"] == "running" else "show_final_result"
    graph.add_conditional_edges("judge_result", route)
    graph.add_edge("show_final_result", END)
    return graph

# ================== 4. 入口 ==================
if __name__ == "__main__":
    game_graph = build_game_graph()
    game = game_graph.compile()
    print("="*50)
    print("🎮 谁是卧底 · 多智能体多轮策略版 启动")
    print("="*50)
    game.invoke(init_game_state())
```

运行结果

```
🎯 词语生成完成：平民词=牙刷 ｜ 卧底词=牙膏

🎭 角色分配完成：
  agent1：平民（词语：牙刷）
  agent2：平民（词语：牙刷）
  agent3：平民（词语：牙刷）
  agent4：卧底（词语：牙膏）

🗣 第1轮发言阶段（建议发言长度：10-100字）：

agent1（平民）
  发言：这是一种日常清洁用品，通常早晚使用，有多种刷毛软硬可选，能有效保持口腔卫生。
  策略：作为平民，直接描述牙刷的核心功能和使用场景，避免提及品牌或具体形状，帮助其他平民识别卧底。

agent2（平民）
  发言：这是一种日常清洁工具，通常早晚各用一次，能有效保持口腔卫生，刷毛有软硬之分，需要定期更换
  策略：作为平民，描述牙刷的核心功能和使用场景，避免直接说出词语，帮助其他平民识别卧底

agent3（平民）
  发言：这是一种日常清洁用品，通常早晚使用，有不同的刷毛软硬度，能有效清洁牙齿和口腔卫生。
  策略：作为平民，描述牙刷的核心特征如清洁用途、使用频率和刷毛特点，避免直接说出词语，帮助其他平民识别卧底。

agent4（卧底）
  发言：这是一种日常清洁用品，每天早晚都会用到，有不同的味道和功效，能保持口腔清新健康
  策略：作为卧底，模仿平民描述日常清洁用品的特征，避免直接说出词语，同时保持描述自然，不暴露身份

🗳 第1轮投票阶段：

agent1（平民）
  投票给：agent4
  理由：agent4描述“不同的味道和功效”更接近牙膏特征，与牙刷的核心功能（刷毛、清洁牙齿）存在偏差。

agent2（平民）
  投票给：agent4
  理由：agent4提到'不同的味道和功效'，这与牙刷的典型特征不符，更像是在描述牙膏，发言可疑。

agent3（平民）
  投票给：agent4
  理由：agent4描述'不同的味道和功效'不符合牙刷的核心特征，更像是牙膏的描述，偏离平民词。

agent4（卧底）
  投票给：agent1
  理由：agent4描述“不同的味道和功效”更贴近牙膏，与平民词牙刷的清洁工具特征不符，易暴露平民身份。

❌ 第1轮淘汰结果：agent4（卧底）
🎉 平民胜利！

==================================================
📜 游戏结束 · 总结
胜利方：平民
平民词：牙刷 | 卧底词：牙膏
总轮次：1
淘汰顺序：['agent4']
==================================================
(base) PS C:\Users\xiong\Desktop\iii> 
```

运行代码包含了所有模块，无需修改，只需确保.env文件中的API密钥正确，运行后即可看到完整的游戏流程（词语生成→角色分配→发言→投票→淘汰→结果展示）。基础薄弱的同学可直接使用此代码完成实战。

## 作品提交要求

本章需完成实战作品，并按以下要求提交

### 1.提交路径

在本项目[easy-langent](https://github.com/datawhalechina/easy-langent)中的[project](https://github.com/datawhalechina/easy-langent/tree/main/project)文件夹提交自己的综合实践作品，具体样例可参考`NovelGenerateDemo`

提交时需在 `project` 文件夹内创建个人专属子文件夹，文件夹命名格式使用驼峰命名的规则进行命名，例如`WhoIsTheSpy`，提交使用GitHub PR（Pull Request）进行提交即可。

### 2 提交文件清单

提交文件需齐全、命名规范，具体清单如下：

1. 核心代码文件（必选）：确保代码可直接运行（替换API密钥后无报错）；
2. 项目说明文档（必选）：命名为“Readme.md”，为Markdown格式，核心内容缺一不可，具体包含：  项目简介、核心功能、Python版本、依赖包安装命令，以及代码运行步骤（确保他人可顺利使用）；

> 重要提示：禁止提交.env文件，避免API密钥泄露，无需在文件夹中放置.env文件。
