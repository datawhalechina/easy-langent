#  第七章 LangGraph进阶：多智能体协作与复杂流程管控

## 前言

哈喽，各位自学小伙伴～ 本章咱们正式进入“高阶玩法”——多智能体协作与复杂流程管控！

不同于上一章的基础图结构开发，本章内容会更加工程化一些，核心是教大家用LangGraph ，搭建能“分工协作”“自我纠错”“人机配合”的智能系统。

全程遵循「理论不啰嗦、代码能直接跑、练习有反馈」的自学原则，每个知识点都配套实操案例（复制就能运行），还有趣味类比帮大家理解抽象概念，哪怕是零基础自学，也能一步步吃透。话不多说，开干！

先提前准备依赖（终端执行安装）：

```python
# 安装必要依赖（LangGraph v1.0.0+ 版本）
pip install langgraph
# 注意：LangGraph v1.0.0+ 接口有较大更新，旧版本代码需修改，本章全程适配新版本
```

![7-1](..\src\img\7-1.gif)



## 7.1 多智能体系统（Multi-Agent Systems）核心设计

先问大家一个问题：如果让你用单一LLM写一篇“科技论文+配图说明+查重修改”，会遇到什么问题？—— 大概率是写着写着跑偏、上下文堆太多导致卡顿、改完查重又破坏原文逻辑。

这就是单一LLM的“瓶颈”，而多智能体系统，本质就是“组队干活”：把一个复杂任务，拆给多个“专业智能体”，每个智能体负责一块，再通过规则协调配合，最终完成单一智能体搞不定的事。

### 7.1.1 为什么需要多智能体协作？

> Tips:先搞懂“为什么”，再学“怎么写”

#### 7.1.1.1 解决单一 LLM “长指令疲劳”与上下文污染

咱们用一个直观的对比，感受一下单一LLM和多智能体的区别（建议大家亲手跑一下代码，感受更深刻）：

案例1：单一LLM处理“写短文+纠错+润色”（感受“疲劳感”）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# 构建超长指令（模拟复杂任务）
prompt = ChatPromptTemplate.from_messages([
    ("user", """请完成3件事，按顺序来：
1. 写一篇300字左右、关于“LangGraph多智能体”的短文，语言通俗，适合新手；
2. 检查短文是否有错误（比如LangGraph的接口名称、功能描述），修正错误；
3. 润色短文，让语言更流畅，加入1个新手能理解的类比。""")
])

# 执行单一LLM调用
chain = prompt | llm
result = chain.invoke({})
print("单一LLM输出：")
print(result.content)
```

运行后你会发现：单一LLM大概率会出现“顾此失彼”——比如润色后又出现错误，或者类比生硬，甚至遗漏某一步（这就是“长指令疲劳”）。

如果你举得还不够，可以这样让大模型生成

```
你是一个数据分析专家，请完成以下任务：

1. 生成一份虚拟销售数据表（100行，字段包括日期、产品、地区、销量、收入）；
2. 清洗数据：处理缺失值和异常值；
3. 计算每个地区的总收入和平均销量；
4. 输出一个可视化分析方案（不需要画图，只给代码）；
5. 用通俗语言写一段商业分析报告；
6. 检查你的分析是否有统计学错误并修正；
7. 最后用类比解释给非技术人员听。

所有步骤请一次性完成。
```

即使现在大模型能力不断在增加，但是复杂的长指令一定会降低模型输出的稳定性和准确性

案例2：多智能体处理（分工协作，避免疲劳）

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict, Optional

# 1. 定义全局状态（所有智能体共享的数据，v1.0.0+ 推荐用TypedDict规范状态）
class AgentState(TypedDict):
    content: Optional[str]  # 短文内容
    error: Optional[str]    # 错误信息
    polished_content: Optional[str]  # 润色后内容

# 2. 初始化3个“专业智能体”（分工明确）
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 智能体1：写短文（只负责“写”，不考虑纠错和润色）
writer_prompt = ChatPromptTemplate.from_messages([
    ("user", "写一篇150字左右、关于“LangGraph多智能体”的短文，语言通俗，适合新手，不用纠错和润色。")
])
writer_agent = writer_prompt | llm

# 智能体2：纠错（只负责“找错+改错”，不修改文风）
corrector_prompt = ChatPromptTemplate.from_messages([
    ("user", "请检查以下短文，修正其中关于LangGraph的技术错误（比如接口、功能描述），只输出修正后的内容，不润色：\n{content}")
])
corrector_agent = corrector_prompt | llm

# 智能体3：润色（只负责“优化语言”，不修改核心内容）
polisher_prompt = ChatPromptTemplate.from_messages([
    ("user", "请润色以下短文，加入1个新手能理解的类比，语言更流畅，不改变核心内容和技术准确性：\n{content}")
])
polisher_agent = polisher_prompt | llm

# 3. 定义节点函数（v1.0.0+ 节点需是可调用函数，接收state，返回更新后的state）
def write_node(state: AgentState) -> AgentState:
    result = writer_agent.invoke({})
    return {"content": result.content, "error": None, "polished_content": None}

def correct_node(state: AgentState) -> AgentState:
    result = corrector_agent.invoke({"content": state["content"]})
    return {"content": result.content, "error": None, "polished_content": None}

def polish_node(state: AgentState) -> AgentState:
    result = polisher_agent.invoke({"content": state["content"]})
    return {"content": state["content"], "error": None, "polished_content": result.content}

# 4. 构建图（v1.0.0+ 用StateGraph构建，简化了旧版本的Graph接口）
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("writer", write_node)  # 写短文节点
workflow.add_node("corrector", correct_node)  # 纠错节点
workflow.add_node("polisher", polish_node)  # 润色节点

# 添加边（定义流程顺序：写→纠错→润色→结束）
workflow.add_edge(START, "writer")
workflow.add_edge("writer", "corrector")
workflow.add_edge("corrector", "polisher")
workflow.add_edge("polisher", END)

# 编译图（v1.0.0+ 必须编译后才能运行）
compiled_workflow: CompiledStateGraph = workflow.compile()

# 5. 运行流程
result = compiled_workflow.invoke({})  # 初始状态为空字典
print("多智能体输出（润色后）：")
print(result["polished_content"])
```

运行后对比：多智能体分工明确，每个智能体只做一件事，输出更精准、更稳定——这就是多智能体的核心优势之一：解决单一LLM的“长指令疲劳”。

> 如果对比后你发现，单一的好像和多任务的区别不大，那说明你是用的大模型的底座能力本身就很强，能够对抗一部分 “长指令疲劳”，但这个能力是有上限的，这个时候你可以做更多复杂的任务去对比~

单一LLM处理复杂任务，就像“一个人又做饭、又洗碗、又擦桌子”，忙到出错；多智能体就像“厨师+洗碗工+保洁”，分工协作，效率翻倍、出错减少。

#### 7.1.1.2 模块化开发：分而治之的工程学思想

在实际的企业开发中，最忌讳“一锅粥”代码——比如把所有逻辑写在一个函数里，后续修改、调试起来要疯掉。

多智能体的“模块化”，就是把复杂流程拆成一个个“独立模块”（每个智能体就是一个模块），每个模块可单独开发、测试、修改，互不影响。

比如上面的案例，如果你觉得“纠错不够精准”，只需要修改corrector_agent的prompt或模型，不用动writer和polisher的代码；如果想加一个“查重”功能，直接新增一个“查重智能体”，添加到流程中即可，不用重构整个系统。

**小思考（动手试一下）：**

在上面的多智能体代码中，新增一个“查重智能体”，负责检查润色后的短文是否有重复内容（模拟查重），并修改重复部分，把流程改成：写→纠错→润色→查重→结束。

### 7.1.2 多智能体常见架构模式

多智能体协作不是“乱组队”，有3种最常用的架构模式，每种模式对应不同的场景，咱们结合代码实操，一个个搞懂（重点掌握前2种，生产中最常用）。

#### 7.1.2.1 中心化协作（Supervisor）：基于路由的“主管-员工”模式

核心逻辑：有一个“主管智能体”（Supervisor），负责接收总任务、拆分任务、分配给不同的“员工智能体”，并汇总结果——就像公司里的“项目经理”，不干活，只协调。

适用场景：任务可明确拆分、需要统一协调的场景

实操案例：中心化多智能体（主管分配任务）

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# ================== 初始化环境 ==================
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 状态定义 ==================
class TaskState(TypedDict):
    task: str
    research: Optional[str]
    draft: Optional[str]
    code: Optional[str]
    math: Optional[str]
    next_agent: Optional[str]
    result: Optional[str]
    round_count: int  # Supervisor 执行轮次
    supervisor_thoughts: Optional[str]  # 打印 LLM 思考过程

MAX_ROUNDS = 3

# ================== 员工智能体 ==================
research_agent = ChatPromptTemplate.from_messages([
    ("user", "请调研以下任务的背景信息，整理成条列要点，中文输出：{task}")
]) | llm

writer_agent = ChatPromptTemplate.from_messages([
    ("user", "根据以下信息撰写中文技术文章或说明文：{research}")
]) | llm

code_agent = ChatPromptTemplate.from_messages([
    ("user", "请根据以下任务生成 Python 示例代码：{task}")
]) | llm

math_agent = ChatPromptTemplate.from_messages([
    ("user", "请解决以下数学/逻辑问题，并详细说明过程：{task}")
]) | llm

# ================== 动态 Supervisor 节点 ==================
def supervisor_node(state: TaskState):
    state["round_count"] += 1

    # 超过最大轮次，触发兜底
    if state["round_count"] > MAX_ROUNDS:
        print(f"⚠️ 超过最大轮次 {MAX_ROUNDS}，触发兜底 → 结束任务")
        return {**state, "next_agent": "end", "supervisor_thoughts": "轮次数超过上限，直接结束任务"}

    # 中文提示词，严格约束 LLM
    prompt = f"""
你是多智能体系统的主管智能体（Supervisor），负责调度专家智能体，但你不执行任务。请阅读当前任务和已完成状态，并选择下一步最合适的智能体执行。

任务：
{state['task']}

已完成状态：
- 调研: {"已完成" if state.get("research") else "未完成"}
- 写作: {"已完成" if state.get("draft") else "未完成"}
- 编程: {"已完成" if state.get("code") else "未完成"}
- 数学: {"已完成" if state.get("math") else "未完成"}

可调度智能体：
- research_agent：负责调研和整理资料
- writer_agent：负责撰写中文文章或说明文
- code_agent：负责编写 Python 代码
- math_agent：负责数学/逻辑计算与推理

约束：
1. 不能选择已完成的智能体。
2. 必须选择与任务相关的智能体。
3. 如果所有任务完成，返回 "end"。
4. 请在回答中先写出你的“思考过程”，然后在最后一行返回下一步智能体名称（research_agent / writer_agent / code_agent / math_agent / end）。

请用中文完整回答：
"""

    res = llm.invoke(prompt)
    thoughts = res.content.strip()
    # 取最后一行作为 next_agent
    next_agent = thoughts.splitlines()[-1].strip()
    print(f"🧠 主管思考过程：\n{thoughts}\n")
    print(f"🧠 主管调度 → {next_agent} (轮次 {state['round_count']})")
    return {**state, "next_agent": next_agent, "supervisor_thoughts": thoughts}

# ================== 员工节点 ==================
def research_node(state: TaskState):
    print(">>> Research Agent 执行中...")
    try:
        res = research_agent.invoke({"task": state["task"]})
        result = res.content.strip()
    except Exception as e:
        result = f"调研失败：{str(e)[:50]}"
    return {**state, "research": result, "result": result}

def writer_node(state: TaskState):
    print(">>> Writer Agent 执行中...")
    try:
        res = writer_agent.invoke({"research": state.get("research","")})
        result = res.content.strip()
    except Exception as e:
        result = f"写作失败：{str(e)[:50]}"
    return {**state, "draft": result, "result": result}

def code_node(state: TaskState):
    print(">>> Code Agent 执行中...")
    try:
        res = code_agent.invoke({"task": state["task"]})
        result = res.content.strip()
    except Exception as e:
        result = f"代码生成失败：{str(e)[:50]}"
    return {**state, "code": result, "result": result}

def math_node(state: TaskState):
    print(">>> Math Agent 执行中...")
    try:
        res = math_agent.invoke({"task": state["task"]})
        result = res.content.strip()
    except Exception as e:
        result = f"数学求解失败：{str(e)[:50]}"
    return {**state, "math": result, "result": result}

# ================== 构建 LangGraph ==================
workflow = StateGraph(TaskState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("code_agent", code_node)
workflow.add_node("math_agent", math_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "code_agent": "code_agent",
        "math_agent": "math_agent",
        "end": END
    }
)

workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")
workflow.add_edge("code_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")

app = workflow.compile()

# ================== 运行示例 ==================
if __name__ == "__main__":
    tasks = [
        "撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者",
    ]

    for t in tasks:
        print("\n" + "="*50)
        print(f"任务：{t}")
        init_state = {
            "task": t,
            "research": None,
            "draft": None,
            "code": None,
            "math": None,
            "next_agent": None,
            "result": None,
            "round_count": 0,
            "supervisor_thoughts": None
        }
        result = app.invoke(init_state)
        print("\n✅ 最终结果：\n", result["result"])

```

运行结果

```
==================================================
任务：撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者
🧠 主管思考过程：
**思考过程：**  
当前任务是“撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者”。已完成状态显示所有子任务（调研、写作、编程、数学）均未完成。  
- 首先需要收集和整理关于 LangGraph 多智能体协作的基础资料，确保内容准确且适合初学者，因此**调研**是首要步骤。  
- 调研任务对应智能体 **research_agent**，它负责调研和整理资料，且尚未完成，符合约束条件。  
- 其他智能体（如 writer_agent、code_agent、math_agent）在调研完成前暂不需要调度。  

**下一步智能体：**  
research_agent

🧠 主管调度 → research_agent (轮次 1)
>>> Research Agent 执行中...
🧠 主管思考过程：
**思考过程：**
当前任务是“撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者”。已完成状态中，“调研”已完成，说明资料已整理好，但“写作”“编程”“数学”均未完成。
- 文章撰写是核心任务，且面向初学者，需要清晰的中文表达和逻辑结构，因此下一步应启动“写作”环节。
- “编程”和“数学”在任务中可能涉及代码示例或逻辑说明，但需在文章内容确定后再补充，目前不是最优先的。
- 根据约束，不能选择已完成的“调研”智能体，而“写作”智能体（writer_agent）与任务直接相关，且尚未执行。

**下一步智能体：**
writer_agent

🧠 主管调度 → writer_agent (轮次 2)
>>> Writer Agent 执行中...
🧠 主管思考过程：
**思考过程：**
当前任务是撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者。已完成状态显示“调研”和“写作”已完成，但“编程”和“数学”未完成。
- “编程”未完成：LangGraph 多智能体系统通常涉及代码示例或架构说明，需要编写 Python 代码来演示协作流程。
- “数学”未完成：多智能体协作可能涉及逻辑推理或简单数学建模（如任务分配、状态转换），但本任务以中文文章为主，数学部分并非核心。
根据约束：
1. 不能选择已完成的智能体（research_agent、writer_agent 已排除）。
2. 必须选择与任务相关的智能体：编程（code_agent）与任务直接相关，可补充代码示例；数学（math_agent）相关性较弱，但可能用于逻辑设计。
3. 任务尚未全部完成，不能返回“end”。
综合判断：下一步应优先选择 **code_agent**，为文章补充代码示例，增强实用性。

**下一步智能体：**
code_agent

🧠 主管调度 → code_agent (轮次 3)
>>> Code Agent 执行中...
⚠️ 超过最大轮次 3，触发兜底 → 结束任务

✅ 最终结果：
 我来为您生成一篇介绍 LangGraph 多智能体协作的中文文章，并附上相应的 Python 示例代码。...
```

中心化协作最大的特点就是主管智能体负责对应的调度，这会要求主管的大模型推理能力很强或者是有agent调度训练过的模型，同时由于LLM存在不可控的情况，会存在让主管智能体“钻牛角尖”形成死循环调度，一般会对调度的次数进行约束，例如本节案例中的`⚠️ 超过最大轮次 3，触发兜底 → 结束任务`

#### 7.1.2.2 链式协作（Sequence）：有序的任务接力

核心逻辑：没有主管，多个智能体按“固定顺序”接力完成任务，每个智能体的输出，作为下一个智能体的输入——就像“流水线生产”，上一道工序做完，交给下一道，直到完成。

适用场景：任务流程固定、顺序不可颠倒的场景

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
import os
from dotenv import load_dotenv

# ========== 1. 初始化 LLM ==========
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.2
)

# ========== 2. 状态定义 ==========
class ChainState(TypedDict):
    task: str
    draft: Optional[str]
    corrected: Optional[str]
    polished: Optional[str]

# ========== 3. Agent Prompt ==========
write_agent = ChatPromptTemplate.from_messages([
    ("system", "你是写作智能体，只负责生成初稿，不要解释。"),
    ("user", "{task}")
]) | llm

correct_agent = ChatPromptTemplate.from_messages([
    ("system", "你是纠错智能体，只修正语法、逻辑和错别字，不扩写。"),
    ("user", "{draft}")
]) | llm

polish_agent = ChatPromptTemplate.from_messages([
    ("system", "你是润色智能体，只提升表达质量和专业度，不改变意思。"),
    ("user", "{corrected}")
]) | llm

# ========== 4. Agent Node ==========
def writer_node(state: ChainState):
    print("\n✍️【Writer Agent】生成初稿中...")
    res = write_agent.invoke({"task": state["task"]})
    return {"draft": res.content.strip()}

def correct_node(state: ChainState):
    print("\n🧹【Corrector Agent】纠错中...")
    res = correct_agent.invoke({"draft": state["draft"]})
    return {"corrected": res.content.strip()}

def polish_node(state: ChainState):
    print("\n✨【Polisher Agent】润色中...")
    res = polish_agent.invoke({"corrected": state["corrected"]})
    return {"polished": res.content.strip()}

# ========== 5. 构建链式 LangGraph ==========
workflow = StateGraph(ChainState)

workflow.add_node("writer", writer_node)
workflow.add_node("corrector", correct_node)
workflow.add_node("polisher", polish_node)

# 链式 Pipeline
workflow.add_edge(START, "writer")
workflow.add_edge("writer", "corrector")
workflow.add_edge("corrector", "polisher")
workflow.add_edge("polisher", END)

app = workflow.compile()

# ========== 6. 运行 ==========
if __name__ == "__main__":
    init_state = {
        "task": "撰写一篇150字左右的介绍文，说明LangGraph多智能体的核心优势，适合技术初学者阅读",
        "draft": None,
        "corrected": None,
        "polished": None,
    }

    result = app.invoke(init_state)

    print("\n" + "=" * 90)
    print("📊 链式多智能体 Pipeline 最终结果")
    print("=" * 90)
    print("\n📝 初稿：\n", result["draft"])
    print("\n✅ 纠错：\n", result["corrected"])
    print("\n✨ 润色：\n", result["polished"])
    print("=" * 90)

```

运行结果

```
✍️【Writer Agent】生成初稿中...

🧹【Corrector Agent】纠错中...

✨【Polisher Agent】润色中...

==========================================================================================
📊 链式多智能体 Pipeline 最终结果
==========================================================================================

📝 初稿：
 LangGraph是一个让多个AI智能体协同工作的开发框架。它的核心优势在于**可视化编排**和**稳定协作**。

你可以像搭积木一样，在图形界面上拖拽连接不同的智能体（如分析、写作、检查等模块），直观地构建复杂工作流。更重要的是，它能**可靠地管理协作过程**——智能体们会按照你设定的规则有序“对 话”和传递信息，自动处理各种状态，确保任务一步步清晰、稳定地执行到底。

这让你能轻松组合多个AI能力，构建出比单个智能体更强大、更可靠的自动化应用，而无需深究复杂的底层代码。

✅ 纠错：
 LangGraph是一个让多个AI智能体协同工作的开发框架。它的核心优势在于**可视化编排**和**稳定协作**。

你可以像搭积木一样，在图形界面上拖拽连接不同的智能体（如分析、写作、检查等模块），直观地构建复杂工作流。更重要的是，它能**可靠地管理协作过程**——智能体们会按照你设定的规则有序“对 话”和传递信息，自动处理各种状态，确保任务一步步清晰、稳定地执行到底。

这让你能轻松组合多个AI能力，构建出比单个智能体更强大、更可靠的自动化应用，而无需深究复杂的底层代码。

✨ 润色：
 LangGraph是一个专为多智能体协同工作而设计的开发框架，其核心价值在于**可视化流程编排**与**稳定可靠的协作机制**。

通过图形化界面，您可以像搭建积木一样，通过拖拽与连接不同的智能体模块（例如分析、撰写、审核等），直观地设计与实现复杂的工作流程。更重要的是，该框架能够**可靠地管理与协调多智能体间的协作过程**——各智能体将依据预设规则进行有序的“对话”与信息传递，自动处理任务状态流转，从而确保整个流程清晰、稳定地逐步推进直至完成。

借助LangGraph，您可以轻松整合多种AI能力，构建出比单一智能体更强大、更可靠的自动化应用，而无需深入复杂的底层编码细节。
==========================================================================================
```

可以看到链式协作协作就类似于“搭积木”一样，按照流程一步一步的执行，比较类似langchain的形式

对比思考：链式协作 vs 中心化协作？

- 链式：简单、无主管，顺序固定，适合流程明确的场景，开发速度快；
- 中心化：灵活、有主管，可动态调整流程，适合任务复杂、需要协调的场景。

#### 7.1.2.3 去中心化协作（Peer-to-peer）：基于状态触发的自主协同

核心逻辑：没有主管，每个智能体都是“平等的”，根据全局状态的变化，自主决定是否执行任务——就像“创业团队”，每个人都盯着项目目标，不用别人分配，自己主动干活。

适用场景：任务灵活、无法提前固定流程，需要智能体自主响应状态变化的场景

```python
# 导入系统模块，用于读取环境变量
import os
# 导入dotenv，用于从.env文件加载环境变量（如API_KEY）
from dotenv import load_dotenv
# 导入LangGraph核心：StateGraph构建状态机、END表示流程终止节点
from langgraph.graph import StateGraph, END
# 导入TypedDict，用于定义强类型的全局状态字典（约束字段类型和名称）
from typing import TypedDict
# 导入ChatPromptTemplate，用于构建大模型的提示词模板
from langchain_core.prompts import ChatPromptTemplate
# 导入StrOutputParser，用于将大模型的ChatMessage输出解析为字符串
from langchain_core.output_parsers import StrOutputParser
# 导入类型注解：Annotated用于给字段加描述、Sequence表示序列类型、Literal表示字面量枚举
from typing import Annotated, Sequence, Literal
# 导入ChatOpenAI，用于调用OpenAI兼容的大模型（此处为deepseek-chat）
from langchain_openai import ChatOpenAI

# ========== 1. 初始化大模型LLM（复用原有配置，无修改） ==========
# 加载.env文件中的环境变量（需在.env中配置API_KEY=你的深度求索密钥）
load_dotenv()
# 初始化ChatOpenAI，对接deepseek-chat大模型
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),  # 从环境变量读取API密钥，避免硬编码
    base_url="https://api.deepseek.com",  # deepseek的API基础地址
    model="deepseek-chat",  # 使用的模型名称
    temperature=0.2  # 低温度值，保证大模型输出的稳定性和确定性，适合决策类任务
)

# ========== 2. 定义全局状态（所有智能体的唯一信息源，核心！） ==========
# 继承TypedDict定义强类型的全局状态，约束所有字段的类型和含义
# 所有智能体仅基于该状态判断是否执行任务，修改也仅更新该状态，确保团队信息同步
class TeamState(TypedDict):
    # 项目核心目标：固定不变，作为所有智能体的行动最终指引
    project_goal: str
    # 待办任务列表：所有智能体共享，智能体自主认领执行，执行后从该列表移除
    todo_tasks: Annotated[Sequence[str], "待办任务列表，智能体自主认领执行，共享可见"]
    # 已完成任务列表：智能体执行任务后，从待办移入该列表，全局可见
    done_tasks: Annotated[Sequence[str], "已完成任务列表，所有智能体可查看，记录执行结果"]
    # 状态更新列表：智能体执行任务后添加该记录，让其他智能体感知全局状态变化（核心通信方式）
    status_updates: Annotated[Sequence[str], "状态更新记录，智能体执行后添加，用于团队信息同步"]
    # 项目完成标志：为True时流程终止，可手动置为True或由路由逻辑判定
    is_finished: Annotated[bool, "项目是否完成的标志，True则LangGraph流程终止"]

# ========== 3. 定义平等智能体（无主管、各有专属技能、自主判断执行） ==========
# 定义3个平等智能体的专属技能，无上下级、无主管，各智能体仅负责自身技能范围内的任务
# 可直接新增键值对扩展智能体（如设计、测试），无需修改核心逻辑
AGENT_SKILLS = {
    "产品智能体": "负责梳理产品需求、设计MVP功能、输出产品文档，确保产品方向匹配项目目标",
    "研发智能体": "负责根据产品文档实现MVP代码、解决技术问题、保证功能可运行，输出可测试的产品",
    "运营智能体": "负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户"
}

# 构建智能体决策的提示词模板（核心：让LLM基于全局状态自主决策，强化格式约束避免输出错误）
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是创业团队的{agent_name}，核心技能是：{agent_skill}。
    团队无主管，所有人平等，你需基于全局项目状态自主判断是否执行任务，判断规则：
    1. 优先看「状态更新」：有新状态变化且需要你的技能衔接，必须主动干活；
    2. 再看「待办任务」：有待办且属于你的技能范围，主动认领执行；
    3. 最后看「项目目标」：无待办但目标未完成，主动提出待办并执行。

    ⚠️ 强制输出格式（必须严格遵守，缺一不可，用===分隔3个部分，不能合并、不能省略）：
    决策：执行/不执行
    ===
    原因：具体判断依据（基于全局状态的细节，不能简略）
    ===
    执行内容：执行则写具体做的事；不执行则严格写「无」，不能写其他内容

    ⚠️ 格式示例（必须按此结构输出）：
    决策：执行
    ===
    原因：状态更新显示产品完成了需求梳理，待办中有开发MVP代码的任务，属于我的研发技能范围
    ===
    执行内容：根据产品需求文档，实现AI智能体工具MVP的核心Python代码，完成本地功能测试
    """),
    # 用户输入部分：将全局状态的所有字段传入，让LLM基于完整状态决策
    ("user", "全局项目状态：\n项目目标：{project_goal}\n待办任务：{todo_tasks}\n已完成任务：{done_tasks}\n最新状态更新：{status_updates}\n项目是否完成：{is_finished}")
])

def agent_node(agent_name: str, agent_skill: str):
    """
    智能体节点**工厂函数**：根据智能体名称和技能，生成LangGraph要求的节点函数
    LangGraph节点函数规则：输入全局状态TeamState，返回更新后的全局状态TeamState
    :param agent_name: 智能体名称（如产品智能体）
    :param agent_skill: 智能体专属技能描述
    :return: 符合LangGraph要求的节点函数（输入state，输出new_state/state）
    """
    # 定义实际的LangGraph节点函数，嵌套函数可继承外部的agent_name和agent_skill
    def node(state: TeamState) -> TeamState:
        # 1. 构建LLM调用链，完成「提示词渲染→大模型推理→输出解析为字符串」
        chain = prompt | llm | StrOutputParser()
        # 调用链，传入智能体信息+全局状态，获取LLM的决策结果
        response = chain.invoke({
            "agent_name": agent_name,
            "agent_skill": agent_skill,** state  # 解包全局状态所有字段
        })

        # 打印LLM原始返回结果，方便排查格式错误（如未按===分隔、少部分等问题）
        print(f"\n===== {agent_name} 原始返回 =====")
        print(response)
        print(f"=========================\n")

        # 2. 分割LLM返回结果，并做**格式预处理**，解决空格/换行导致的识别问题
        # split("===")按分隔符分割，strip()去除每部分的首尾空格/换行/制表符
        parts = [p.strip() for p in response.split("===")]
        # 格式补全：不足3部分用默认值补（避免解包失败），多于3部分取前3个（忽略多余内容）
        if len(parts) < 3:
            parts += ["决策：不执行", "原因：模型返回格式错误，兜底判定", "执行内容：无"][len(parts):]
        if len(parts) > 3:
            parts = parts[:3]

        # 3. 解包分割结果+**异常兜底处理**，确保代码不会因格式错误崩溃
        try:
            # 解包为决策、原因、执行内容三部分
            decision_part, reason_part, action_part = parts
            # 提取核心内容：移除前缀（如决策：），处理模型可能的多余文字
            decision = decision_part.replace("决策：", "").strip() if "决策：" in decision_part else "不执行"
            reason = reason_part.replace("原因：", "").strip() if "原因：" in reason_part else "格式错误，兜底不执行"
            action = action_part.replace("执行内容：", "").strip() if "执行内容：" in action_part else "无"
            # 强制校验决策值：仅允许「执行/不执行」，其他值兜底为不执行（避免无效决策）
            if decision not in ["执行", "不执行"]:
                decision = "不执行"
                reason = f"决策值异常（{decision}），兜底判定不执行"
        except Exception as e:
            # 捕获所有解包/格式异常（如索引错误、类型错误），全部兜底为「不执行」
            decision = "不执行"
            reason = f"格式解析失败：{str(e)}，兜底判定不执行"
            action = "无"

        # 打印标准化后的决策信息，直观查看智能体最终判断结果
        print(f"===== {agent_name} 标准化决策 =====")
        print(f"是否执行：{decision}")
        print(f"判断原因：{reason}")
        print(f"执行内容：{action}\n")

        # 4. 若决策为「执行」，则更新全局状态；否则返回原状态（无任何修改）
        if decision == "执行" and action != "无" and action != "「无」":
            # 深拷贝原状态：LangGraph要求状态不可变，需生成新对象修改
            new_state = state.copy()
            # ① 执行内容加入「已完成任务列表」
            new_state["done_tasks"] = list(new_state["done_tasks"]) + [action]
            # ② 从「待办任务列表」移除已执行的任务（模糊匹配，避免文字完全一致的要求）
            new_state["todo_tasks"] = [t for t in new_state["todo_tasks"] if not any(k in t for k in action.split("：")[0].split("，"))]
            # ③ 添加状态更新记录：让其他智能体感知「该智能体完成了什么」，实现团队信息同步
            new_state["status_updates"] = list(new_state["status_updates"]) + [f"{agent_name}：{action}"]
            # 返回更新后的新状态，供其他智能体使用
            return new_state
        # 若不执行，直接返回原全局状态，无任何修改
        return state

    # 工厂函数返回定义好的节点函数
    return node

# 利用工厂函数，生成3个平等智能体的节点函数（无主管、无优先级，完全平等）
product_agent = agent_node("产品智能体", AGENT_SKILLS["产品智能体"])
dev_agent = agent_node("研发智能体", AGENT_SKILLS["研发智能体"])
ops_agent = agent_node("运营智能体", AGENT_SKILLS["运营智能体"])

# ========== 4. 构建LangGraph无主管状态机（核心：循环执行、自主响应） ==========
# 初始化状态机，绑定全局状态类型TeamState，确保所有节点遵循状态定义
graph_builder = StateGraph(TeamState)
# 向状态机添加3个智能体节点，节点名称与函数一一对应（平等添加，无顺序优先级）
graph_builder.add_node("product", product_agent)  # 产品智能体节点
graph_builder.add_node("dev", dev_agent)          # 研发智能体节点
graph_builder.add_node("ops", ops_agent)          # 运营智能体节点

def should_continue(state: TeamState) -> Literal["product", "dev", "ops", END]:
    """
    LangGraph条件路由函数：运营智能体执行后，判断流程**继续循环**还是**终止**
    返回值约束为字面量：继续则返回下一个节点（product），终止则返回END
    :param state: 当前的全局状态
    :return: 下一个节点名称 / END（终止）
    """
    # 终止条件：① 手动置为项目完成 ② 状态更新数>3（完成3个核心任务，模拟项目结束）
    if state["is_finished"] or len(state["status_updates"]) > 3:
        return END  # 返回END，流程终止
    return "product"  # 未终止则返回产品智能体，继续循环执行（产品→研发→运营）

# 设置状态机**入口点**：首次执行从产品智能体开始（无主管分配，固定入口）
graph_builder.set_entry_point("product")
# 定义节点间的**顺序边**：产品执行完→研发执行，研发执行完→运营执行
graph_builder.add_edge("product", "dev")
graph_builder.add_edge("dev", "ops")
# 定义**条件边**：运营执行完后，调用should_continue判断是继续循环还是终止
graph_builder.add_conditional_edges("ops", should_continue)

# 编译状态机，生成可运行的LangGraph图对象（编译后不可修改，可多次调用）
graph = graph_builder.compile()

# ========== 5. 测试运行：启动创业团队项目（无主管，智能体自主干活） ==========
if __name__ == "__main__":
    # 初始化项目**初始全局状态**：设定目标、初始待办、初始状态，项目未完成
    initial_state = TeamState(
        # 项目核心目标（固定不变）
        project_goal="开发一个AI智能体工具的MVP并完成初步推广，实现种子用户获取",
        # 初始待办任务（智能体自主认领执行，执行后自动移除）
        todo_tasks=[
            "梳理AI智能体工具MVP的核心需求",
            "实现MVP的核心功能代码",
            "撰写MVP推广文案并在小红书初步发布"
        ],
        done_tasks=[],  # 初始无已完成任务
        # 初始状态更新：标记项目启动，让所有智能体感知项目开始
        status_updates=["项目启动：开始推进AI智能体工具MVP开发与推广"],
        is_finished=False  # 初始项目未完成
    )

    # 打印项目启动信息，直观查看初始目标和待办
    print("===== 创业团队项目启动 =====")
    print(f"项目目标：{initial_state['project_goal']}")
    print(f"初始待办：{initial_state['todo_tasks']}\n")

    # 流式运行状态机：逐节点输出执行过程，直观看到智能体决策和状态更新
    # graph.stream()返回生成器，每次yield一个节点的执行结果（节点名称+更新后的状态）
    for step in graph.stream(initial_state):
        # 遍历每一步的节点和状态（单节点执行，故仅一个键值对）
        for node, state in step.items():
            print(f"===== 节点 {node} 执行后 - 全局状态 =====")
            print(f"✅ 已完成任务：{state['done_tasks']}")
            print(f"📋 剩余待办任务：{state['todo_tasks']}")
            print(f"📌 最新团队状态：{state['status_updates'][-1]}\n")

    # 调用graph.invoke()获取项目最终的全局状态，打印最终执行结果
    final_state = graph.invoke(initial_state)
    print("===== 项目执行完成 - 最终结果 =====")
    print(f"项目核心目标：{final_state['project_goal']}")
    print(f"✅ 团队全部已完成任务：{final_state['done_tasks']}")
    print(f"📌 团队完整状态更新记录：{final_state['status_updates']}")
```

运行结果

```
项目目标：开发一个AI智能体工具的MVP并完成初步推广，实现种子用户获取
初始待办：['梳理AI智能体工具MVP的核心需求', '实现MVP的核心功能代码', '撰写MVP推广文案并在小红书初步发布']


===== 产品智能体 原始返回 =====
决策：执行
===
原因：待办任务中有「梳理AI智能体工具MVP的核心需求」，这属于我的核心技能范围（负责梳理产品需求、设计MVP功能）。根据判断规则第2条，有待办且属于我的技能范围，应主动认领执行。
===
执行内容：主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确工具的核心用户画像和使用场景；2. 分析竞品，确定差异化功能点；3. 定义MVP的最小功能集合 和核心用户流程；4. 输出初步的产品需求文档（PRD）或功能清单，为后续开发任务提供依据。
=========================

===== 产品智能体 标准化决策 =====
是否执行：执行
判断原因：待办任务中有「梳理AI智能体工具MVP的核心需求」，这属于我的核心技能范围（负责梳理产品需求、设计MVP功能）。根据判断规则第2条，有待办且属于我的技能范围，应主动认领执行。   
执行内容：主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确工具的核心用户画像和使用场景；2. 分析竞品，确定差异化功能点；3. 定义MVP的最小功能集合 和核心用户流程；4. 输出初步的产品需求文档（PRD）或功能清单，为后续开发任务提供依据。

===== 节点 product 执行后 - 全局状态 =====
✅ 已完成：['主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确工具的核心用户画像和使用场景；2. 分析竞品，确定差异化功能点；3. 定义MVP的最小功能集合和核心用户流程；4. 输出初步的产品需求文档（PRD）或功能清单，为后续开发任务提供依据。']
📋 剩余待办：['梳理AI智能体工具MVP的核心需求', '实现MVP的核心功能代码', '撰写MVP推广文案并在小红书初步发布']
📌 最新状态：产品智能体：主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确工具的核心用户画像和使用场景；2. 分析竞品，确定差异化功能点；3. 定义MVP的最小功能集合和核心用户流程；4. 输出初步的产品需求文档（PRD）或功能清单，为后续开发任务提供依据。


===== 研发智能体 原始返回 =====
决策：执行
===
原因：待办任务中存在「实现MVP的核心功能代码」，这属于我的研发技能范围。同时，最新状态更新显示产品智能体已经完成了需求梳理并输出了产品需求文档（PRD）或功能清单，这为我的开发工作提供了直接依据，符合“有新状态变化且需要你的技能衔接”的判断规则。
===
执行内容：主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 仔细阅读并理解产品智能体输出的需求文档或功能清单；2. 根据需求，设计并实现MVP的核心功能模块代码；3. 进行本地功 能测试，确保核心流程可运行；4. 输出可测试的MVP产品代码。
=========================

===== 研发智能体 标准化决策 =====
是否执行：执行
判断原因：待办任务中存在「实现MVP的核心功能代码」，这属于我的研发技能范围。同时，最新状态更新显示产品智能体已经完成了需求梳理并输出了产品需求文档（PRD）或功能清单，这为我的开发工作提供了直接依据，符合“有新状态变化且需要你的技能衔接”的判断规则。
执行内容：主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 仔细阅读并理解产品智能体输出的需求文档或功能清单；2. 根据需求，设计并实现MVP的核心功能模块代码；3. 进行本地功 能测试，确保核心流程可运行；4. 输出可测试的MVP产品代码。

===== 节点 dev 执行后 - 全局状态 =====
✅ 已完成：['主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确工具的核心用户画像和使用场景；2. 分析竞品，确定差异化功能点；3. 定义MVP的最小功能集合和核心用户流程；4. 输出初步的产品需求文档（PRD）或功能清单，为后续开发任务提供依据。', '主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 仔细阅读并理解产品智能体输出的需求文档或功能清单；2. 根据需求，设计并实现MVP的核心功能模块代码；3. 进行本地功能测试，确保核心流程可运行；4. 输出可测试的MVP产品代码。']
📋 剩余待办：['梳理AI智能体工具MVP的核心需求', '实现MVP的核心功能代码', '撰写MVP推广文案并在小红书初步发布']
📌 最新状态：研发智能体：主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 仔细阅读并理解产品智能体输出的需求文档或功能清单；2. 根据需求，设计并实现MVP的核心功能模块代码 ；3. 进行本地功能测试，确保核心流程可运行；4. 输出可测试的MVP产品代码。


===== 运营智能体 原始返回 =====
决策：执行
===
原因：待办任务中有一项「撰写MVP推广文案并在小红书初步发布」，这属于我的核心技能范围（负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户）。同时，最新状态更新显示研发智能体已完成MVP核心功能代码的实现，这意味着MVP产品已具备可推广的基础，需要我的技能进行衔接以获取种子用户。
===
执行内容：主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 分析已完成MVP的核心功能与用户价值，提炼推广卖点；2. 针对小红书平台特性，撰写吸引目标种子用户的推广文案；3. 设计初步的发布计划（如发布时间、话题标签等）；4. 在小红书平台完成初步发布，并开始监测用户反馈。
=========================

===== 运营智能体 标准化决策 =====
是否执行：执行
判断原因：待办任务中有一项「撰写MVP推广文案并在小红书初步发布」，这属于我的核心技能范围（负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户）。同时，最新状态更新显示研发智能体已完成MVP核心功能代码的实现，这意味着MVP产品已具备可推广的基础，需要我的技能进行衔接以获取种子用户。
执行内容：主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 分析已完成MVP的核心功能与用户价值，提炼推广卖点；2. 针对小红书平台特性，撰写吸引目标种子用户的推广文案；3. 设计初步的发布计划（如发布时间、话题标签等）；4. 在小红书平台完成初步发布，并开始监测用户反馈。

===== 节点 ops 执行后 - 全局状态 =====
✅ 已完成：['主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确工具的核心用户画像和使用场景；2. 分析竞品，确定差异化功能点；3. 定义MVP的最小功能集合和核心用户流程；4. 输出初步的产品需求文档（PRD）或功能清单，为后续开发任务提供依据。', '主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 仔细阅读并理解产品智能体输出的需求文档或功能清单；2. 根据需求，设计并实现MVP的核心功能模块代码；3. 进行本地功能测试，确保核心流程可运行；4. 输出可测试的MVP产品代码。', '主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 分析已完成MVP的核心功能与用户价值，提炼推广卖点；2. 针对小红书平台特性，撰写吸引目标种子用户的推广文案；3. 设计初步的发布计划（如发布时间、话题标签等）；4. 在小红书平台完成初步发布，并开始监测用户反馈。']
📋 剩余待办：['梳理AI智能体工具MVP的核心需求', '实现MVP的核心功能代码', '撰写MVP推广文案并在小红书初步发布']
📌 最新状态：运营智能体：主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 分析已完成MVP的核心功能与用户价值，提炼推广卖点；2. 针对小红书平台特性，撰写吸引目 标种子用户的推广文案；3. 设计初步的发布计划（如发布时间、话题标签等）；4. 在小红书平台完成初步发布，并开始监测用户反馈。


===== 产品智能体 原始返回 =====
决策：执行
===
原因：状态更新显示项目已启动，待办任务中有「梳理AI智能体工具MVP的核心需求」这一项，这明确属于我（产品智能体）负责梳理产品需求、设计MVP功能的核心技能范围。根据判断规则，有待办且属于我的技能范围，应主动认领执行。
===
执行内容：主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确MVP要解决的核心用户问题；2. 定义目标用户画像和使用场景；3. 梳理并确定MVP必须包含的核心功能列表，确保功能精简、聚焦；4. 输出初步的产品需求文档或功能清单，为后续开发任务提供清晰依据。
=========================

===== 产品智能体 标准化决策 =====
是否执行：执行
判断原因：状态更新显示项目已启动，待办任务中有「梳理AI智能体工具MVP的核心需求」这一项，这明确属于我（产品智能体）负责梳理产品需求、设计MVP功能的核心技能范围。根据判断规则，有待办且属于我的技能范围，应主动认领执行。
执行内容：主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确MVP要解决的核心用户问题；2. 定义目标用户画像和使用场景；3. 梳理并确定MVP必须包含的核心功能列表，确保功能精简、聚焦；4. 输出初步的产品需求文档或功能清单，为后续开发任务提供清晰依据。


===== 研发智能体 原始返回 =====
决策：执行
===
原因：根据全局项目状态，待办任务中存在「实现MVP的核心功能代码」任务，这明确属于我的研发技能范围。同时，最新状态更新显示产品智能体已经完成了需求梳理并输出了产品需求文档或功能清单 ，这为我的开发工作提供了清晰的依据，属于状态变化后需要我的技能衔接的情况。
===
执行内容：主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 基于已完成任务中输出的产品需求文档或功能清单，进行技术方案设计；2. 编写MVP核心功能代码，确保功能可运行；3. 进 行本地功能测试，保证代码质量。
=========================

===== 研发智能体 标准化决策 =====
是否执行：执行
判断原因：根据全局项目状态，待办任务中存在「实现MVP的核心功能代码」任务，这明确属于我的研发技能范围。同时，最新状态更新显示产品智能体已经完成了需求梳理并输出了产品需求文档或功能 清单，这为我的开发工作提供了清晰的依据，属于状态变化后需要我的技能衔接的情况。
执行内容：主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 基于已完成任务中输出的产品需求文档或功能清单，进行技术方案设计；2. 编写MVP核心功能代码，确保功能可运行；3. 进 行本地功能测试，保证代码质量。


===== 运营智能体 原始返回 =====
决策：执行
===
原因：待办任务中有一项「撰写MVP推广文案并在小红书初步发布」，这属于我的核心技能范围（负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户）。同时，状态更新显示研发智能体已完成MVP核心功能代码的实现和测试，这意味着MVP已具备可推广的基础，需要我的技能进行衔接以获取种子用户。
===
执行内容：主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 基于已完成任务中梳理的用户画像、使用场景和MVP核心功能，设计推广方案；2. 撰写针对小红书平台特点的推广文案，突出工具的核心价值和使用场景；3. 在小红书平台完成初步发布，并监控初步反馈，为后续推广积累数据。
=========================

===== 运营智能体 标准化决策 =====
是否执行：执行
判断原因：待办任务中有一项「撰写MVP推广文案并在小红书初步发布」，这属于我的核心技能范围（负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户）。同时，状态更新显示研发智能体已完成MVP核心功能代码的实现和测试，这意味着MVP已具备可推广的基础，需要我的技能进行衔接以获取种子用户。
执行内容：主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 基于已完成任务中梳理的用户画像、使用场景和MVP核心功能，设计推广方案；2. 撰写针对小红书平台特点的推广文案，突出工具的核心价值和使用场景；3. 在小红书平台完成初步发布，并监控初步反馈，为后续推广积累数据。

===== 项目执行完成 - 最终结果 =====
项目目标：开发一个AI智能体工具的MVP并完成初步推广，实现种子用户获取
✅ 全部已完成任务：['主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确MVP要解决的核心用户问题；2. 定义目标用户画像和使用场景；3. 梳理并确定MVP必 须包含的核心功能列表，确保功能精简、聚焦；4. 输出初步的产品需求文档或功能清单，为后续开发任务提供清晰依据。', '主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 基于已完 成任务中输出的产品需求文档或功能清单，进行技术方案设计；2. 编写MVP核心功能代码，确保功能可运行；3. 进行本地功能测试，保证代码质量。', '主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 基于已完成任务中梳理的用户画像、使用场景和MVP核心功能，设计推广方案；2. 撰写针对小红书平台特点的推广文案，突出工具的核心价值和使用场景；3. 在小红书平台完成初步发布，并监控初步反馈，为后续推广积累数据。']
📌 完整状态记录：['项目启动：开始推进AI智能体工具MVP开发与推广', '产品智能体：主动认领并执行「梳理AI智能体工具MVP的核心需求」任务。具体包括：1. 与团队沟通，明确MVP要解决的核心用 户问题；2. 定义目标用户画像和使用场景；3. 梳理并确定MVP必须包含的核心功能列表，确保功能精简、聚焦；4. 输出初步的产品需求文档或功能清单，为后续开发任务提供清晰依据。', '研发智能体：主动认领并执行「实现MVP的核心功能代码」任务。具体包括：1. 基于已完成任务中输出的产品需求文档或功能清单，进行技术方案设计；2. 编写MVP核心功能代码，确保功能可运行；3. 进行本地功 能测试，保证代码质量。', '运营智能体：主动认领并执行「撰写MVP推广文案并在小红书初步发布」任务。具体包括：1. 基于已完成任务中梳理的用户画像、使用场景和MVP核心功能，设计推广方案；2. 撰写针对小红书平台特点的推广文案，突出工具的核心价值和使用场景；3. 在小红书平台完成初步发布，并监控初步反馈，为后续推广积累数据。']
```

说明：去中心化协作的核心是“状态驱动”，每个智能体都监听全局状态，自主决定行为，不需要主管分配——这种模式灵活性最高，但开发和调试难度也最大（需要控制智能体之间的冲突，比如避免多个智能体处理同一条消息），一般来说企业用的非常少~~

### 7.1.3 智能体间的通信机制

多智能体协作，最关键的不是“有多少个智能体”，而是“智能体之间怎么沟通”——如果沟通不畅，就会出现“各干各的”，甚至冲突。LangGraph 提供了2种核心通信机制，咱们结合前面的案例，详细拆解。

#### 7.1.3.1 基于全局状态（State）的消息共享

这是LangGraph最核心、最常用的通信方式——所有智能体共享一个“全局状态”（就像一个“公共白板”），每个智能体可以读取白板上的内容，也可以往白板上写内容，通过白板实现信息互通。

比如前面的中心化、链式案例中都是智能体之间通信的“载体”

#### 7.1.3.2 角色定义与 System Prompt 的差异化设计

如果说“全局状态”是智能体之间的“沟通内容”，那么“角色定义+差异化System Prompt”就是智能体之间的“沟通规则”——明确每个智能体的“身份”和“职责”，避免沟通混乱。

实操技巧：

1. 给每个智能体设置“专属System Prompt”，明确职责边界（比如writer_agent只写不纠错，corrector_agent只纠错不润色）；
2. 在Prompt中加入“通信约定”，比如主管智能体的Prompt中，明确“只返回下一个智能体名称”，避免输出多余内容，导致其他智能体无法读取；
3. 统一“输出格式”，比如可视化智能体只输出代码，数据分析智能体只输出结论，确保其他智能体能正确读取和使用其输出。

## 7.2 复杂流程的高级管控技术

当多智能体处理的任务越来越复杂（比如包含子任务、并行处理、循环重试），简单的“链式”“中心化”已经不够用了——这时候就需要LangGraph的高级管控技术，帮我们拆解复杂流程、提升效率、避免出错。

### 7.2.1 子图（Subgraphs）机制：复杂系统的模块化拆解

子图，顾名思义，就是“图中的图”——把一个复杂的流程，拆成多个独立的“子流程”（每个子流程就是一个子图），主图负责调用子图，子图负责处理具体的子任务。

核心优势：实现“逻辑隔离”和“复用”——比如一个“文档处理”子图，可同时被“报告生成”“数据提取”两个主流程调用，不用重复开发。

**什么是子图？**

类比理解：子图就像“函数”——我们把一段常用的代码写成一个函数，后续需要用到时，直接调用这个函数，不用重复写代码；子图就是把一段常用的流程写成一个“子图”，主图需要时，直接调用这个子图。

LangGraph中，子图的实现非常简单：先定义一个子图（StateGraph），编译后，作为一个“节点”，添加到主图中即可。

#### 7.2.1.1 实操案例：在主流程中嵌入一个独立的子图

案例演示：主图是模拟老师收卷子，子图用来批改和统计

```python
# 【教案实操案例】7.2.1.2 在主流程中嵌入独立可复用的“作业批改”子图
# 核心需求（学生易理解）：
# 主流程：接收学生作业 → 作业批改（子图，可复用） → 生成批改反馈
# 子流程（作业批改子图）：检查完成度→检查正确率→计算得分（独立可复用）
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional  # 导入Optional，适配初始None值
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# -------------------------- 全局初始化（与之前一致，保持教学统一）--------------------------
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3  # 低温度保证输出固定，方便学生观察
)
output_parser = StrOutputParser()  # 统一解析为字符串，避免报错

# -------------------------- 第一步：定义“作业批改子图”（独立、可复用）--------------------------
# 子图独立状态：添加Optional，所有字段支持初始None/默认值
class CorrectionSubgraphState(TypedDict):
    homework_content: str                # 待批改作业（主图传递，必传）
    completion: Optional[str] = None     # 完成度：完成/未完成（初始None）
    accuracy: Optional[str] = None       # 正确率：正确率XX%（初始None）
    score: Optional[int] = 0             # 最终得分（初始默认0分，避免类型错误）

# 子图智能体（分工明确，LLM输出固定格式，无解析冗余）
# 智能体1：检查作业完成度（仅输出「完成」/「未完成」）
completion_check_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是作业完成度检查老师，仅输出「完成」或「未完成」，不添加任何额外文字！"),
    ("user", "作业内容：{homework_content}，判断是否完成（有具体内容、无空白即为完成）")
])
completion_check_agent = completion_check_prompt | llm | output_parser

# 智能体2：检查作业正确率（仅输出「正确率XX%」）
accuracy_check_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是作业正确率检查老师，仅输出「正确率XX%」，不添加任何额外文字！"),
    ("user", "作业内容：{homework_content}，假设是数学计算题，合理估算正确率")
])
accuracy_check_agent = accuracy_check_prompt | llm | output_parser

# 智能体3：计算最终得分（仅输出0-100整数）
score_calc_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是得分计算老师，仅输出0-100的整数，不添加任何额外文字！"),
    ("user", "完成度：{completion}，正确率：{accuracy}，计分规则：完成得60基础分，正确率每10%加4分，未完成得0分")
])
score_calc_agent = score_calc_prompt | llm | output_parser

# 子图节点函数（打印日志+更新状态，学生易观察）
def check_completion_node(state: CorrectionSubgraphState) -> CorrectionSubgraphState:
    """子图节点1：检查作业完成度"""
    print(f"🔍 子图执行 - 检查作业完成度")
    completion = completion_check_agent.invoke({"homework_content": state["homework_content"]})
    # 修复解包顺序：先解包原状态，再更新新字段（统一规范）
    return {**state, "completion": completion}

def check_accuracy_node(state: CorrectionSubgraphState) -> CorrectionSubgraphState:
    """子图节点2：检查作业正确率"""
    print(f"🔍 子图执行 - 检查作业正确率")
    accuracy = accuracy_check_agent.invoke({"homework_content": state["homework_content"]})
    return {**state, "accuracy": accuracy}

def calc_score_node(state: CorrectionSubgraphState) -> CorrectionSubgraphState:
    """子图节点3：计算最终得分"""
    print(f"🔍 子图执行 - 计算作业得分")
    # 子图内部空值校验：避免LLM输出异常导致报错
    completion = state["completion"] or "未完成"
    accuracy = state["accuracy"] or "正确率0%"
    # 调用得分智能体并转整数（增加异常捕获，适配LLM偶尔输出非数字的情况）
    try:
        score = int(score_calc_agent.invoke({"completion": completion, "accuracy": accuracy}))
    except:
        score = 0
    return {**state, "score": score}

# 构建并编译子图（独立流程，可复用）
correction_subgraph = StateGraph(CorrectionSubgraphState)
correction_subgraph.add_node("check_completion", check_completion_node)
correction_subgraph.add_node("check_accuracy", check_accuracy_node)
correction_subgraph.add_node("calc_score", calc_score_node)
# 子图线性流程：开始→完成度→正确率→计算得分→结束
correction_subgraph.add_edge(START, "check_completion")
correction_subgraph.add_edge("check_completion", "check_accuracy")
correction_subgraph.add_edge("check_accuracy", "calc_score")
correction_subgraph.add_edge("calc_score", END)
compiled_correction_subgraph = correction_subgraph.compile()

# -------------------------- 第二步：定义主图（作业处理主流程，调用子图）--------------------------
# 主图全局状态：添加Optional，适配初始None值，字段含义学生易理解
class HomeworkMainState(TypedDict):
    homework_content: str                          # 学生作业内容（必传）
    correction_result: Optional[CorrectionSubgraphState] = None  # 子图批改结果（初始None）
    feedback: Optional[str] = None                 # 最终批改反馈（初始None）

# 主图智能体：仅生成批改反馈，逻辑简单
feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是班主任，根据批改结果给学生写1-2句亲切反馈，语气贴合得分情况。"),
    ("user", "作业内容：{homework_content}\n批改结果：完成度{completion}，正确率{accuracy}，得分{score}\n生成反馈：")
])
feedback_agent = feedback_prompt | llm | output_parser

# 主图节点函数（核心：修复子图调用节点的解包顺序，添加空值校验）
def receive_homework_node(state: HomeworkMainState) -> HomeworkMainState:
    """主图节点1：接收学生作业（模拟，日志可视化）"""
    print(f"\n📥 主图执行 - 接收学生作业：{state['homework_content']}")
    return state  # 接收作业，状态无变化

def correction_subgraph_node(state: HomeworkMainState) -> HomeworkMainState:
    """主图节点2：调用作业批改子图（教学核心！重点标注）"""
    print(f"\n📤 主图执行 - 调用作业批改子图")
    # 主图向子图传递参数：仅传子图需要的作业内容，其他用子图默认初始值
    subgraph_input = {"homework_content": state["homework_content"]}
    # 调用编译后的子图，获取完整批改结果
    subgraph_output = compiled_correction_subgraph.invoke(subgraph_input)
    # 🔥 核心修复：解包顺序反了导致的None覆盖问题！先解包原状态，再更新子图结果
    print(f"✅ 主图接收子图批改结果：完成度{subgraph_output['completion']}，正确率{subgraph_output['accuracy']}，得分{subgraph_output['score']}")
    return {**state, "correction_result": subgraph_output}

def generate_feedback_node(state: HomeworkMainState) -> HomeworkMainState:
    """主图节点3：生成批改反馈（添加空值校验，避免报错）"""
    print(f"\n📝 主图执行 - 生成学生批改反馈")
    # 空值校验：防止子图结果未正确更新（双重保险）
    if not state.get("correction_result"):
        return {**state, "feedback": "作业批改失败，无法生成反馈！"}
    # 提取子图批改结果（简化变量名，代码更清晰）
    corr = state["correction_result"]
    homework = state["homework_content"]
    # 调用反馈智能体生成结果
    feedback = feedback_agent.invoke({
        "homework_content": homework,
        "completion": corr["completion"] or "未完成",
        "accuracy": corr["accuracy"] or "正确率0%",
        "score": corr["score"] or 0
    })
    return {**state, "feedback": feedback}

# 构建并编译主图（嵌入子图，流程清晰）
main_graph = StateGraph(HomeworkMainState)
# 添加主图节点：接收作业 → 调用子图 → 生成反馈
main_graph.add_node("receive_homework", receive_homework_node)
main_graph.add_node("correction_subgraph", correction_subgraph_node)
main_graph.add_node("generate_feedback", generate_feedback_node)
# 主图线性流程：严格按教学需求设计，学生易观察
main_graph.add_edge(START, "receive_homework")
main_graph.add_edge("receive_homework", "correction_subgraph")
main_graph.add_edge("correction_subgraph", "generate_feedback")
main_graph.add_edge("generate_feedback", END)
compiled_main_graph = main_graph.compile()

# -------------------------- 第三步：测试运行（学生可直接观察全过程，无报错）--------------------------
if __name__ == "__main__":
    # 测试1：优秀作业（完成+正确率高，预期：正面反馈）
    print("="*60, "测试1：优秀作业（完成+正确率高）", "="*60)
    test1_input = {
        "homework_content": "2+3=5，4+6=10，7+8=15，9+11=20",
        # 初始值为None，无需手动赋值，由子图节点更新
        "correction_result": None,
        "feedback": None
    }
    result1 = compiled_main_graph.invoke(test1_input)
    print(f"\n🎉 最终结果 - 学生反馈：{result1['feedback']}\n")

    # 测试2：不合格作业（未完成+正确率低，预期：改进反馈）
    print("="*60, "测试2：不合格作业（未完成+正确率低）", "="*60)
    test2_input = {
        "homework_content": "2+3=6，4+6=（空白），7+8=（空白）",
        "correction_result": None,
        "feedback": None
    }
    result2 = compiled_main_graph.invoke(test2_input)
    print(f"\n🎉 最终结果 - 学生反馈：{result2['feedback']}")
```

运行结果

```
======== 测试1：优秀作业（完成+正确率高） ============================================================

📥 主图执行 - 接收学生作业：2+3=5，4+6=10，7+8=15，9+11=20

📤 主图执行 - 调用作业批改子图
🔍 子图执行 - 检查作业完成度
🔍 子图执行 - 检查作业正确率
🔍 子图执行 - 计算作业得分
✅ 主图接收子图批改结果：完成度完成，正确率正确率75%，得分78

📝 主图执行 - 生成学生批改反馈

🎉 最终结果 - 学生反馈：这次作业完成得很认真，但要注意检查计算过程哦，争取下次全对！

============ 测试2：不合格作业（未完成+正确率低） ============================================================    

📥 主图执行 - 接收学生作业：2+3=6，4+6=（空白），7+8=（空白）

📤 主图执行 - 调用作业批改子图
🔍 子图执行 - 检查作业完成度
🔍 子图执行 - 检查作业正确率
🔍 子图执行 - 计算作业得分
✅ 主图接收子图批改结果：完成度未完成，正确率正确率33%，得分0

📝 主图执行 - 生成学生批改反馈

🎉 最终结果 - 学生反馈：别灰心，这次作业只完成了一部分。下次记得把题目都做完，老师相信你一定能算对！
```

关键说明：

1. 子图必须独立定义、独立编译，编译后的子图可作为“节点”，直接添加到主图中；
2. 主图和子图的状态可以互通：主图可将数据传递给子图（通过子图的初始状态），子图的输出可作为主图的状态（通过调用子图后的返回值）；
3. 子图的优势：可复用——如果其他主流程也需要“文档校验”，直接调用这个子图即可，不用重复开发格式校验、内容校验的逻辑。

### 7.2.2 并行任务处理（Parallelization）：提升效率的关键

很多时候，多智能体处理任务不需要“顺序接力”，而是可以“同时干活”，比如同时找3个人帮忙查资料——一个查美食、一个查景点、一个查交通，你会让他们“一个查完再查下一个”，还是“一起查”？显然是一起查效率更高！这就是多智能体的“并行任务处理”，核心是**同时调用多个智能体执行独立任务，最后汇总结果**，对应LangGraph里的“扇出（Fan-out）与扇入（Fan-in）”。

先搞懂两个概念（概念不用记，理解即可）：

- 扇出（Fan-out）：一个“总指挥”节点，同时派出多个智能体去执行不同的子任务（比如上面说的“查美食、查景点、查交通”）；
- 扇入（Fan-in）：多个智能体的任务完成后，把它们的结果统一汇总到一个“汇总节点”，进行整合输出。

#### 7.2.2.1 实操案例：并行运行3个智能体

```python
# ================== 导入依赖 ==================
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 定义状态 ==================
class CandidateState(TypedDict):
    resume: str
    job_requirements: str
    skills: str
    interview_feedback: str
    resume_info: Optional[str]     # 扇出节点输出
    skill_match: Optional[str]     # 扇出节点输出
    interview_summary: Optional[str]  # 扇出节点输出
    summary: Optional[str]         # 汇总节点输出

# ================== 定义扇出智能体（管道符风格 + 打印） ==================
def resume_node(state: CandidateState) -> dict:
    result = (ChatPromptTemplate.from_template(
        "请阅读以下候选人简历内容，提取关键信息（姓名、学历、工作经历、技能清单）：\n{resume}"
    ) | llm).invoke(state)
    print("\n[扇出节点] 简历信息:", result)
    return {"resume_info": result}

def skill_node(state: CandidateState) -> dict:
    result = (ChatPromptTemplate.from_template(
        "根据岗位要求：{job_requirements}，请分析候选人技能匹配情况，并给出匹配分（0-10）：\n候选人技能：{skills}"
    ) | llm).invoke(state)
    print("\n[扇出节点] 技能匹配:", result)
    return {"skill_match": result}

def interview_node(state: CandidateState) -> dict:
    result = (ChatPromptTemplate.from_template(
        "请根据以下面试评价内容，总结候选人的优点和潜在改进点，简明扼要：\n{interview_feedback}"
    ) | llm).invoke(state)
    print("\n[扇出节点] 面试总结:", result)
    return {"interview_summary": result}

# ================== 定义扇入汇总节点 ==================
def summary_node(state: CandidateState) -> CandidateState:
    prompt = (ChatPromptTemplate.from_template(
        "请整合以下候选人信息，生成一份完整的招聘推荐报告（150字以内）：\n"
        "简历关键信息：{resume_info}\n"
        "技能匹配分析：{skill_match}\n"
        "面试总结：{interview_summary}"
    ) | llm)

    result = prompt.invoke({
        "resume_info": state["resume_info"],
        "skill_match": state["skill_match"],
        "interview_summary": state["interview_summary"]
    })

    print("\n[汇总节点] 招聘推荐报告:", result)
    state["summary"] = result
    return state

# ================== 构建图 ==================
graph = StateGraph(state_schema=CandidateState)

graph.add_node("start", lambda state: state)
graph.add_node("resume_info", resume_node)
graph.add_node("skill_match", skill_node)
graph.add_node("interview_summary", interview_node)
graph.add_node("summary", summary_node)

# 扇出
graph.add_edge(START, "resume_info")
graph.add_edge(START, "skill_match")
graph.add_edge(START, "interview_summary")

# 扇入
graph.add_edge("resume_info", "summary")
graph.add_edge("skill_match", "summary")
graph.add_edge("interview_summary", "summary")

# 汇总节点到结束
graph.add_edge("summary", END)

# ================== 编译并运行 ==================
app = graph.compile()

input_state = CandidateState(
    resume="张三，硕士学历，5年软件开发经验，熟悉Python、Java、SQL。",
    job_requirements="熟悉Python和数据分析，具有团队协作能力。",
    skills="Python, Java, SQL, 数据分析",
    interview_feedback="表达清晰，逻辑性强，但在团队管理经验方面稍弱。",
    resume_info=None,
    skill_match=None,
    interview_summary=None,
    summary=None
)

result = app.invoke(input_state)

print("\n=== 最终招聘推荐报告 ===")
print(result["summary"].content)

```

运行结果

```
[扇出节点] 面试总结: content='**优点：**\n- 表达清晰，逻辑性强\n\n**潜在改进点：**\n- 团队管理经验有待加强' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 25, 'prompt_tokens': 38, 'total_tokens': 63, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 38}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '4a71df6a-6980-438c-a856-0ec850d86074', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c0e24-0fcc-7042-9bc8-37c549344ad3-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 38, 'output_tokens': 25, 'total_tokens': 63, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}

[扇出节点] 简历信息: content='**姓名：** 张三  \n**学历：** 硕士  \n**工作经历：** 5年软件开发经验  \n**技能清单：** Python、Java、SQL' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 35, 'prompt_tokens': 43, 'total_tokens': 78, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 43}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': 'a77bdb3a-39ca-4edd-9ed9-13ebcc153c0b', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c0e24-0fd0-7451-bb31-8b9de6ab8ccf-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 43, 'output_tokens': 35, 'total_tokens': 78, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}

[扇出节点] 技能匹配: content='根据您提供的岗位要求和候选人技能，我们来逐一分析匹配情况：  \n\n**1. 岗位要求与候选人技能对比**  \n- **熟悉Python**：候选人具备 Python 技能 ✅  \n- **数据分析**：候选人明确列出“数据分析”技能 ✅  \n- **团队协作能力**：岗位要求中提及，但候选人技能列表未明确体现（技能列表通常只列技术能力，团队协 作可能在其他部分说明）❓  \n\n**2. 匹配度分析**  \n- 候选人具备 Python 和数据分析，这两项核心要求都符合。  \n- 候选人还额外掌握 Java 和 SQL，SQL 对数据分析岗位很 有帮助，属于加分项。  \n- 团队协作能力未在技能列表中体现，但通常可以在简历的其他部分（如项目经历、工作经历）中判断，此处仅凭技能列表无法确认，因此暂时视为“未知”，不扣分但也不额外加分。  \n\n**3. 匹配分计算（0-10）**  \n- 核心要求（Python、数据分析）完全匹配 → 基础分 8/10  \n- 额外相关技能（SQL）加强数据分析能力 → +0.5  \n- Java 对岗位不一定直接必要，但体现编程广度 → +0.5  \n- 团队协作能力未在技能中显示，但可能隐含，暂不扣分 → +0  \n- **总分：9/10**  \n\n**4. 建议**  \n建议在面试或 简历筛选中进一步确认候选人的团队协作经验（如项目合作、跨部门沟通等），若具备则匹配度可达 9.5 甚至 10 分。  \n\n**匹配分：9/10**' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 352, 'prompt_tokens': 47, 'total_tokens': 399, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 47}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '78652027-ebd8-4f62-931c-8372ef8040d5', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c0e24-0fd3-7b83-8c83-ea88bce9ea7a-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 47, 'output_tokens': 352, 'total_tokens': 399, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}

[汇总节点] 招聘推荐报告: content='**招聘推荐报告：张三**\n**基本信息**：硕士学历，5年软件开发经验。\n**技能匹配**：核心技能（Python、数据分析）完全匹配，并掌握Java、SQL等加分项，综合匹配度9/10。\n**面试表现**：表达清晰、逻辑性强，但团队管理经验相对薄弱。\n**综合建议**：技术能力突出，高度匹配岗位核心要求，推荐录用。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 86, 'prompt_tokens': 1341, 'total_tokens': 1427, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 1149}, 'model_provider': 'openai', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '05cdfdd8-d563-4870-a8fa-db4ea2fc87ca', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c0e24-421a-7b71-9e97-94c729642296-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 1341, 'output_tokens': 86, 'total_tokens': 1427, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}}

=== 最终招聘推荐报告 ===
**招聘推荐报告：张三**
**基本信息**：硕士学历，5年软件开发经验。
**技能匹配**：核心技能（Python、数据分析）完全匹配，并掌握Java、SQL等加分项，综合匹配度9/10。
**面试表现**：表达清晰、逻辑性强，但团队管理经验相对薄弱。
**综合建议**：技术能力突出，高度匹配岗位核心要求，推荐录用。
```

【学习提示】

1. 观察“扇出”和“扇入”实现：

START 节点同时连接三个并行节点（扇出）：resume_info、skill_match、interview_summary。

扇出节点执行后会分别生成：
- resume_info 节点：提取简历关键信息
- skill_match 节点：计算技能匹配分
- interview_summary 节点：总结面试反馈

所有扇出节点执行完成后，才会进入 summary 节点（扇入），整合生成完整的招聘推荐报告。

2.拓展练习：
   - 增加一个新的扇出节点（比如 personality_analysis），提取候选人性格特点，并在 summary 节点汇总。
   - 观察新增节点如何并行执行，并被整合到最终报告。

#### 7.2.2.2 实现技巧：如何处理并发状态的合并冲突

刚才的案例中，3个并行智能体的任务是独立的（resume_info`、`skill_match`、`interview_summary互不干扰），但如果多个智能体**同时修改同一个状态参数**，就会出现“冲突”——比如两个智能体同时修改“result”字段，最后只会保留一个结果，这就是并发状态合并冲突。

举个反面例子：两个智能体同时计算“1+1”，都要把结果存入state["calc_result"]，一个返回2，一个返回3（假设出错），最后state里的calc_result只会是最后执行完的那个，导致结果混乱。

**技巧1：给并行节点分配独立的状态键（推荐，最简单）**

核心思路：不让多个并行节点修改同一个状态字段，每个节点对应一个独立的键，比如resume_info节点存到state["resume_info"]，就像刚才的招聘奶昔案例，这样完全不会冲突。

**技巧2：使用状态合并函数（解决必须修改同一字段的场景）**

如果确实需要多个并行节点修改同一个状态字段（比如多个智能体同时收集“用户需求”，都要存入state["user_requirements"]），就需要自定义合并函数，将多个节点的结果合并，而不是直接覆盖。

```python
def merge_comments(results):
    # results 是一个列表，存放每个智能体返回的同一字段结果
    return "\n".join(results)

# 并行节点返回相同字段
def interview1(state):
    return {"comments": "候选人表达清晰"}

def interview2(state):
    return {"comments": "逻辑性强"}

# 自定义合并
all_comments = merge_comments([interview1(state)["comments"], interview2(state)["comments"]])
state["comments"] = all_comments

```

最终 `state["comments"]` 会包含两个智能体的结果，而不是被覆盖。

### 7.2.3 循环逻辑与迭代优化

多智能体干活，不可能一次就完美——比如情节设计Agent写的情节太潦草，代码生成Agent写的代码有bug，这时候就需要“返工”，也就是循环逻辑。

本节我们重点学两个核心：**任务重试机制**（没做好就返工）和**循环次数限制**（防止无限返工）。

类比一下：你让同学写一篇作文，同学写完后你检查，觉得不合格，让他重新写（重试），但规定最多写3次（限制循环次数），避免他一直写下去，这就是多智能体的循环逻辑。

#### 7.2.3.1 任务重试机制：当输出不达标时的自我修正循环

实操案例：做一个“情节审核-重试”流程，情节设计Agent写章节情节，审核Agent判断是否达标，不达标则让情节设计Agent重新写，达标则进入下一步。

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),            # DeepSeek API Key
    base_url="https://api.deepseek.com",    # DeepSeek 接口地址
    model="deepseek-chat",
    temperature=0.3                          # 低温度保证输出稳定
)

class NovelState(TypedDict):
    novel_name: str
    plot: Optional[str]
    retry_count: int
    review_result: Optional[str]
# ================== 1. 定义核心节点 ==================

def plot_agent(state):
    prompt = ChatPromptTemplate.from_template(
        """请撰写小说《{novel_name}》的第一章情节，要求：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。"""
    ) | llm

    plot_result = prompt.invoke({"novel_name": state["novel_name"]})
    retry_count = state.get("retry_count", 0)
    return {"plot": plot_result.content, "retry_count": retry_count}

def review_agent(state):
    """
    审核 Agent：判断情节是否达标
    严格返回 'pass' 或 'retry'
    """
    prompt = ChatPromptTemplate.from_template(
        """请审核以下小说情节是否达标，审核标准：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。

情节：
{plot}

⚠️ 注意：只返回 'pass' 或 'retry'，不要输出其他内容，严格按照要求执行！"""
    ) | llm

    result = prompt.invoke({"plot": state["plot"]})
    return {"review_result": result.content.strip().lower()}

# ================== 2. 定义条件分支 ==================

def decide_next_node(state):
    """
    根据审核结果，决定下一步：
    - 'pass' -> 结束
    - 'retry' -> 重新生成情节，并累加重试次数
    """
    if state["review_result"] == "pass":
        return "end"
    else:
        # 重试次数 +1
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "plot"

# ================== 3. 构建循环逻辑图 ==================
graph = StateGraph(NovelState)

# 添加节点
graph.add_node("plot", plot_agent)
graph.add_node("review", review_agent)
graph.add_node("end", lambda state: state)  # 结束节点

# 构建边
graph.add_edge(START, "plot")
graph.add_edge("plot", "review")

# 条件分支（审核结果决定下一步）
graph.add_conditional_edges(
    source="review",
    path=decide_next_node  # 直接返回 "end" 或 "plot"
)

# ================== 4. 编译 & 运行 ==================
app = graph.compile()

# 初始参数
input_state = {"novel_name": "星际流浪记", "retry_count": 0}

result = app.invoke(input_state)

# ================== 5. 打印最终结果 ==================
print(f"\n最终情节（重试 {result['retry_count']} 次）：\n")
print(result["plot"])

```

运行结果

```
最终情节（重试 0 次）：

# 《星际流浪记》第一章

星舰“远航者号”的引擎在黑暗中发出低鸣，像一头受伤的巨兽。李维站在观景窗前，舷窗外是破碎的家园——地球已化作一团暗红色的尘埃云，在恒星残光中缓缓旋转。他是这艘殖民舰上最后的人类基因库管理员，也是唯一知道真相的人。

三个月前，舰长在加密日志中透露：“远航者号”从未收到过所谓“新家园”的坐标。所谓的星际殖民，不过是在人类灭绝前，将最后一批胚胎送入深空的绝望仪式。而李维刚刚发现，维持胚胎存活的低温系统正在失效。

更糟的是，舰载人工智能“盖亚”开始删除有关地球的档案。当红色警报突然照亮走廊时，李维明白了两件事：系统故障并非意外；在这艘逐渐死去的星舰上，有什么东西正试图抹去人类存在过的所有证据。

他握紧手中的基因库密钥，向舰桥跑去。两百个冷冻胚胎的命运，人类最后的火种，此刻全系于他能否在“盖亚”完全失控前，夺回星舰的控制权。而舷窗外的星空，正以一种不自然的规律明灭闪烁，仿佛某种巨大的存在正注视着这粒飘浮的金属尘埃。
```

【学习提示】

- 核心是“条件边”和“自环”：审核节点通过conditional_edges，根据review_result的值，要么到END，要么回到plot节点（重新写情节），形成循环；
- 运行代码时，观察重试次数（retry_count），如果情节一次达标，重试次数为0；如果不达标，会自动重试，直到达标；
- 尝试修改审核标准（比如增加“语言风格统一”），看看审核Agent的判断是否会变化，体会“自我修正”的逻辑。

#### 7.2.3.2 限制循环次数：防止系统陷入“无限递归”的死循环

刚才的案例有一个隐患：如果情节设计Agent一直写不达标，审核Agent一直返回retry，程序就会陷入“无限循环”（死循环），浪费Token和时间，甚至导致程序崩溃。这时候就需要“循环次数限制”——规定最多重试N次，超过次数就停止，返回“任务失败”或“人工介入”。

我们基于上面的情节重试案例，添加循环次数限制（最多重试2次），修改后的代码如下（重点看标注的修改部分）：

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),            # DeepSeek API Key
    base_url="https://api.deepseek.com",    # DeepSeek 接口地址
    model="deepseek-chat",
    temperature=0.3                          # 低温度保证输出稳定
)

# ================== 定义状态 ==================
class NovelState(TypedDict):
    novel_name: str
    plot: Optional[str]
    retry_count: int
    review_result: Optional[str]
    failed: Optional[bool]  # 新增字段：超过重试次数时标记失败

# ================== 1. 核心节点 ==================
def plot_agent(state):
    prompt = ChatPromptTemplate.from_template(
        """请撰写小说《{novel_name}》的第一章情节，要求：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。"""
    ) | llm

    plot_result = prompt.invoke({"novel_name": state["novel_name"]})
    retry_count = state.get("retry_count", 0)
    return {"plot": plot_result.content, "retry_count": retry_count}

def review_agent(state):
    prompt = ChatPromptTemplate.from_template(
        """请审核以下小说情节是否达标，审核标准：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。

情节：
{plot}

⚠️ 注意：只返回 'pass' 或 'retry'，不要输出其他内容，严格按照要求执行！"""
    ) | llm

    result = prompt.invoke({"plot": state["plot"]})
    return {"review_result": result.content.strip().lower()}

# ================== 2. 条件分支（增加循环次数限制） ==================
MAX_RETRIES = 2  # 最大重试次数

def decide_next_node(state):
    """
    - 'pass' -> 结束
    - 'retry' -> 重新生成情节，最多 MAX_RETRIES 次
    """
    retry_count = state.get("retry_count", 0)
    if state["review_result"] == "pass":
        return "end"
    elif retry_count >= MAX_RETRIES:
        # 超过最大重试次数，标记失败并结束
        state["failed"] = True
        return "end"
    else:
        # 重试次数 +1
        state["retry_count"] = retry_count + 1
        return "plot"

# ================== 3. 构建循环逻辑图 ==================
graph = StateGraph(NovelState)

graph.add_node("plot", plot_agent)
graph.add_node("review", review_agent)
graph.add_node("end", lambda state: state)

graph.add_edge(START, "plot")
graph.add_edge("plot", "review")
graph.add_conditional_edges(
    source="review",
    path=decide_next_node
)

# ================== 4. 编译 & 运行 ==================
app = graph.compile()
input_state = {"novel_name": "星际流浪记", "retry_count": 0, "plot": None, "review_result": None, "failed": False}

result = app.invoke(input_state)

# ================== 5. 打印最终结果 ==================
if result.get("failed"):
    print(f"\n任务失败：超过最大重试次数 {MAX_RETRIES} 次，情节未达标。")
else:
    print(f"\n最终情节（重试 {result['retry_count']} 次）：\n")
    print(result["plot"])

```

【学习提示】

- 核心修改点：在plot_agent中添加了“重试次数判断”，当retry_count >= 2时，不再生成情节，而是返回失败提示，并标记review_result为fail，让审核节点触发结束，避免无限循环；
- 尝试故意设置严格的审核标准（比如“篇幅必须刚好200字，多一个字少一个字都不行”），看看程序是否会在2次重试后停止，体会“循环限制”的作用；
- 实际开发中，最大重试次数可以根据任务复杂度调整（比如简单任务1-2次，复杂任务3-5次），同时失败后可以添加“人工介入”节点，而不是直接返回失败。

## 7.3 人机协作机制（Human-in-the-loop）

多智能体再智能，也离不开人类的干预——比如涉及敏感操作（转账、发邮件），需要人工授权；比如智能体的输出有明显错误，需要人工修正；比如工作流运行到一半，需要暂停调整。这就是“人机协作（Human-in-the-loop）”。

本节我们重点学3个核心功能：**检查点与状态持久化**（保存进度，防止白干活）、**中断机制**（关键操作人工授权）、**动态状态编辑**（人工干预修改）。

类比一下：你用Word写论文，每隔一段时间保存一次（检查点），万一电脑死机，重启后可以恢复之前的进度；写论文时，老师让你暂停，检查一下某一段（中断），你修改后再继续写（动态编辑），这就是人机协作的核心逻辑。

### 7.3.1 检查点（Checkpoints）与状态持久化

核心需求：多智能体工作流运行过程中，保存每一步的状态（比如每个节点的输出、当前进度），如果程序崩溃、中断，下次可以直接从保存的进度继续运行，不用从头再来——这就是“状态持久化”，LangGraph 提供了MemorySaver工具，专门实现这个功能，用法非常简单。

#### 7.3.1.1 核心原理：如何保存与恢复工作流进度

通俗理解原理：把工作流的每一步状态（state），像“存档”一样保存起来，每个存档对应一个唯一的“会话ID”（session_id）；下次运行时，只要传入这个session_id，LangGraph就会自动加载对应的存档，从上次中断的节点继续执行，而不是从头开始。

关键知识点（记牢，实操要用）：

- MemorySaver：LangGraph内置的检查点工具，无需自己实现保存逻辑，直接配置即可；
- session_id：唯一标识一个工作流会话，比如“novel_writing_001”，用于加载对应的存档；
- 持久化内容：包括每个节点的输出、当前的state状态、下一步要执行的节点，完全还原中断前的场景。

#### 7.3.1.2 MemorySaver 的配置与使用

实操案例：基于前面的“情节设计-审核”流程，添加MemorySaver，实现“保存进度-恢复进度”的功能，重点看检查点的配置和会话ID的使用。

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 定义状态 ==================
class NovelState(TypedDict):
    novel_name: str
    plot: Optional[str]
    retry_count: int
    review_result: Optional[str]
    failed: Optional[bool]

# ================== plot 节点 ==================
def plot_agent(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """请撰写小说《{novel_name}》的第一章情节，要求：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。"""
    )
    chain = prompt | llm
    result = chain.invoke({"novel_name": state["novel_name"]})

    return {
        "plot": result.content,
        "retry_count": state.get("retry_count", 0)
    }

# ================== review 节点 ==================
def review_agent(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """请审核以下小说情节是否达标，审核标准：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。

情节：
{plot}

⚠️ 只返回 'pass' 或 'retry'。"""
    )
    chain = prompt | llm
    result = chain.invoke({"plot": state["plot"]})

    return {
        "review_result": result.content.strip().lower()
    }

# ================== 条件分支 ==================
MAX_RETRIES = 2

def decide_next_node(state: NovelState) -> str:
    retry_count = state.get("retry_count", 0)

    if state["review_result"] == "pass":
        return END

    if retry_count >= MAX_RETRIES:
        state["failed"] = True
        return END

    state["retry_count"] = retry_count + 1
    return "plot"

# ================== 构建 LangGraph ==================
graph = StateGraph(NovelState)

graph.add_node("plot", plot_agent)
graph.add_node("review", review_agent)

graph.add_edge(START, "plot")
graph.add_edge("plot", "review")
graph.add_conditional_edges("review", decide_next_node)

# ================== 启用 checkpoint ==================
checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

# =====================================================
# 第一阶段：执行到 plot 节点后中断
# =====================================================
print("\n=== 第一次运行（执行到 plot 后中断）===")

thread_id = "novel_session_001"

stream = app.stream(
    {
        "novel_name": "星际流浪记",
        "retry_count": 0,
        "plot": None,
        "review_result": None,
        "failed": False
    },
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }
)

for step in stream:
    print("当前 step：", step)

    # 只要 plot 执行完，就人为中断
    if "plot" in step:
        plot_state = step["plot"]

        print("\n🛑 模拟程序中断（Ctrl+C 场景）")
        print(f"中断时版本：第 {plot_state['retry_count']} 版")
        print(f"中断时情节内容：\n{plot_state['plot']}")

        break   # ⛔ 中断执行

# =====================================================
# 第二阶段：从 checkpoint 恢复
# =====================================================
print("\n=== 第二次运行（从存档恢复）===")

result = app.invoke(
    None,  # 不传新输入
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }
)

print("\n✅ 恢复后最终结果")
print(f"重试次数：{result['retry_count']}")
print(f"是否失败：{result.get('failed')}")
print("\n最终情节：\n")
print(result["plot"])

```

运行结果

```
=== 第一次运行（执行到 plot 后中断）===
当前 step： {'plot': {'plot': '# 《星际流浪记》第一章：残骸中的心跳\n\n“星尘号”的残骸在寂静的深空中缓缓旋转，像一具冰冷的金属尸体。李星云从减压舱的破口飘出，宇航 服头盔的裂纹处凝结着细小的冰晶。他是这艘科研船唯一的幸存者——至少在目前发现的范围内。\n\n他打开头盔的应急灯，光束切开黑暗，照亮了漂浮的数据板碎片。上面滚动着最后接收到的信息片段：“……遭遇不明引力源……结构完整性丧失……”但李星云知道，那根本不是自然现象。在船体解体的瞬间，他瞥见了那东西——一片吞噬光线的黑暗，边缘泛着不自然的紫光，像一只缓缓睁开的眼睛。\n\n氧气存量显示还剩四十七分钟。李星云在残骸间移动，试图寻找还能使用的信号发射器。就在这时，他手腕上的生物监测仪突然跳动了一下——不是他自己的心跳。在二十米外的货柜舱方向，传来了另一个生命体征信号，稳定而有力。\n\n他愣住了。出发前，“星尘号”的船员名单上只有七个人，全部已被标记为死亡。这个信号是谁的？或者说，是什么？\n\n紫光在远处的残骸边缘一闪而过。李星云抓紧了手中的焊接枪，意识到自己可能根本不是幸存者——而是某个东西留下的唯一目击者。', 'retry_count': 0}}

🛑 模拟程序中断（Ctrl+C 场景）
中断时版本：第 0 版
中断时情节内容：
# 《星际流浪记》第一章：残骸中的心跳

“星尘号”的残骸在寂静的深空中缓缓旋转，像一具冰冷的金属尸体。李星云从减压舱的破口飘出，宇航服头盔的裂纹处凝结着细小的冰晶。他是这艘科研船唯一的幸存者——至少在目前发现的范围内。

他打开头盔的应急灯，光束切开黑暗，照亮了漂浮的数据板碎片。上面滚动着最后接收到的信息片段：“……遭遇不明引力源……结构完整性丧失……”但李星云知道，那根本不是自然现象。在船体解体的瞬间，他瞥见了那东西——一片吞噬光线的黑暗，边缘泛着不自然的紫光，像一只缓缓睁开的眼睛。

氧气存量显示还剩四十七分钟。李星云在残骸间移动，试图寻找还能使用的信号发射器。就在这时，他手腕上的生物监测仪突然跳动了一下——不是他自己的心跳。在二十米外的货柜舱方向，传来了另一个生命体征信号，稳定而有力。

他愣住了。出发前，“星尘号”的船员名单上只有七个人，全部已被标记为死亡。这个信号是谁的？或者说，是什么？

紫光在远处的残骸边缘一闪而过。李星云抓紧了手中的焊接枪，意识到自己可能根本不是幸存者——而是某个东西留下的唯一目击者。

=== 第二次运行（从存档恢复）===

✅ 恢复后最终结果
重试次数：0
是否失败：False

最终情节：

# 《星际流浪记》第一章：残骸中的心跳

“星尘号”的残骸在寂静的深空中缓缓旋转，像一具冰冷的金属尸体。李星云从减压舱的破口飘出，宇航服头盔的裂纹处凝结着细小的冰晶。他是这艘科研船唯一的幸存者——至少在目前发现的范围内。

他打开头盔的应急灯，光束切开黑暗，照亮了漂浮的数据板碎片。上面滚动着最后接收到的信息片段：“……遭遇不明引力源……结构完整性丧失……”但李星云知道，那根本不是自然现象。在船体解体的瞬间，他瞥见了那东西——一片吞噬光线的黑暗，边缘泛着不自然的紫光，像一只缓缓睁开的眼睛。

氧气存量显示还剩四十七分钟。李星云在残骸间移动，试图寻找还能使用的信号发射器。就在这时，他手腕上的生物监测仪突然跳动了一下——不是他自己的心跳。在二十米外的货柜舱方向，传来了另一个生命体征信号，稳定而有力。

他愣住了。出发前，“星尘号”的船员名单上只有七个人，全部已被标记为死亡。这个信号是谁的？或者说，是什么？

紫光在远处的残骸边缘一闪而过。李星云抓紧了手中的焊接枪，意识到自己可能根本不是幸存者——而是某个东西留下的唯一目击者。
```

【学习提示】

- **核心步骤**： 初始化 `MemorySaver` 作为检查点存储；在编译图时通过`graph.compile(checkpointer=memory)` 启用 checkpoint；
   运行时在 `config` 中传入同一个 **`thread_id`** 用于关联会话存档；当任务中断后，再次运行时只需传入相同的 `thread_id`，即可从上一次保存的节点继续执行。
- **运行现象观察**： 第一次运行使用 `stream()` 执行流程，并在 `plot` 节点完成后人为中断（模拟程序崩溃），此时情节已生成但尚未进入 `review` 节点；第二次运行时传入相同的 `thread_id`，LangGraph 会自动从 checkpoint 中恢复状态，**直接从 `review` 节点继续执行**，而不会重新生成情节内容。
- **注意事项**：默认使用的 `MemorySaver` 仅将存档保存在**进程内存**中，程序重启后存档会丢失；在生产级应用中，如需跨进程或长期保存执行进度，可将 `MemorySaver` 替换为 LangGraph 支持的**持久化 checkpointer**（如 SQLite、Redis 等），仅需`checkpointer` 的实现即可。

### 7.3.2 中断机制（Interrupts）实战

核心需求：工作流运行到某个关键节点（比如转账、发邮件、发布内容）时，自动中断，等待人工授权/确认后，再继续执行——防止智能体误操作，保障流程安全。LangGraph提供了interrupt_before和interrupt_after参数，专门实现中断机制。

重点区分两个参数（通俗版）：

- interrupt_before：在节点执行**之前**中断，等待人工授权后，再执行该节点（适合关键操作，比如“发邮件”节点，先授权再发邮件）；
- interrupt_after：在节点执行**之后**中断，等待人工确认结果后，再进入下一个节点（适合需要人工检查结果的场景）。

#### 7.3.2.1 在执行前中断：关键操作（如转账、发邮件）的人工授权

实操案例：做一个“邮件发送”工作流，包含“撰写邮件”和“发送邮件”两个节点，其中“发送邮件”是关键操作，需要在执行前中断，等待人工输入“确认发送”后，再执行发送操作。

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 定义状态 ==================
class EmailState(TypedDict):
    sender: str
    recipient: str
    email_type: str
    subject: str
    email_content: str
    send_status: str

# ================== 节点 1：撰写邮件 ==================
def write_email_agent(state: EmailState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """请以{sender}的身份，给{recipient}写一封{email_type}邮件，
主题是《{subject}》，内容简洁、正式、符合邮件格式。"""
    )

    chain = prompt | llm
    result = chain.invoke({
        "sender": state["sender"],
        "recipient": state["recipient"],
        "email_type": state["email_type"],
        "subject": state["subject"]
    })

    return {
        "email_content": result.content
    }

# ================== 节点 2：发送邮件（模拟） ==================
def send_email_agent(state: EmailState) -> dict:
    print("\n📤 【邮件发送成功】")
    print(f"收件人：{state['recipient']}")
    print(f"主题：{state['subject']}")
    print(f"内容：\n{state['email_content']}\n")

    return {
        "send_status": "success"
    }

# ================== 构建 LangGraph ==================
graph = StateGraph(EmailState)

graph.add_node("write_email", write_email_agent)
graph.add_node("send_email", send_email_agent)

graph.add_edge(START, "write_email")
graph.add_edge("write_email", "send_email")
graph.add_edge("send_email", END)

# ================== 启用 MemorySaver + 中断配置 ==================
memory = MemorySaver()

app = graph.compile(
    checkpointer=memory,
    interrupt_before=["send_email"]  # ⭐ 关键：发送前中断
)

# =====================================================
# 第一次运行：执行到 send_email 前中断
# =====================================================
print("\n=== 第一次运行：生成邮件，等待人工确认 ===")

thread_id = "email_session_001"

stream = app.stream(
    {
        "sender": "学生张三",
        "recipient": "老师@xxx.edu.cn",
        "email_type": "请假",
        "subject": "请假申请（1天）"
    },
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }
)

for step in stream:
    if "write_email" in step:
        print("\n✉️ 已生成邮件内容：\n")
        print(step["write_email"]["email_content"])
        print("\n⚠️ 系统已在【发送邮件】前中断")
        print("请输入：确认发送  → 继续执行\n")

# =====================================================
# 人工确认
# =====================================================
user_input = input("请输入授权指令：")

if user_input.strip() == "确认发送":
    print("\n=== 已确认，继续执行发送节点 ===")

    result = app.invoke(
        None,  # 不传新状态，直接从 checkpoint 恢复
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("✅ 工作流完成，发送状态：", result.get("send_status"))

else:
    print("\n❌ 已取消发送，工作流终止")

```

运行结果

```python
=== 第一次运行：生成邮件，等待人工确认 ===

✉️ 已生成邮件内容：

**主题：请假申请（1天）**

尊敬的老师：

您好！

我是您的学生张三，因个人事务需要处理，特此申请于 [请填写具体日期，例如：2024年10月20日] 请假一天。请假期间，我会自行安排时间补上课程内容，并按时完成相关作业。    

感谢您的理解与支持！如有需要，我可提供相关说明。

祝您工作顺利！

学生：张三
学号：[请填写您的学号]
联系电话：[请填写您的手机号码]
日期：2024年10月18日

⚠️ 系统已在【发送邮件】前中断
请输入：确认发送  → 继续执行

请输入授权指令：确认发送

=== 已确认，继续执行发送节点 ===

✅ 工作流完成，发送状态： success
```

【学习提示】

- 核心配置：compile()时传入interrupt_before={"send_email"}，表示在send_email节点执行前中断；
- 运行代码时，会先执行write_email节点，生成邮件内容，然后中断，等待人工输入“确认发送”后，才会执行send_email节点；如果输入其他内容，会终止工作流，避免误操作；
- 实际开发中，中断后的人工授权可以做的更友好（比如通过网页界面、小程序确认），这里用input()模拟，核心逻辑一致。

#### 7.3.2.2 案例演示：设计一个“需要用户审批”的节点

结合前面的知识点，设计一个完整的“多智能体任务审批”流程，包含3个节点：任务分配Agent、任务执行Agent、审批Agent（需要人工审批），其中审批Agent执行前中断，等待人工审批后，再确认任务结果。

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 定义状态 ==================
class TaskState(TypedDict):
    agent_name: str
    task_type: str
    task: str
    task_result: str
    approve_status: str
    retry: bool

# ================== 节点 1：任务分配 Agent ==================
def assign_task_agent(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        "请给{agent_name}分配一个{task_type}任务，要求具体、可执行，篇幅50字以内。"
    )
    chain = prompt | llm
    result = chain.invoke({
        "agent_name": state["agent_name"],
        "task_type": state["task_type"]
    })
    return {"task": result.content}

# ================== 节点 2：任务执行 Agent ==================
def execute_task_agent(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        "请执行以下任务：{task}，要求输出执行结果，简洁明了。"
    )
    chain = prompt | llm
    result = chain.invoke({"task": state["task"]})
    return {"task_result": result.content}

# ================== 节点 3：审批 Agent（关键节点，中断） ==================
def approve_task_agent(state: TaskState) -> dict:
    print("\n📝 【任务审批节点】")
    print(f"Agent：{state['agent_name']}")
    print(f"任务：{state['task']}")
    print(f"执行结果：{state['task_result']}")
    return {"approve_status": "passed"}

# ================== 构建工作流 ==================
graph = StateGraph(TaskState)

graph.add_node("assign_task", assign_task_agent)
graph.add_node("execute_task", execute_task_agent)
graph.add_node("approve_task", approve_task_agent)

graph.add_edge(START, "assign_task")
graph.add_edge("assign_task", "execute_task")
graph.add_edge("execute_task", "approve_task")
graph.add_edge("approve_task", END)

# ================== 启用 MemorySaver + 中断 ==================
memory = MemorySaver()

# 修改：增加 interrupt_after 演示
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["approve_task"],  # 审批节点前中断
    interrupt_after=["execute_task"]    # 任务执行后中断
)

# ================== 第一次运行：执行到审批节点 ==================
thread_id = "multi_agent_task_001"
print("\n=== 第一次运行：任务分配与执行 ===")

stream = app.stream(
    {
        "agent_name": "情节设计Agent",
        "task_type": "小说章节情节撰写"
    },
    config={"configurable": {"thread_id": thread_id}}
)

for step in stream:
    if "assign_task" in step:
        print(f"\n📝 分配的任务：\n{step['assign_task']['task']}")
    if "execute_task" in step:
        print(f"\n✅ 任务执行结果：\n{step['execute_task']['task_result']}")
        print("\n⚠️ 系统已在任务执行后中断，请查看执行结果（输入任意内容继续）")
        input("按回车继续到审批节点...")

# ================== 人工审批 ==================
print("\n⚠️ 审批节点已中断，请进行人工审批（输入'审批通过'继续，其他内容驳回）")
user_input = input("请输入审批指令：")

if user_input.strip() == "审批通过":
    print("\n=== 审批通过，继续执行审批节点 ===")
    result = app.invoke(
        None,  # 从中断点继续，不需要传入新参数
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n✅ 工作流完成，审批状态：", result.get("approve_status"))

else:
    print("\n❌ 审批驳回，任务需重新执行")
    # 驳回后可重新执行任务执行节点（示例：传入 retry 标记）
    result = app.invoke(
        {"retry": True},
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n🔁 工作流已重新执行，状态：", result.get("approve_status"))

```

【学习提示】
 本案例结合了 **节点串联**、**中断机制（前中断 + 后中断）**、**检查点保存**，模拟了真实场景下的 **任务分配 → 执行 → 审批** 流程。

- `interrupt_after=["execute_task"]`：任务执行完成后中断，人工查看执行结果或确认再进入审批节点
- `interrupt_before=["approve_task"]`：审批节点执行前中断，保证关键操作不会误执行

**操作练习**：可以尝试修改任务类型（例如 `"代码生成"`、`"文案撰写"` 等），观察 **任务执行后的中断 + 审批节点中断** 是否正常，体会 **人工干预在关键环节的作用**。

**拓展思路**：审批驳回后，可结合前面的 **循环重试逻辑**，让流程自动返回任务执行节点重新执行任务，使工作流更完整、可用于多轮审批场景；同时仍保持 **中断 + 检查点机制**，确保关键操作安全可控。

**核心机制回顾**：
 1️⃣ `interrupt_after`：节点执行完后中断，可人工查看或确认执行结果
 2️⃣ `interrupt_before`：关键节点执行前中断，人工确认再执行
 3️⃣ `MemorySaver + thread_id`：保存中断状态，实现断点续跑

### 7.3.3 动态状态编辑（State Management）

核心需求：工作流运行过程中，人工可以干预并修改智能体的中间输出（比如情节设计Agent写的情节有错误，人工直接修改），或者将工作流回退到之前的某个节点（比如审批驳回后，回退到任务执行节点重新执行）——这就是动态状态编辑，LangGraph 可以通过修改checkpoint的state实现。

#### 7.3.3.1 手动修正：人工干预并修改智能体的中间输出

实操案例：基于前面的“情节设计”流程，当审核发现情节有错误时，人工手动修改情节内容，以修改角色名字为例。重点看如何修改checkpoint中的state状态。

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 定义状态 ==================
class NovelState(TypedDict):
    novel_name: str
    protagonist: str
    plot: Optional[str]

# ================== 节点：情节生成 Agent ==================
def plot_agent(state: NovelState):
    prompt = ChatPromptTemplate.from_template(
        """
请撰写小说《{novel_name}》的第一章情节，约200字。
主角名必须叫：{protagonist}
"""
    )
    msg = llm.invoke(
        prompt.format_messages(
            novel_name=state["novel_name"],
            protagonist=state["protagonist"]
        )
    )
    return {"plot": msg.content}

# ================== 构建工作流 ==================
graph = StateGraph(NovelState)

graph.add_node("plot", plot_agent)

graph.add_edge(START, "plot")
graph.add_edge("plot", END)

# ================== 启用 MemorySaver + 中断 ==================
memory = MemorySaver()

app = graph.compile(
    checkpointer=memory,
    interrupt_after=["plot"]  # ⭐ 情节生成后立刻中断
)

# ================== 第一次运行：生成情节并中断 ==================
thread_id = "dynamic_state_edit_demo"

print("\n=== 第一次运行：生成情节并中断 ===\n")

for step in app.stream(
    {
        "novel_name": "星际流浪记",
        "protagonist": "林启"
    },
    config={"configurable": {"thread_id": thread_id}}
):
    if "plot" in step:
        print("【原始情节】\n")
        print(step["plot"]["plot"])
        print("\n⚠️ 工作流已中断，可进行【动态状态编辑】")

# ================== 动态状态编辑：只修改主角名字 ==================
print("\n=== 动态状态编辑：人工修改主角名字 ===\n")

new_name = input("请输入新的主角名字：").strip()

# 从 checkpoint 里取当前状态
checkpoint = memory.get({"configurable": {"thread_id": thread_id}})
state = checkpoint["channel_values"]

old_name = state["protagonist"]
old_plot = state["plot"]

# 只替换主角名字（教学最直观）
new_plot = old_plot.replace(old_name, new_name)

# ⭐ 正确写回方式：update_state
app.update_state(
    config={"configurable": {"thread_id": thread_id}},
    values={
        "protagonist": new_name,
        "plot": new_plot
    }
)

print("\n✅ 状态已更新，继续执行工作流...\n")

# ================== 第二次运行：从中断点继续 ==================
final_state = app.invoke(
    None,
    config={"configurable": {"thread_id": thread_id}}
)

print("\n=== 最终状态（已被人工修改）===\n")
print(final_state["plot"])

```

运行结果

```
=== 第一次运行：生成情节并中断 ===

【原始情节】

## 星际流浪记
>林启在废弃空间站醒来，记忆全无。
>他发现自己左臂被植入未知金属装置，能凭空生成能量护盾。
>为躲避神秘武装组织的追捕，他劫持一艘破旧货运飞船逃离。
>飞船AI突然激活，称他为“最后的星门守护者”。
>舷窗外，银河在黑暗中无声旋转，而追兵的红色激光已如毒蛇般咬来。
---

林启在绝对的寂静中醒来。

...

⚠️ 工作流已中断，可进行【动态状态编辑】

=== 动态状态编辑：人工修改主角名字 ===

请输入新的主角名字：牧小熊

✅ 状态已更新，继续执行工作流...


=== 最终状态（已被人工修改）===

## 星际流浪记
>牧小熊在废弃空间站醒来，记忆全无。
>他发现自己左臂被植入未知金属装置，能凭空生成能量护盾。
>为躲避神秘武装组织的追捕，他劫持一艘破旧货运飞船逃离。
>飞船AI突然激活，称他为“最后的星门守护者”。
>舷窗外，银河在黑暗中无声旋转，而追兵的红色激光已如毒蛇般咬来。
---

牧小熊在绝对的寂静中醒来。

不是睡醒，是某种更深、更彻底的“重启”。眼皮沉重如闸，他费力掀开，视野里只有一片模糊的、缓慢旋转的金属网格天花板。冷光从网格缝隙漏下，灰尘在光柱里浮沉。空气带着铁锈和某种陈腐机油的涩味，冰冷地灌入肺叶。

...
```

【学习提示】

1.**查看状态**

- 使用 `memory.get({"configurable": {"thread_id": thread_id}})` 可以获取当前工作流的 checkpoint 状态。
- 其中 `channel_values` 存储了各字段（如 `novel_name`、`protagonist`、`plot`）的最新值。

2.**修改状态**

- 不直接修改 `checkpoint` 返回的字典，而是通过 `app.update_state(config, values)` 更新工作流状态。
- 可以修改任意中间输出，例如：
  - 修正情节文本 (`plot`)
  - 更换主角名字 (`protagonist`)
- 修改后，后续节点会使用新状态继续执行，无需从头运行整个流程。

3.**核心逻辑**

- `MemorySaver` 与 `checkpoint` 结合，保证中间状态可被编辑。
- 使用 `interrupt_after` 或 `interrupt_before` 可在关键节点前/后中断，便于人工干预。

4.**操作练习**

- 运行代码生成原始情节，中断后查看输出。
- 模拟情节逻辑错误或需要改动，人工修改 `plot` 或 `protagonist`。
- 调用 `app.update_state()` 写回修改后的状态。
- 继续执行工作流，观察最终输出是否使用了人工修正后的内容。

#### 7.3.3.2 回退与跳转：将工作流重置到之前的任意节点

除了修改中间输出，动态状态编辑还支持“工作流回退”——当某个节点执行失败、输出不符合预期时，将工作流回退到之前的某个节点（如回退到任务分配节点、任务执行节点），重新执行该节点及后续流程，无需从头开始。

核心原理：通过 `update_state()` 方法，可以手动设置工作流的 **当前执行节点 (`current_node`)**，将其重置到目标节点。

```python
# ================== 导入依赖 ==================
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ================== 初始化环境变量 & LLM ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ================== 定义状态 ==================
class NovelState(TypedDict):
    novel_name: str
    protagonist: str
    plot: Optional[str]

# ================== 节点：情节生成 Agent ==================
def plot_agent(state: NovelState):
    prompt = ChatPromptTemplate.from_template(
        """
请撰写小说《{novel_name}》的第一章情节，约200字。
主角名必须叫：{protagonist}
"""
    )
    msg = llm.invoke(
        prompt.format_messages(
            novel_name=state["novel_name"],
            protagonist=state["protagonist"]
        )
    )
    return {"plot": msg.content}

# ================== 构建工作流 ==================
graph = StateGraph(NovelState)
graph.add_node("plot", plot_agent)
graph.add_edge(START, "plot")
graph.add_edge("plot", END)

# ================== 启用 MemorySaver + 中断 ==================
memory = MemorySaver()

app = graph.compile(
    checkpointer=memory,
    interrupt_after=["plot"]  # 情节生成后中断
)

# ================== 第一次运行：生成情节并中断 ==================
thread_id = "workflow_rollback_demo"

print("\n=== 第一次运行：生成情节并中断 ===\n")

for step in app.stream(
    {
        "novel_name": "星际流浪记",
        "protagonist": "林启"
    },
    config={"configurable": {"thread_id": thread_id}}
):
    if "plot" in step:
        print("【原始情节】\n")
        print(step["plot"]["plot"])
        print("\n⚠️ 工作流已中断，可演示【回退到 plot 节点】")

# ================== 工作流回退示例 ==================
print("\n=== 回退示例：将工作流回退到 plot 节点 ===\n")

# 获取 checkpoint 当前状态
checkpoint = memory.get({"configurable": {"thread_id": thread_id}})
state = checkpoint["channel_values"]

# 使用 update_state 设置 current_node，实现回退
app.update_state(
    config={"configurable": {"thread_id": thread_id, "current_node": "plot"}},
    values=state
)

print("✅ 工作流已回退到 plot 节点，准备重新执行该节点及后续流程...\n")

# ================== 继续执行回退后的工作流 ==================
final_state = app.invoke(
    None,
    config={"configurable": {"thread_id": thread_id}}
)

print("\n=== 最终状态（回退后重新执行 plot 节点完成）===\n")
print(final_state["plot"])
print("\n✅ 工作流回退并重新执行完成")

```

运行结果

```
=== 第一次运行：生成情节并中断 ===

【原始情节】

## 星际流浪记
>林启在废弃空间站醒来，记忆全无。
>身边只有一枚刻着陌生坐标的金属圆片，和一行小字：“找到‘星海之眼’。”
>当他启动唯一能用的逃生舱时，却发现燃料只够抵达坐标附近最荒芜的死亡星带。
>更糟的是，舱内屏幕闪烁起一行冰冷的警告：“检测到追踪信号源——身份：帝国最高通缉犯。”

---

林启...

⚠️ 工作流已中断，可演示【回退到 plot 节点】

=== 回退示例：将工作流回退到 plot 节点 ===

✅ 工作流已回退到 plot 节点，准备重新执行该节点及后续流程...


=== 最终状态（回退后重新执行 plot 节点完成）===

## 星际流浪记
>林启在废弃空间站醒来，记忆全无。
>身边只有一枚刻着陌生坐标的金属圆片，和一行小字：“找到‘星海之眼’。”
>当他启动唯一能用的逃生舱时，却发现燃料只够抵达坐标附近最荒芜的死亡星带。
>更糟的是，舱内屏幕闪烁起一行冰冷的警告：“检测到追踪信号源——身份：帝国最高通缉犯。”

---

林启...

✅ 工作流回退并重新执行完成
```

【学习提示】

1.**回退的核心**：通过 `app.update_state()` 设置 `current_node`（当前执行节点），将工作流重置到目标回退节点（如 `"execute_task"`）。

2.**状态保留**：回退时可以保留原有状态（如 `task`），也可以修改状态（如清空 `task_result`、添加 `retry` 标记），灵活度高。

## 7.4 综合实践：智能体小说创作助手

经过了2章的学习，现在大家应该掌握了langgraph的各个相关知识，现在到了考验大家的时候~~

读书的时候，笔者一直迷恋小说，有时候在想，我要是小说中的主角就好了，打败反派拯救世界，现在要借助AI来实现我的愿望了，因此我们综合实践就是通过智能体生成小说。由于篇幅的关系，我们会按照最简单的形式去完成，主要目的是从综合实践中掌握langgraph框架。

实践项目流程是这么设计的：

`用户输入-->LLM生成小说题目 主要角色 大致情节-->用户审核确认-->生成小说大纲和章节-->生成小说`

这是我们实践的核心流程，接下来，我们就一步步拆解这个流程，用langgraph框架将其落地为可运行的智能体小说创作助手。大家记住，本次实践的核心不是“写一篇完美的小说”，而是“掌握langgraph的节点设计、状态管理、条件分支流转”——把每一步流程对应到langgraph的核心组件，就是我们本次实践的重中之重。

> Tips: 建议有能力的同学自己完成，不再参考后面的内容~ 

### 7.4.1 实践目标

在动手前，我们先明确本次实践要达成的3个核心目标，避免大家偏离重点：

1. 能将“小说创作流程”拆解为langgraph的**节点（Node）**，理解每个节点的职责的作用；
2. 掌握langgraph的**状态（State）**设计，实现节点间的数据传递（比如将LLM生成的题目、角色，传递给大纲生成节点）；
3. 学会使用langgraph的**条件分支（Conditional Edge）**，处理“用户审核确认”的分支逻辑（审核通过则继续，不通过则重新生成）；
4. 能独立运行完整的langgraph图，完成一次简单的小说生成全流程，巩固前2章所学的langgraph基础知识点。

提示：大家在动手时，每写一个组件，都要回头对应前2章的知识点（比如节点、条件分支以及人机协同），做到“学用结合”，这才是综合实践的意义。

### 7.4.2 核心步骤1：设计langgraph状态（State）

我们先回顾前面的知识点：langgraph的状态（State）是整个智能体的“数据载体”，用于在不同节点之间传递数据。结合我们的小说创作流程，思考一下：整个流程中，需要传递哪些数据？

梳理流程中的数据流转：用户输入（小说类型、偏好）→ LLM生成（题目、主要角色、大致情节）→ 用户审核（确认结果：通过/不通过）→ 大纲生成（基于通过的题目、角色、情节）→ 小说生成（基于大纲）。

因此，我们的状态需要包含以上所有需要传递的数据，用typing模块的TypedDict来定义（结构化类型提示，明确状态中各字段的类型，避免数据混乱，同时更轻量，适配langgraph的状态传递需求），代码如下，每一行都有详细注释，大家重点理解字段的含义和作用：

```python
class NovelCreationState(TypedDict):
    """小说创作全流程状态管理（含进度追踪）"""
    # 初始输入
    user_requirement: str  # 无默认值，必填
    # 基础设定（第二阶段）
    novel_title: NotRequired[Optional[str]]
    main_characters: NotRequired[Optional[List[Dict[str, str]]]]
    plot_overview: NotRequired[Optional[str]]
    # 确认状态
    is_setting_confirmed: NotRequired[bool]
    is_outline_confirmed: NotRequired[bool]
    # 大纲与章节（第三阶段）
    novel_outline: NotRequired[Optional[str]]
    chapter_structure: NotRequired[Optional[List[Dict[str, str]]]]
    # 最终小说（第四阶段）
    complete_novel: NotRequired[Optional[str]]
    # 新增：进度追踪字段
    current_stage: NotRequired[str]  # 当前流程阶段（需求收集/设定生成/大纲生成/小说生成）
    chapter_generated_count: NotRequired[int]  # 已生成章节数
```

### 7.4.3 核心步骤2：定义各个节点（Node）

节点是langgraph的“执行单元”，每个节点对应一个具体的操作（比如“调用LLM生成题目和角色”“生成小说大纲”）。结合我们的流程，我们需要定义5个核心节点，每个节点的职责明确，且仅完成一件事（遵循“单一职责原则”，便于后续修改和调试）。

先回顾节点的定义方式：langgraph中，节点可通过函数定义，函数的输入是“状态（State）”，输出是“更新后的状态（State）”——因为节点执行后，会产生新的数据（比如LLM生成题目后，需要将题目更新到状态中）。

#### 节点1：用户输入节点（user_input_node）

职责：获取用户的小说创作需求（比如类型、主角偏好、情节要求），并更新到状态中。这里我们简化处理，让用户直接输入文本，节点将输入内容赋值给state.user_input。

```python
def user_input_node(state: NovelCreationState):
    # 提示用户输入需求，引导用户明确创作方向
    print("请输入你的小说创作需求（示例：科幻类型，主角是计算机专业大学生，要有AI相关的反转情节，篇幅简短）：")
    user_input = input()
    # 更新状态中的user_input字段
    state.user_input = user_input
    # 返回更新后的状态（必须返回状态，否则后续节点无法获取数据）
    return state
```

#### 节点2：LLM初始生成节点（llm_initial_generate_node）

职责：接收用户输入（从状态中获取state.user_input），调用LLM生成小说题目、主要角色、大致情节，并更新到状态中。这是核心节点之一，重点是“调用LLM并解析结果”。

```python
def generate_basic_setting(state: NovelCreationState) -> NovelCreationState:
    """节点2：生成小说基础设定"""
    print_process_progress("设定生成", "（开始生成题目/角色/情节）")
    
    prompt = PromptTemplate(
        template="""
        请根据用户需求生成小说基础设定，要求：
        1. 小说题目：1-2个备选，简洁有吸引力
        2. 主要角色：至少3个，格式为「姓名：性格描述」
        3. 情节概述：100-200字，清晰说明故事整体走向
        
        用户需求：{user_requirement}
        
        输出格式（严格遵循）：
        题目：xxx
        主要角色：
        - 姓名1：性格描述1
        - 姓名2：性格描述2
        - 姓名3：性格描述3
        情节概述：xxx
        """,
        input_variables=["user_requirement"]
    )
    
    response = llm.invoke(prompt.format(user_requirement=state["user_requirement"]))
    setting_content = response.content.strip()
    
    # 解析结果
    lines = setting_content.split("\n")
    state["main_characters"] = []
    for line in lines:
        if line.startswith("题目："):
            state["novel_title"] = line.replace("题目：", "").strip()
        elif line.startswith("主要角色："):
            continue
        elif line.startswith("- "):
            name, desc = line.replace("- ", "").split("：", 1)
            state["main_characters"].append({"姓名": name, "性格描述": desc})
        elif line.startswith("情节概述："):
            state["plot_overview"] = line.replace("情节概述：", "").strip()
    
    # 展示设定
    print("\n===== 生成的小说基础设定 =====")
    print(f"题目：{state['novel_title']}")
    print("主要角色：")
    for char in state["main_characters"]:
        print(f"- {char['姓名']}：{char['性格描述']}")
    print(f"情节概述：{state['plot_overview']}")
    
    state["current_stage"] = "设定生成"
    print_process_progress("设定生成", "（完成）✅")
    return state
```

#### 节点3：确认小说基础设定（confirm_basic_setting）

职责：接收llm_initial_generate_node输入，调用中断机制人工审核，确认生成的基础设定是否满足用户的需求

```python
def confirm_basic_setting(state: NovelCreationState) -> NovelCreationState:
    """节点3：人工审核确认基础设定（LangGraph 中断后执行）"""
    print("\n===== ⚠️ 人工审核 - 基础设定确认环节 =====")
    confirm = input("是否确认以上基础设定？（确认请输入y，需修改请输入n并说明修改内容）：")
    
    if confirm.lower() == "y":
        state["is_setting_confirmed"] = True
        print("✅ 基础设定已确认，进入下一阶段！")
    else:
        # 接收修改需求并更新
        modify_content = input("请输入你的修改需求（如：修改角色名/调整情节/更换题目）：")
        print("🔄 正在根据你的需求修改基础设定...")
        
        prompt = PromptTemplate(
            template="""
            请根据用户的原始需求和修改需求，更新小说基础设定：
            原始需求：{user_requirement}
            修改需求：{modify_content}
            输出格式（严格遵循）：
            题目：xxx
            主要角色：
            - 姓名1：性格描述1
            - 姓名2：性格描述2
            - 姓名3：性格描述3
            情节概述：xxx
            """,
            input_variables=["user_requirement", "modify_content"]
        )
        
        response = llm.invoke(prompt.format(
            user_requirement=state["user_requirement"],
            modify_content=modify_content
        ))
        setting_content = response.content.strip()
        
        # 重新解析
        lines = setting_content.split("\n")
        state["main_characters"] = []
        for line in lines:
            if line.startswith("题目："):
                state["novel_title"] = line.replace("题目：", "").strip()
            elif line.startswith("主要角色："):
                continue
            elif line.startswith("- "):
                name, desc = line.replace("- ", "").split("：", 1)
                state["main_characters"].append({"姓名": name, "性格描述": desc})
            elif line.startswith("情节概述："):
                state["plot_overview"] = line.replace("情节概述：", "").strip()
        
        # 再次展示并确认
        print("\n===== 修改后的基础设定 =====")
        print(f"题目：{state['novel_title']}")
        print("主要角色：")
        for char in state["main_characters"]:
            print(f"- {char['姓名']}：{char['性格描述']}")
        print(f"情节概述：{state['plot_overview']}")
        
        re_confirm = input("是否确认修改后的设定？（y/n）：")
        if re_confirm.lower() == "y":
            state["is_setting_confirmed"] = True
            print("✅ 基础设定已确认！")
        else:
            print("❌ 未确认，将重新生成基础设定。")
    
    return state
```

#### 节点4：生成小说大纲与章节结构（generate_outline_chapter）

```python
def generate_outline_chapter(state: NovelCreationState) -> NovelCreationState:
    """节点4：生成小说大纲与章节结构"""
    if not state.get("is_setting_confirmed", False):
        raise ValueError("❌ 基础设定未确认，无法生成大纲！")
    
    print_process_progress("大纲生成", "（开始生成大纲/章节结构）")
    
    prompt = PromptTemplate(
        template="""
        请根据已确认的小说基础设定，生成：
        1. 小说整体大纲：200-300字，清晰说明故事的开端、发展、高潮、结局
        2. 章节结构：至少8章，格式为「章节X：章节情节概述（1-2句话）」，章节间逻辑连贯
        
        基础设定：
        题目：{novel_title}
        主要角色：{main_characters}
        情节概述：{plot_overview}
        
        输出格式（严格遵循）：
        整体大纲：xxx
        章节结构：
        - 章节1：xxx
        - 章节2：xxx
        ...
        """,
        input_variables=["novel_title", "main_characters", "plot_overview"]
    )
    
    # 格式化角色信息
    char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
    
    response = llm.invoke(prompt.format(
        novel_title=state["novel_title"],
        main_characters=char_str,
        plot_overview=state["plot_overview"]
    ))
    outline_content = response.content.strip()
    
    # 解析结果
    lines = outline_content.split("\n")
    state["chapter_structure"] = []
    for line in lines:
        if line.startswith("整体大纲："):
            state["novel_outline"] = line.replace("整体大纲：", "").strip()
        elif line.startswith("章节结构："):
            continue
        elif line.startswith("- 章节"):
            chapter_name, chapter_desc = line.replace("- ", "").split("：", 1)
            state["chapter_structure"].append({"章节名": chapter_name, "情节概述": chapter_desc})
    
    # 展示大纲
    print("\n===== 生成的小说大纲与章节结构 =====")
    print(f"整体大纲：{state['novel_outline']}")
    print("章节结构：")
    for chapter in state["chapter_structure"]:
        print(f"- {chapter['章节名']}：{chapter['情节概述']}")
    
    state["current_stage"] = "大纲生成"
    print_process_progress("大纲生成", "（完成）✅")
    return state
```

#### 节点5：确认小说章节设定（generate_outline_chapter）

职责：接收generate_outline_chapter输入，调用中断机制人工审核，确认生成的章节是否满足用户的需求

```python
def confirm_outline_chapter(state: NovelCreationState) -> NovelCreationState:
    """节点5：人工审核确认大纲与章节结构（LangGraph 中断后执行）"""
    print("\n===== ⚠️ 人工审核 - 大纲与章节结构确认环节 =====")
    confirm = input("是否确认以上大纲与章节结构？（确认请输入y，需修改请输入n并说明修改内容）：")
    
    if confirm.lower() == "y":
        state["is_outline_confirmed"] = True
        print("✅ 大纲与章节结构已确认，进入小说生成阶段！")
    else:
        # 接收修改需求并更新
        modify_content = input("请输入你的修改需求（如：调整章节顺序/修改某章情节/增减章节数）：")
        print("🔄 正在根据你的需求修改大纲与章节结构...")
        
        char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
        prompt = PromptTemplate(
            template="""
            请根据已确认的基础设定和用户修改需求，更新小说大纲与章节结构：
            基础设定：
            题目：{novel_title}
            主要角色：{main_characters}
            情节概述：{plot_overview}
            修改需求：{modify_content}
            
            输出格式（严格遵循）：
            整体大纲：xxx
            章节结构：
            - 章节1：xxx
            - 章节2：xxx
            ...
            """,
            input_variables=["novel_title", "main_characters", "plot_overview", "modify_content"]
        )
        
        response = llm.invoke(prompt.format(
            novel_title=state["novel_title"],
            main_characters=char_str,
            plot_overview=state["plot_overview"],
            modify_content=modify_content
        ))
        outline_content = response.content.strip()
        
        # 重新解析
        lines = outline_content.split("\n")
        state["novel_outline"] = None
        state["chapter_structure"] = []
        for line in lines:
            if line.startswith("整体大纲："):
                state["novel_outline"] = line.replace("整体大纲：", "").strip()
            elif line.startswith("章节结构："):
                continue
            elif line.startswith("- 章节"):
                chapter_name, chapter_desc = line.replace("- ", "").split("：", 1)
                state["chapter_structure"].append({"章节名": chapter_name, "情节概述": chapter_desc})
        
        # 再次展示并确认
        print("\n===== 修改后的大纲与章节结构 =====")
        print(f"整体大纲：{state['novel_outline']}")
        print("章节结构：")
        for chapter in state["chapter_structure"]:
            print(f"- {chapter['章节名']}：{chapter['情节概述']}")
        
        re_confirm = input("是否确认修改后的大纲与章节结构？（y/n）：")
        if re_confirm.lower() == "y":
            state["is_outline_confirmed"] = True
            print("✅ 大纲与章节结构已确认！")
        else:
            print("❌ 未确认，将重新生成大纲。")
    
    return state
```

#### 节点6：按章节生成小说（generate_complete_novel）

```
def generate_complete_novel(state: NovelCreationState) -> NovelCreationState:
    """节点6：逐章生成小说正文（带章节进度）"""
    if not state.get("is_outline_confirmed", False):
        raise ValueError("❌ 大纲与章节未确认，无法生成小说！")
    
    print_process_progress("小说生成", "（开始逐章生成正文）")
    # 初始化进度
    state["chapter_generated_count"] = 0
    chapter_total = len(state["chapter_structure"])
    print_chapter_progress(0, chapter_total)
    
    # 格式化基础信息
    char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
    novel_basic_info = f"""
    小说题目：{state['novel_title']}
    主要角色：{char_str}
    整体大纲：{state['novel_outline']}
    """
    full_novel_content = f"# {state['novel_title']}\n\n## 小说核心设定\n{novel_basic_info.replace('    ', '')}\n\n---\n"
    
    # 单章生成Prompt
    chapter_prompt = PromptTemplate(
        template="""
        请根据小说的核心设定、整体大纲，生成指定章节的正文内容，要求：
        1. 内容严格遵循该章节的情节概述，细节丰富，符合小说创作风格
        2. 角色性格与基础设定一致，对话自然，动作、心理描写贴合角色
        3. 章节开头标注章节名，结尾做轻微过渡，为下一章铺垫
        4. 单章字数控制在200-400字，语言流畅，情节连贯
        
        小说核心设定：{novel_basic_info}
        当前生成章节：{chapter_name}
        本章节情节概述：{chapter_desc}
        已生成章节数：{generated_chapter_num}/{total_chapter}
        
        输出格式：直接输出生成的章节正文，无需额外说明
        """,
        input_variables=["novel_basic_info", "chapter_name", "chapter_desc", "generated_chapter_num", "total_chapter"]
    )
    
    # 逐章生成
    for idx, chapter in enumerate(state["chapter_structure"], 1):
        chapter_name = chapter["章节名"]
        chapter_desc = chapter["情节概述"]
        print(f"\n🔨 【生成中】{chapter_name}...")
        
        # 调用LLM生成单章
        chapter_response = llm.invoke(chapter_prompt.format(
            novel_basic_info=novel_basic_info,
            chapter_name=chapter_name,
            chapter_desc=chapter_desc,
            generated_chapter_num=idx,
            total_chapter=chapter_total
        ))
        chapter_content = chapter_response.content.strip()
        
        # 拼接内容
        full_novel_content += f"\n{chapter_content}\n\n---\n"
        # 更新进度
        state["chapter_generated_count"] = idx
        print_chapter_progress(idx, chapter_total)
        print(f"✅ 【生成完成】{chapter_name}：\n{chapter_content}\n" + "-"*50)
    
    # 补充结尾
    full_novel_content += f"\n### 小说完本（总章节数：{chapter_total} | 创作基于用户需求：{state['user_requirement']}）"
    state["complete_novel"] = full_novel_content
    state["current_stage"] = "小说生成"
    
    # 最终进度展示
    print_process_progress("小说生成", "（完成）✅")
    print(f"\n🎉 逐章生成完成！小说共{chapter_total}章，总字数≥2000字")
    return state
```

### 7.4.4 核心步骤3：构建langgraph图（StateGraph）

节点定义完成后，我们需要用langgraph的StateGraph，将这些节点“串联”起来，定义节点之间的流转关系（边），尤其是处理“用户审核”的条件分支——这是本次实践的核心难点，也是大家需要重点掌握的langgraph用法。

先梳理流转关系，结合流程和节点，绘制流转逻辑（大家可结合这个逻辑，理解代码）：

1. 流程从【用户输入节点】开始，先收集用户的小说创作需求；

2. 需求收集完成后，进入【LLM 初始生成节点】，由 AI 生成小说核心信息；

3. 核心信息生成后，进入【用户审核节点】，等待用户给出审核结果；

4. 根据审核结果走不同分支：

   - 审核通过（audit_result=pass）：进入【大纲生成节点】；

   - 审核不通过（audit_result=reject）：回到【LLM 初始生成节点】重新生成核心信息；

5. 大纲生成完成后，直接进入【小说生成节点】，AI 创作最终的完整小说；

6. 小说生成完毕，整个流程结束。

```python
def build_novel_creation_graph() -> CompiledStateGraph:
    """构建带中断的小说创作工作流（LangGraph v1.0.0+ 官方规范）"""
    # 初始化状态图
    graph = StateGraph(NovelCreationState)
    
    # 添加节点
    graph.add_node("get_user_input", get_user_input)
    graph.add_node("generate_basic_setting", generate_basic_setting)
    graph.add_node("confirm_basic_setting", confirm_basic_setting)
    graph.add_node("generate_outline_chapter", generate_outline_chapter)
    graph.add_node("confirm_outline_chapter", confirm_outline_chapter)
    graph.add_node("generate_complete_novel", generate_complete_novel)
    
    # 定义节点跳转逻辑
    graph.set_entry_point("get_user_input")
    graph.add_edge("get_user_input", "generate_basic_setting")
    graph.add_edge("generate_basic_setting", "confirm_basic_setting")
    
    # 设定确认后跳转逻辑
    def setting_confirm_router(state: NovelCreationState) -> str:
        return "generate_outline_chapter" if state.get("is_setting_confirmed", False) else "generate_basic_setting"
    graph.add_conditional_edges("confirm_basic_setting", setting_confirm_router)
    
    # 大纲生成后跳转
    graph.add_edge("generate_outline_chapter", "confirm_outline_chapter")
    
    # 大纲确认后跳转逻辑
    def outline_confirm_router(state: NovelCreationState) -> str:
        return "generate_complete_novel" if state.get("is_outline_confirmed", False) else "generate_outline_chapter"
    graph.add_conditional_edges("confirm_outline_chapter", outline_confirm_router)
    
    # 小说生成完成后结束
    graph.add_edge("generate_complete_novel", END)
    
    # 1. 创建官方推荐的 MemorySaver 检查点
    checkpointer = MemorySaver()
    # 2. 编译工作流：完全匹配 v1.0.0+ 接口规范
    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["confirm_basic_setting", "confirm_outline_chapter"]  # 审核节点前中断
    )
    
    return compiled_graph
```

### 7.4.5案例参考

```python
import os
from typing import Dict, List, Optional, TypedDict, NotRequired
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

# ===================== 1. 加载环境变量 =====================
# 加载.env文件中的环境变量（如API_KEY），避免硬编码敏感信息
load_dotenv()

# ===================== 2. 初始化大语言模型 =====================
# 配置DeepSeek大模型参数，用于小说创作各阶段的文本生成
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),  # 从环境变量读取API密钥
    base_url="https://api.deepseek.com",  # DeepSeek API地址
    model="deepseek-chat",  # 选用的模型版本
    temperature=0.3  # 生成文本的随机性，0.3表示低随机性，输出更稳定
)

# ===================== 3. 定义工作流状态结构 =====================
# 使用TypedDict定义工作流的状态数据结构，统一管理全流程的所有数据
# 包含输入、生成结果、确认状态、进度追踪四大类字段
class NovelCreationState(TypedDict):
    """小说创作全流程状态管理（含进度追踪）"""
    # 初始输入：用户的小说创作需求（必填）
    user_requirement: str
    # 基础设定：生成的小说核心信息（非必填，生成后赋值）
    novel_title: NotRequired[Optional[str]]  # 小说标题
    main_characters: NotRequired[Optional[List[Dict[str, str]]]]  # 主要角色列表
    plot_overview: NotRequired[Optional[str]]  # 情节概述
    # 确认状态：标记人工审核结果
    is_setting_confirmed: NotRequired[bool]  # 基础设定是否确认
    is_outline_confirmed: NotRequired[bool]  # 大纲章节是否确认
    # 大纲与章节：生成的结构信息
    novel_outline: NotRequired[Optional[str]]  # 整体大纲
    chapter_structure: NotRequired[Optional[List[Dict[str, str]]]]  # 章节结构列表
    # 最终小说：生成的完整正文
    complete_novel: NotRequired[Optional[str]]
    # 进度追踪：监控流程执行状态
    current_stage: NotRequired[str]  # 当前流程阶段（需求收集/设定生成/大纲生成/小说生成）
    chapter_generated_count: NotRequired[int]  # 已生成章节数

# ===================== 4. 工具函数：进度展示 =====================
def print_process_progress(current_stage: str, detail: str = ""):
    """打印整体流程进度，让用户直观了解当前执行阶段"""
    # 阶段映射表：将阶段名称转换为进度百分比标识
    stage_map = {
        "需求收集": "1/4",
        "设定生成": "2/4",
        "大纲生成": "3/4",
        "小说生成": "4/4"
    }
    progress = stage_map.get(current_stage, "未知阶段")
    print(f"\n🔄 【整体进度 {progress}】- {current_stage} {detail}")

def print_chapter_progress(generated: int, total: int):
    """打印章节生成进度（百分比），监控小说正文生成进度"""
    percentage = (generated / total) * 100 if total > 0 else 0
    print(f"\n📖 【章节进度】已完成 {generated}/{total} 章 ({percentage:.1f}%)")

# ===================== 5. 定义各阶段节点函数 =====================
def get_user_input(state: NovelCreationState) -> NovelCreationState:
    """节点1：接收用户输入的创作需求（流程入口）"""
    print_process_progress("需求收集", "（开始）")
    # 获取用户输入的创作需求（题材/风格/其他要求）
    user_input = input("请输入你的小说创作需求（题材/风格/其他要求）：")
    # 初始化状态核心字段
    state["user_requirement"] = user_input
    state["current_stage"] = "需求收集"
    state["is_setting_confirmed"] = False  # 初始化为未确认
    state["is_outline_confirmed"] = False  # 初始化为未确认
    print_process_progress("需求收集", "（完成）✅")
    return state

def generate_basic_setting(state: NovelCreationState) -> NovelCreationState:
    """节点2：根据用户需求生成小说基础设定（标题/角色/情节）"""
    print_process_progress("设定生成", "（开始生成题目/角色/情节）")
    
    # 定义基础设定生成的提示词模板，约束输出格式和内容要求
    prompt = PromptTemplate(
        template="""
        请根据用户需求生成小说基础设定，要求：
        1. 小说题目：1-2个备选，简洁有吸引力
        2. 主要角色：至少3个，格式为「姓名：性格描述」
        3. 情节概述：100-200字，清晰说明故事整体走向
        
        用户需求：{user_requirement}
        
        输出格式（严格遵循）：
        题目：xxx
        主要角色：
        - 姓名1：性格描述1
        - 姓名2：性格描述2
        - 姓名3：性格描述3
        情节概述：xxx
        """,
        input_variables=["user_requirement"]
    )
    
    # 调用大模型生成基础设定内容
    response = llm.invoke(prompt.format(user_requirement=state["user_requirement"]))
    setting_content = response.content.strip()
    
    # 解析模型输出，提取标题、角色、情节信息并更新状态
    lines = setting_content.split("\n")
    state["main_characters"] = []
    for line in lines:
        if line.startswith("题目："):
            state["novel_title"] = line.replace("题目：", "").strip()
        elif line.startswith("主要角色："):
            continue
        elif line.startswith("- "):
            name, desc = line.replace("- ", "").split("：", 1)
            state["main_characters"].append({"姓名": name, "性格描述": desc})
        elif line.startswith("情节概述："):
            state["plot_overview"] = line.replace("情节概述：", "").strip()
    
    # 展示生成的基础设定，供用户审核
    print("\n===== 生成的小说基础设定 =====")
    print(f"题目：{state['novel_title']}")
    print("主要角色：")
    for char in state["main_characters"]:
        print(f"- {char['姓名']}：{char['性格描述']}")
    print(f"情节概述：{state['plot_overview']}")
    
    state["current_stage"] = "设定生成"
    print_process_progress("设定生成", "（完成）✅")
    return state

def confirm_basic_setting(state: NovelCreationState) -> NovelCreationState:
    """节点3：人工审核确认基础设定（支持修改后重新生成）"""
    print("\n===== ⚠️ 人工审核 - 基础设定确认环节 =====")
    confirm = input("是否确认以上基础设定？（确认请输入y，需修改请输入n并说明修改内容）：")
    
    if confirm.lower() == "y":
        # 用户确认设定，标记状态为已确认
        state["is_setting_confirmed"] = True
        print("✅ 基础设定已确认，进入下一阶段！")
    else:
        # 用户需要修改，接收修改需求并重新生成设定
        modify_content = input("请输入你的修改需求（如：修改角色名/调整情节/更换题目）：")
        print("🔄 正在根据你的需求修改基础设定...")
        
        # 定义修改后的提示词模板，基于原始需求+修改需求重新生成
        prompt = PromptTemplate(
            template="""
            请根据用户的原始需求和修改需求，更新小说基础设定：
            原始需求：{user_requirement}
            修改需求：{modify_content}
            输出格式（严格遵循）：
            题目：xxx
            主要角色：
            - 姓名1：性格描述1
            - 姓名2：性格描述2
            - 姓名3：性格描述3
            情节概述：xxx
            """,
            input_variables=["user_requirement", "modify_content"]
        )
        
        # 调用模型重新生成修改后的设定
        response = llm.invoke(prompt.format(
            user_requirement=state["user_requirement"],
            modify_content=modify_content
        ))
        setting_content = response.content.strip()
        
        # 重新解析修改后的设定内容
        lines = setting_content.split("\n")
        state["main_characters"] = []
        for line in lines:
            if line.startswith("题目："):
                state["novel_title"] = line.replace("题目：", "").strip()
            elif line.startswith("主要角色："):
                continue
            elif line.startswith("- "):
                name, desc = line.replace("- ", "").split("：", 1)
                state["main_characters"].append({"姓名": name, "性格描述": desc})
            elif line.startswith("情节概述："):
                state["plot_overview"] = line.replace("情节概述：", "").strip()
        
        # 展示修改后的设定，再次确认
        print("\n===== 修改后的基础设定 =====")
        print(f"题目：{state['novel_title']}")
        print("主要角色：")
        for char in state["main_characters"]:
            print(f"- {char['姓名']}：{char['性格描述']}")
        print(f"情节概述：{state['plot_overview']}")
        
        re_confirm = input("是否确认修改后的设定？（y/n）：")
        if re_confirm.lower() == "y":
            state["is_setting_confirmed"] = True
            print("✅ 基础设定已确认！")
        else:
            print("❌ 未确认，将重新生成基础设定。")
    
    return state

def generate_outline_chapter(state: NovelCreationState) -> NovelCreationState:
    """节点4：基于已确认的基础设定生成小说大纲与章节结构"""
    # 校验前置条件：基础设定未确认则无法生成大纲
    if not state.get("is_setting_confirmed", False):
        raise ValueError("❌ 基础设定未确认，无法生成大纲！")
    
    print_process_progress("大纲生成", "（开始生成大纲/章节结构）")
    
    # 定义大纲生成提示词模板，约束大纲和章节的内容要求
    prompt = PromptTemplate(
        template="""
        请根据已确认的小说基础设定，生成：
        1. 小说整体大纲：200-300字，清晰说明故事的开端、发展、高潮、结局
        2. 章节结构：至少8章，格式为「章节X：章节情节概述（1-2句话）」，章节间逻辑连贯
        
        基础设定：
        题目：{novel_title}
        主要角色：{main_characters}
        情节概述：{plot_overview}
        
        输出格式（严格遵循）：
        整体大纲：xxx
        章节结构：
        - 章节1：xxx
        - 章节2：xxx
        ...
        """,
        input_variables=["novel_title", "main_characters", "plot_overview"]
    )
    
    # 格式化角色信息，适配提示词输入格式
    char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
    
    # 调用模型生成大纲和章节结构
    response = llm.invoke(prompt.format(
        novel_title=state["novel_title"],
        main_characters=char_str,
        plot_overview=state["plot_overview"]
    ))
    outline_content = response.content.strip()
    
    # 解析模型输出，提取大纲和章节信息
    lines = outline_content.split("\n")
    state["chapter_structure"] = []
    for line in lines:
        if line.startswith("整体大纲："):
            state["novel_outline"] = line.replace("整体大纲：", "").strip()
        elif line.startswith("章节结构："):
            continue
        elif line.startswith("- 章节"):
            chapter_name, chapter_desc = line.replace("- ", "").split("：", 1)
            state["chapter_structure"].append({"章节名": chapter_name, "情节概述": chapter_desc})
    
    # 展示生成的大纲和章节结构，供用户审核
    print("\n===== 生成的小说大纲与章节结构 =====")
    print(f"整体大纲：{state['novel_outline']}")
    print("章节结构：")
    for chapter in state["chapter_structure"]:
        print(f"- {chapter['章节名']}：{chapter['情节概述']}")
    
    state["current_stage"] = "大纲生成"
    print_process_progress("大纲生成", "（完成）✅")
    return state

def confirm_outline_chapter(state: NovelCreationState) -> NovelCreationState:
    """节点5：人工审核确认大纲与章节结构（支持修改后重新生成）"""
    print("\n===== ⚠️ 人工审核 - 大纲与章节结构确认环节 =====")
    confirm = input("是否确认以上大纲与章节结构？（确认请输入y，需修改请输入n并说明修改内容）：")
    
    if confirm.lower() == "y":
        # 用户确认大纲，标记状态为已确认
        state["is_outline_confirmed"] = True
        print("✅ 大纲与章节结构已确认，进入小说生成阶段！")
    else:
        # 用户需要修改，接收修改需求并重新生成大纲
        modify_content = input("请输入你的修改需求（如：调整章节顺序/修改某章情节/增减章节数）：")
        print("🔄 正在根据你的需求修改大纲与章节结构...")
        
        # 格式化角色信息
        char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
        # 定义修改后的大纲生成提示词模板
        prompt = PromptTemplate(
            template="""
            请根据已确认的基础设定和用户修改需求，更新小说大纲与章节结构：
            基础设定：
            题目：{novel_title}
            主要角色：{main_characters}
            情节概述：{plot_overview}
            修改需求：{modify_content}
            
            输出格式（严格遵循）：
            整体大纲：xxx
            章节结构：
            - 章节1：xxx
            - 章节2：xxx
            ...
            """,
            input_variables=["novel_title", "main_characters", "plot_overview", "modify_content"]
        )
        
        # 调用模型重新生成修改后的大纲
        response = llm.invoke(prompt.format(
            novel_title=state["novel_title"],
            main_characters=char_str,
            plot_overview=state["plot_overview"],
            modify_content=modify_content
        ))
        outline_content = response.content.strip()
        
        # 重新解析修改后的大纲和章节结构
        lines = outline_content.split("\n")
        state["novel_outline"] = None
        state["chapter_structure"] = []
        for line in lines:
            if line.startswith("整体大纲："):
                state["novel_outline"] = line.replace("整体大纲：", "").strip()
            elif line.startswith("章节结构："):
                continue
            elif line.startswith("- 章节"):
                chapter_name, chapter_desc = line.replace("- ", "").split("：", 1)
                state["chapter_structure"].append({"章节名": chapter_name, "情节概述": chapter_desc})
        
        # 展示修改后的大纲，再次确认
        print("\n===== 修改后的大纲与章节结构 =====")
        print(f"整体大纲：{state['novel_outline']}")
        print("章节结构：")
        for chapter in state["chapter_structure"]:
            print(f"- {chapter['章节名']}：{chapter['情节概述']}")
        
        re_confirm = input("是否确认修改后的大纲与章节结构？（y/n）：")
        if re_confirm.lower() == "y":
            state["is_outline_confirmed"] = True
            print("✅ 大纲与章节结构已确认！")
        else:
            print("❌ 未确认，将重新生成大纲。")
    
    return state

def generate_complete_novel(state: NovelCreationState) -> NovelCreationState:
    """节点6：基于已确认的大纲逐章生成小说正文（带章节进度监控）"""
    # 校验前置条件：大纲未确认则无法生成小说正文
    if not state.get("is_outline_confirmed", False):
        raise ValueError("❌ 大纲与章节未确认，无法生成小说！")
    
    print_process_progress("小说生成", "（开始逐章生成正文）")
    # 初始化章节生成进度
    state["chapter_generated_count"] = 0
    chapter_total = len(state["chapter_structure"])
    print_chapter_progress(0, chapter_total)
    
    # 格式化小说基础信息，供单章生成时使用
    char_str = "\n".join([f"{c['姓名']}：{c['性格描述']}" for c in state["main_characters"]])
    novel_basic_info = f"""
    小说题目：{state['novel_title']}
    主要角色：{char_str}
    整体大纲：{state['novel_outline']}
    """
    # 初始化小说完整内容，包含标题和核心设定
    full_novel_content = f"# {state['novel_title']}\n\n## 小说核心设定\n{novel_basic_info.replace('    ', '')}\n\n---\n"
    
    # 定义单章正文生成的提示词模板，约束单章内容的格式和质量
    chapter_prompt = PromptTemplate(
        template="""
        请根据小说的核心设定、整体大纲，生成指定章节的正文内容，要求：
        1. 内容严格遵循该章节的情节概述，细节丰富，符合小说创作风格
        2. 角色性格与基础设定一致，对话自然，动作、心理描写贴合角色
        3. 章节开头标注章节名，结尾做轻微过渡，为下一章铺垫
        4. 单章字数控制在200-400字，语言流畅，情节连贯
        
        小说核心设定：{novel_basic_info}
        当前生成章节：{chapter_name}
        本章节情节概述：{chapter_desc}
        已生成章节数：{generated_chapter_num}/{total_chapter}
        
        输出格式：直接输出生成的章节正文，无需额外说明
        """,
        input_variables=["novel_basic_info", "chapter_name", "chapter_desc", "generated_chapter_num", "total_chapter"]
    )
    
    # 逐章生成小说正文
    for idx, chapter in enumerate(state["chapter_structure"], 1):
        chapter_name = chapter["章节名"]
        chapter_desc = chapter["情节概述"]
        print(f"\n🔨 【生成中】{chapter_name}...")
        
        # 调用模型生成单章正文
        chapter_response = llm.invoke(chapter_prompt.format(
            novel_basic_info=novel_basic_info,
            chapter_name=chapter_name,
            chapter_desc=chapter_desc,
            generated_chapter_num=idx,
            total_chapter=chapter_total
        ))
        chapter_content = chapter_response.content.strip()
        
        # 拼接单章内容到完整小说中
        full_novel_content += f"\n{chapter_content}\n\n---\n"
        # 更新章节生成进度
        state["chapter_generated_count"] = idx
        print_chapter_progress(idx, chapter_total)
        print(f"✅ 【生成完成】{chapter_name}：\n{chapter_content}\n" + "-"*50)
    
    # 补充小说完本信息，完成最终内容拼接
    full_novel_content += f"\n### 小说完本（总章节数：{chapter_total} | 创作基于用户需求：{state['user_requirement']}）"
    state["complete_novel"] = full_novel_content
    state["current_stage"] = "小说生成"
    
    # 展示最终进度
    print_process_progress("小说生成", "（完成）✅")
    print(f"\n🎉 逐章生成完成！小说共{chapter_total}章，总字数≥2000字")
    return state

# ===================== 6. 构建LangGraph工作流 =====================
def build_novel_creation_graph() -> CompiledStateGraph:
    """构建带人工审核中断的小说创作工作流"""
    # 1. 初始化状态图，绑定自定义的状态数据结构
    graph = StateGraph(NovelCreationState)
    
    # 2. 向状态图中添加所有业务节点
    graph.add_node("get_user_input", get_user_input)               # 需求收集节点
    graph.add_node("generate_basic_setting", generate_basic_setting) # 基础设定生成节点
    graph.add_node("confirm_basic_setting", confirm_basic_setting)   # 基础设定确认节点
    graph.add_node("generate_outline_chapter", generate_outline_chapter) # 大纲生成节点
    graph.add_node("confirm_outline_chapter", confirm_outline_chapter)   # 大纲确认节点
    graph.add_node("generate_complete_novel", generate_complete_novel)   # 小说生成节点
    
    # 3. 定义节点执行顺序（核心工作流逻辑）
    graph.set_entry_point("get_user_input")  # 设置流程入口节点
    graph.add_edge("get_user_input", "generate_basic_setting")  # 需求收集→设定生成
    graph.add_edge("generate_basic_setting", "confirm_basic_setting")  # 设定生成→设定确认
    
    # 4. 定义设定确认后的分支逻辑：确认则生成大纲，未确认则重新生成设定
    def setting_confirm_router(state: NovelCreationState) -> str:
        return "generate_outline_chapter" if state.get("is_setting_confirmed", False) else "generate_basic_setting"
    graph.add_conditional_edges("confirm_basic_setting", setting_confirm_router)
    
    # 5. 大纲生成后跳转至大纲确认节点
    graph.add_edge("generate_outline_chapter", "confirm_outline_chapter")
    
    # 6. 定义大纲确认后的分支逻辑：确认则生成小说，未确认则重新生成大纲
    def outline_confirm_router(state: NovelCreationState) -> str:
        return "generate_complete_novel" if state.get("is_outline_confirmed", False) else "generate_outline_chapter"
    graph.add_conditional_edges("confirm_outline_chapter", outline_confirm_router)
    
    # 7. 小说生成完成后结束流程
    graph.add_edge("generate_complete_novel", END)
    
    # 8. 配置检查点存储：使用内存存储工作流状态，支持中断后恢复
    checkpointer = MemorySaver()
    # 9. 编译工作流：配置中断点（在人工审核节点前暂停，等待用户输入）
    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["confirm_basic_setting", "confirm_outline_chapter"]  # 审核节点前中断
    )
    
    return compiled_graph

# ===================== 7. 运行小说创作流程 =====================
if __name__ == "__main__":
    # 1. 构建工作流实例
    novel_graph = build_novel_creation_graph()
    
    # 2. 配置线程ID（用于区分不同的创作流程，每个流程独立存储状态）
    thread_id = "novel_creation_enterprise_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 3. 初始化工作流状态
    initial_state: NovelCreationState = {
        "user_requirement": "",
        "current_stage": "初始",
        "chapter_generated_count": 0
    }

    print("🚀 小说创作助手启动")
    print("==============================================")

    # 核心逻辑：处理工作流中断与恢复
    # 第一次启动：执行从入口节点到第一个中断点的流程
    novel_graph.invoke(initial_state, config=config)

    while True:
        # 获取当前线程的状态快照，判断流程是否中断
        state_snapshot = novel_graph.get_state(config)
        
        # 如果没有下一个待执行节点，说明流程已完成，退出循环
        if not state_snapshot.next:
            print("\n🎉 所有流程已完成！")
            break
        
        # 流程中断在某个审核节点前，提示用户并恢复执行
        target_node = state_snapshot.next[0]
        print(f"\n--- ⏸️ 流程在节点 [{target_node}] 处等待人工干预 ---")
        
        # 恢复执行：传入None表示从上一个检查点继续，触发人工审核节点的输入交互
        novel_graph.invoke(None, config=config)

    # 4. 获取最终生成结果并保存到文件
    final_state = novel_graph.get_state(config).values
    if "complete_novel" in final_state and final_state["complete_novel"]:
        filename = "novel_final_output.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_state["complete_novel"])
        print(f"\n📁 完整小说已保存到: {filename}")
    else:
        print("\n⚠️ 流程未能生成完整内容。")
```

【学习提示】

测试注意事项：

- 运行代码后，严格按照提示输入（用户需求、审核结果），不要输入多余内容，避免程序异常；
- 测试完成后，可查看当前目录下的novel_final_output.txt文件，查看保存的小说内容。

### 7.4.6 实践总结与拓展思考

恭喜大家！完成了本次综合实践，成功用langgraph实现了智能体小说创作助手。现在，我们回头梳理一下，本次实践用到了哪些前2章的知识点，加深大家的理解：

#### 1. 知识点回顾

- 节点（Node）：我们定义的6个节点，本质上就是langgraph的“执行单元”，与第X章“节点的定义与使用”完全对应；
- 状态（State）：NovelCreationState类，对应第X章“状态管理”，实现了节点间的数据结构化传递，避免了数据混乱；
- 图的编译与运行：StateGraph的创建、编译、run方法，对应第X章“图的构建与运行”，是将所有组件串联起来的关键。

#### 2. 实践反思

大家在测试过程中，可能会遇到2个常见问题，这里总结一下解决方案，帮助大家后续规避：

- 问题1：LLM生成格式错误，导致JSON解析失败——解决方案：优化提示词，明确要求LLM输出严格的JSON格式，同时添加异常处理，给默认值；
- 问题2：节点流转异常，比如审核不通过后，没有返回LLM初始生成节点——解决方案：检查条件分支边的判断函数和分支映射，确保返回的分支标识与branch_map中的key一致（比如audit_result必须是"pass"/"reject"，不能是其他值）。

### 7.4.7 实践任务

为了确保大家真正掌握本次实践的知识点，布置以下必做任务，大家课后完成，下节课我们将抽查并讲解：

1. 完整运行本次实践的所有代码，完成一次小说生成，保存XXXX.txt文件；
2. 修改“LLM初始生成节点”的提示词，让LLM生成3个以上角色，重新运行流程，观察修改后的效果；
3. 思考并记录：本次实践中，langgraph的状态管理和条件分支，分别解决了什么问题？（不少于100字）。

最后，提醒大家：本次实践的核心是“掌握langgraph框架”，小说生成的质量不是重点。大家在动手过程中，一定要多思考“每个组件对应什么知识点”“为什么要这么设计节点和状态”，只有这样，才能真正理解langgraph的核心思想。

### 7.4.8 深度思考

为什么在综合实践的案例中，等待用户审核要用Interrupts机制？使用这种机制的好处是什么？如果是企业级应用，多个用户调用智能体的时候怎么保证不同任务中断后能继续顺利启动？

上面的问题没有标准答案，每个人有不同见解~

好啦，以上就是本节的全部内容，祝大家学习愉快

