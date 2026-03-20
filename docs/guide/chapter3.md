# 第三章 LangChain进阶组件实操

## 前言

上一章我们了解了LangChain的“核心”——组件，包括模型调用、提示词模板以及输出解析。

从这一章开始，我们将聚焦LangChain生态中的核心进阶组件，从状态管理、外部行动两个核心维度拆解组件原理，再通过组件组合实践掌握复杂应用的构建方法。

通过本章学习，你将具备构建带记忆、能交互外部系统的智能应用的能力。

Go Go Go，我们就出发吧！

## 3.1 状态管理层（Memory）：让模型拥有记忆能力

大语言模型（LLM）的原生调用是无状态的，即每次对话都是独立的请求，无法主动记住上下文信息。LangChain的Memory组件正是为解决这一问题而生，它通过结构化的方式存储、管理对话历史，让AI具备“记忆能力”。

### 3.1.1 对话记忆的本质与作用

#### 3.1.1.1 核心本质

其实Memory组件的工作逻辑特别好理解，就像我们聊天时记笔记一样：

每次你和AI对话后，它会把“你说的话”和“它的回复”整理好存起来（这一步叫“存储”）；等你下一次提问时，它会先把之前存的笔记拿出来，和你新的问题拼在一起再交给LLM（这一步叫“提取”）。这样LLM就能看到完整的对话上下文，自然就能记住你之前说的内容了。

从技术实现上，Memory组件通过两个核心动作完成工作：

- 存储（Save）：将每一轮的用户输入（HumanMessage）和AI输出（AIMessage）保存到指定存储介质（内存、数据库等）；

- 提取（Load）：新一轮对话时，从存储介质中提取历史对话，注入到Prompt中供LLM参考。

#### 3.1.1.2 核心作用

记忆功能看似简单，但能帮我们解决很多实际问题：

- 避免重复提问：比如你说过“我叫小明”，之后不用再重复，AI也能叫出你的名字；
- 支撑复杂任务：比如你让AI“先梳理我的需求，再生成方案”，它能记住中间的需求梳理结果，不会中途失忆；
- 简化交互：不用每次提问都把前因后果说一遍，比如问“这个组件怎么用？”，AI知道你说的是之前聊的Memory组件。

### 3.1.2 三种基础Memory组件实操

LangChain提供了多种Memory实现，适用于不同场景。我们重点学最常用的三种——全量记忆、窗口记忆、摘要记忆。它们各有适用场景，学会了就能应对大部分需求。

需要注意的是，LangChain推荐使用LCEL（LangChain Expression Language）架构，通过 `RunnableWithMessageHistory` 结合 `BaseChatMessageHistory` 抽象类实现对话记忆管理，替代后续不支持的 `ConversationChain`。

LCEL（LangChain Expression Language）是 LangChain 0.2+ 的核心表达式语言，用管道符 `|` 把组件串成流水线，前一个组件的输出自动作为后一个的输入，就像工厂的生产线。

本教程将基于该架构分别实现三种核心记忆模式：

- **全量记忆**：完整保存所有对话历史，适用于短对话场景
- **窗口记忆**：仅保留最近N轮对话，控制Token消耗
- **摘要记忆**：通过LLM生成对话摘要替代完整历史，平衡上下文连贯性与效率

核心优势：支持多会话隔离（通过session_id）、自动管理历史消息的注入与保存、适配现代LLM模型的会话交互逻辑。

session_id 是用户的唯一标识，不同 session_id 的对话记忆相互隔离，就像微信不同聊天窗口的记录互不干扰。

【前置准备】所有案例需先完成环境配置：

```bash
# 安装完整依赖
pip install langchain langchain-openai python-dotenv langchain-experimental
```

**.env 文件**示例

```
API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BASE_URL="xxxxxx"
```

实践代码

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 加载环境变量（确保.env文件中配置了API_KEY）
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化LLM模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3  # 降低随机性，保证输出稳定
)
```

#### 3.1.2.1 全量记忆

使用`InMemoryChatMessageHistory` 存储完整对话历史，每次调用时自动注入所有历史消息到提示词中，适用于对话轮数少、需要完整上下文的场景。

```python
# 1. 定义提示词模板（包含历史消息占位符）
full_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于完整的历史对话回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 历史消息占位符
    ("human", "{user_input}")  # 用户当前输入
])

# 2. 构建基础链（提示词 + LLM）
base_chain = full_memory_prompt | llm

# 3. 会话历史存储（内存模式，生产环境可替换为数据库存储）
full_memory_store = {}

# 4. 定义会话历史获取函数（核心：返回完整历史）
def get_full_memory_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取会话历史，不存在则创建新的历史记录"""
    if session_id not in full_memory_store:
        full_memory_store[session_id] = InMemoryChatMessageHistory()
    return full_memory_store[session_id]

# 5. 构建带全量记忆的对话链
full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_full_memory_history,
    input_messages_key="user_input",  # 输入中用户问题的键名
    history_messages_key="chat_history"  # 传入提示词的历史消息键名
)
```

**ChatPromptTemplate.from_messages** 创建了一个对话提示模板，它就像是给AI设定了一个“剧本”，规定了对话的结构和角色。

**MessagesPlaceholder(variable_name="chat_history")**
这是一个历史消息占位符。这是实现对话记忆（记忆功能）的关键。它不在模板中写死任何内容，而是在程序运行时，动态地将之前的对话记录（比如用户之前问了什么，AI回答了什么）插入到这个位置。这样，AI在回答新问题时就能参考上下文，实现连贯的多轮对话。

**base_chain = full_memory_prompt | llm**
这行代码使用管道操作符 `|`将两个组件连接起来，形成了一个简单的处理链，其含义是**将前一个组件的输出，作为后一个组件的输入**。

- **执行 `prompt`**：`prompt`组件接收一个包含 `input`（用户新问题）和 `history`（历史对话）的变量字典，然后根据模板生成一个结构化的消息列表。

- **管道传递 (`|`)**：这个生成好的消息列表被自动传递给 `llm`（大型语言模型，如 GPT-4）。

- **执行 `llm`**：`llm`组件根据收到的消息列表，生成一段连贯且符合上下文的回答。

测试验证

```python
# 测试多轮对话（指定session_id=user_001，隔离不同用户）
config = {"configurable": {"session_id": "user_001"}}

# 第一轮对话
response1 = full_memory_chain.invoke({"user_input": "我叫小明，喜欢编程"}, config=config)
print("助手回复1：", response1.content)
# 输出示例：你好小明！编程是一项很有创造力的技能，你平时常用什么编程语言呢？

# 第二轮对话（验证记忆：询问历史信息）
response2 = full_memory_chain.invoke({"user_input": "我刚才说我喜欢什么？"}, config=config)
print("助手回复2：", response2.content)
# 输出示例：你刚才说你喜欢编程呀～

# 查看完整历史记录
print("\n全量记忆的对话历史：")
for msg in get_full_memory_history("user_001").messages:
    print(f"{msg.type}: {msg.content}")
```

运行结果

```
助手回复1： 你好小明！很高兴认识你！编程是个非常棒的爱好，能创造、解决问题，还能实现各种有趣的想法。你主要对哪种编程语言或领域感兴趣呢？比如网页开发、数据分析、游戏设计，还是其他方向？ 😊
助手回复2： 你刚才提到你喜欢编程！需要我推荐一些学习资源、项目灵感，或者聊聊编程相关的话题吗？ 😄
助手回复3： 你刚才告诉我，你的名字是**小明**！需要我帮你记录或规划与编程相关的学习目标吗？ 😊

全量记忆的对话历史：
human: 我叫小明，喜欢编程
ai: 你好小明！很高兴认识你！编程是个非常棒的爱好，能创造、解决问题，还能实现各种有趣的想法。你主要对哪种编程语言或领域感兴趣呢？比如网页开发、数据分析、游戏设计，还是其他方向？  😊
human: 我刚才说我喜欢什么？
ai: 你刚才提到你喜欢编程！需要我推荐一些学习资源、项目灵感，或者聊聊编程相关的话题吗？ 😄
human: 我叫什么名字
ai: 你刚才告诉我，你的名字是**小明**！需要我帮你记录或规划与编程相关的学习目标吗？ 😊
```

通过运行结果可以发现：全量记忆会完整保存所有对话历史，AI 能准确回答所有基于历史的问题，适合短对话场景。

#### 3.1.2.2 窗口记忆

全量记忆适合短对话，那长对话怎么办？

这时候就需要“窗口记忆”——它只保留最近的N轮对话（N用k参数控制），早期的对话会自动丢弃。这样能有效控制文字量，适合客服、长期陪伴等长对话场景。

```python
# 1. 定义提示词模板（与全量记忆通用，可复用）
window_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于最近的对话历史回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 2. 构建基础链
window_base_chain = window_memory_prompt | llm

# 3. 会话历史存储
window_memory_store = {}
WINDOW_SIZE = 2  # 保留最近2轮对话（即最近4条消息：用户-助手-用户-助手）

# 4. 定义带窗口限制的会话历史获取函数
def get_window_memory_history(session_id: str) -> BaseChatMessageHistory:
    """获取会话历史，仅保留最近WINDOW_SIZE轮对话"""
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    
    # 获取完整历史，截取最近WINDOW_SIZE轮（每轮2条消息）
    history = window_memory_store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        # 截取后WINDOW_SIZE轮消息（保留最新的）
        history.messages = history.messages[-2 * WINDOW_SIZE:]
    return history

# 5. 构建带窗口记忆的对话链
window_memory_chain = RunnableWithMessageHistory(
    runnable=window_base_chain,
    get_session_history=get_window_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)
```

测试验证

```python
# 测试多轮对话（session_id=user_002，与全量记忆会话隔离）
config = {"configurable": {"session_id": "user_002"}}

# 模拟5轮对话，验证窗口记忆的截断效果
inputs = [
    "我叫小红",
    "我喜欢画画",
    "我来自上海",
    "我是一名学生",
    "我刚才说我来自哪里？",  # 第5轮：询问第3轮的信息，验证窗口截断
    "我叫什么名字？"  # 第6轮：询问第1轮的信息，验证窗口记忆
]

for i, user_input in enumerate(inputs, 1):
    response = window_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看窗口记忆的最终历史（仅保留最近2轮）
print("\n窗口记忆的最终对话历史（最近2轮）：")
for msg in get_window_memory_history("user_002").messages:
    print(f"{msg.type}: {msg.content}")
```

运行结果

```
第1轮 - 助手回复： 你好小红！很高兴认识你！有什么我可以帮你的吗？

第2轮 - 助手回复： 画画是很棒的爱好呢！你通常喜欢画什么类型的作品？比如风景、人物，还是抽象画？

第3轮 - 助手回复： 上海是个充满艺术气息的城市呢！那里有很多美术馆和创意园区，比如西岸艺术中心、M50创意园，说不定能给你的创作带来灵感哦～

第4轮 - 助手回复： 学生时期能有时间坚持爱好真不容易呢！你是通过学校社团、课外班自学，还是纯粹当作放松的方式呢？

第5轮 - 助手回复： 你刚才提到你来自上海～需要我帮你推荐些适合学生参观的艺术展览或创意市集吗？(๑•̀ㅂ•́)و✧

窗口记忆的最终对话历史（最近2轮）：
human: 我是一名学生
ai: 学生时期能有时间坚持爱好真不容易呢！你是通过学校社团、课外班自学，还是纯粹当作放松的方式呢？
human: 我刚才说我来自哪里？
ai: 你刚才提到你来自上海～需要我帮你推荐些适合学生参观的艺术展览或创意市集吗？(๑•̀ㅂ•́)و✧
```

> 注意：窗口记忆仅保留最近 2 轮对话，第 3 轮 “来自上海” 的信息被截断，因此 AI 无法回答该问题，符合预期逻辑。

#### 3.1.2.3 摘要记忆

如果需要超长时间的对话（比如几小时的咨询），即使是窗口记忆也可能不够用。这时候就需要“摘要记忆”——它不保存对话原文，而是用LLM把历史对话总结成一段简洁的摘要。既能保留核心信息，又能最大程度节省文字量，缺点是可能会丢失一些细节（比如具体的数字、名字）。

```python
# 1. 定义摘要生成提示词（用于压缩对话历史）
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是对话摘要助手，需简洁总结以下对话的核心信息（包含用户身份、偏好、关键问题等），不超过50字。"),
    ("human", "对话历史：{chat_history_text}\n请生成摘要：")
])

# 2. 构建摘要生成链（输入完整历史文本，输出摘要）
summary_chain = summary_prompt | llm

# 3. 定义对话记忆提示词（注入摘要而非完整历史）
summary_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于对话摘要回答用户问题，摘要包含核心上下文信息。"),
    ("system", "对话摘要：{chat_summary}"),  # 注入摘要
    ("human", "{user_input}")
])

# 4. 构建基础对话链（提示词 + LLM）
summary_base_chain = (
    RunnablePassthrough.assign(
        chat_summary=lambda x: summary_chain.invoke(
            {
                "chat_history_text": "\n".join(
                    [f"{msg.type}: {msg.content}" for msg in x["chat_history"]]
                )
            }
        ).content
    )
    | summary_memory_prompt
    | llm
)

# 5. 会话历史存储（保存完整历史用于生成摘要）
summary_memory_store = {}

# 6. 定义会话历史获取函数
def get_summary_memory_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in summary_memory_store:
        summary_memory_store[session_id] = InMemoryChatMessageHistory()
    return summary_memory_store[session_id]

# 7. 构建带摘要记忆的对话链
summary_memory_chain = RunnableWithMessageHistory(
    runnable=summary_base_chain,
    get_session_history=get_summary_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"  # 传入完整历史用于生成摘要
)
```

测试验证

```python
# 测试多轮对话（session_id=user_003）
config = {"configurable": {"session_id": "user_003"}}

# 多轮对话输入
inputs = [
    "我叫小李，是一名产品经理",
    "我负责一款电商APP的迭代",
    "最近在优化用户下单流程",
    "遇到了用户流失率高的问题",
    "你能给我一些优化建议吗？"
]

for i, user_input in enumerate(inputs, 1):
    response = summary_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看完整历史与最终摘要
history = get_summary_memory_history("user_003")
print("\n摘要记忆的完整对话历史：")
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")

# 单独生成最终摘要验证
final_summary = summary_chain.invoke({
    "chat_history_text": "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
}).content
print(f"\n最终对话摘要：{final_summary}")
# 输出示例：摘要：小李，产品经理，负责电商APP迭代，优化下单流程时遇用户流失率高问题，寻求建议。
```

运行结果

```
第1轮 - 助手回复： 你好小李！很高兴认识你！作为产品经理，你的工作一定充满挑战和创意吧？如果需要讨论产品设计、用户需求或任何相关话题，我随时可以帮忙哦！ 😊

第2轮 - 助手回复： 很高兴能为您提供帮助！作为产品经理，您对电商APP的迭代有什么具体方向或问题需要探讨吗？比如用户增长、功能优化、体验提升，或是数据驱动决策等方面？

第3轮 - 助手回复： 好的，小李。优化下单流程是提升转化率和用户体验的关键。基于我们之前的讨论，这里有几个核心方向和具体建议供你参考：
...省略

第4轮 - 助手回复： 根据对话摘要，您正在优化电商APP的下单流程。针对用户流失率高的问题，可以结合AI之前的建议，从以下几个方向入手：

...省略

摘要记忆的完整对话历史：
human: 我叫小李，是一名产品经理
ai: 你好小李！很高兴认识你！作为产品经理，你的工作一定充满挑战和创意吧？如果需要讨论产品设计、用户需求或任何相关话题，我随时可以帮忙哦！ 😊
human: 我负责一款电商APP的迭代
ai: 很高兴能为您提供帮助！作为产品经理，您对电商APP的迭代有什么具体方向或问题需要探讨吗？比如用户增长、功能优化、体验提升，或是数据驱动决策等方面？
human: 最近在优化用户下单流程
ai: 好的，小李。优化下单流程是提升转化率和用户体验的关键。基于我们之前的讨论，这里有几个核心方向和具体建议供你参考：
...省略

需要进一步讨论具体功能或数据指标吗？

最终对话摘要：小李是电商APP产品经理，正优化下单流程以解决用户流失率高的问题，需要具体优化建议。
```

仔细观察发现，记忆里不是逐字逐句的对话原文，而是一段总结。这样即使对话很多，摘要也不会太长，非常适合超长对话场景。

> 优化提示：生成摘要会额外消耗 LLM 算力，生产环境可缓存摘要（比如每 5 轮更新一次），减少调用次数。

#### 3.1.2.4 三种Memory怎么选？

学完三种模式后，大家可能会纠结“该用哪个？”，这里整理了一张对比表，一看就懂：

| 记忆模式 | 核心优势                                | 局限性                                    | 适用场景                                                 |
| :------- | :-------------------------------------- | :---------------------------------------- | :------------------------------------------------------- |
| 全量记忆 | 上下文完整，无信息丢失，实现简单        | Token消耗随轮数线性增长，不适用于长对话   | 短对话、需要完整上下文的场景（如一对一咨询）             |
| 窗口记忆 | Token消耗可控，性能稳定，实现难度低     | 可能丢失早期关键信息，上下文连贯性有限    | 中长对话、对早期信息要求不高的场景（如闲聊）             |
| 摘要记忆 | Token消耗低，支持超长对话，保留核心信息 | 额外消耗LLM算力生成摘要，可能丢失细节信息 | 超长对话、需要平衡上下文与效率的场景（如客服、长期协作） |

#### 3.1.2.5 工程建议

- **存储优化**：示例中使用内存存储（InMemoryChatMessageHistory），生产环境需替换为持久化存储（如Redis、PostgreSQL、MongoDB），基于 `BaseChatMessageHistory` 实现自定义存储类。
- **性能优化**：摘要记忆可缓存摘要结果，避免每次调用都重新生成；窗口记忆可预计算历史消息长度，精准控制Token上限。
- **多模型适配**：可替换LLM为开源模型（如Llama 3、Mistral），降低API调用成本。
- **错误处理**：添加会话历史清理、异常重试、Token溢出检测等逻辑，提升系统稳定性。

#### 3.1.2.6 深入理解：记忆是怎么注入到对话里的？

可能有同学会好奇：“记忆到底是怎么被加到对话里的？” 

核心链路其实很简单：`用户新问题 → 记忆组件提取历史对话 → 把“历史+新问题”拼起来 → 发给LLM → LLM生成回复 → 把“新问题+回复”存到记忆里 → 输出结果`

要实现这个链路，需要两个核心组件配合：

1. ChatMessageHistory：相当于“记忆笔记本”，负责具体的存和取操作；
2. RunnableWithMessageHistory：相当于“记忆调度员”，负责协调整个流程——在调用LLM前自动取历史，调用后自动存新对话。

知道了原理，可以自己动手实践，不借助LangChain的框架，自己实现全量记忆。

手动实现全量记忆demo版（供参考）

```python
# 手动实现全量记忆（无LangChain框架，理解核心逻辑）
chat_history = []  # 存储对话历史的列表

def chat_with_memory(user_input):
    # 1. 拼接历史+新问题
    prompt = "你是友好的助手，结合历史对话回答：\n"
    for msg in chat_history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"用户：{user_input}"
    
    # 2. 调用LLM
    response = llm.invoke(prompt).content
    
    # 3. 保存新对话到历史
    chat_history.append({"role": "用户", "content": user_input})
    chat_history.append({"role": "AI", "content": response})
    
    return response

# 测试
print(chat_with_memory("我叫小明"))
print(chat_with_memory("我刚才叫什么名字？"))
```

## 3.2 外部行动层（Tool）：让AI能“动手”解决问题

学完记忆组件，AI已经能记住我们说的话了，但还有一个局限：它的知识只停留在训练数据里，没法获取实时信息（比如今天的天气）、操作电脑文件，也没法调用其他API。而Tool组件，就是给AI装上“手和脚”，让它能调用外部工具，解决这些原生能力解决不了的问题。

> Tips: LangChain 的 Agent = LangGraph 的轻量外部封装，Tool 调用底层都是由 LangGraph 的图节点管理和调度的。等后续学习了 LangGraph 后你大概就能理解了~~

### 3.2.1 先搞懂：工具调用的核心逻辑

简单说，工具调用就是让AI学会“思考→行动→反馈”的循环。比如我们问“今天北京天气怎么样？”，AI会这样做：

1. 思考：“这个问题我没法直接回答，需要调用天气工具查询”；
2. 行动：生成工具调用指令（比如“调用WeatherQuery工具，参数是北京”）；
3. 反馈：工具返回查询结果（比如“北京今天20℃，晴”），AI结合这个结果生成最终回答。

#### 3.2.1.1 三个关键组件

要实现工具调用，需要三个核心组件配合，就像一个团队：

- Tool（工具）：具体的“干活工具”，比如查询天气的工具、读文件的工具。每个工具都有明确的名称和描述（AI就是通过描述知道该用哪个工具的）；
- Toolkit（工具包）：把相关的工具打包在一起，比如“文件操作工具包”里包含读文件、写文件、列目录三个工具；
- Agent（智能体）：团队的“指挥官”，负责协调LLM和工具——判断要不要调用工具、用哪个工具、怎么处理工具返回的结果。

#### 3.2.1.2 学习案例：查天气

> 一般来说，应该是使用请求api获得实时的天气情况，但考虑到这类api一般是要收费的，为了学习方便，这里案例调整为固定函数的形式。

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# ======================
# 1. 环境
# ======================
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
)

# ======================
# 2. 工具
# ======================
@tool
def weather_query(city: str) -> str:
    """查询指定城市天气"""
    weather_data = {
        "北京": "北京今日天气：晴，-2~8℃",
        "上海": "上海今日天气：多云，5~12℃",
        "广州": "广州今日天气：小雨，18~25℃",
    }
    return weather_data.get(city, f"暂无 {city} 数据")

tools = [weather_query]

# ======================
# 3. 创建 Agent（开启 debug）
# ======================
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 👈 打开过程打印
)

# ======================
# 4. 运行
# ======================
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "北京今天的天气怎么样？"}
    ]
})

print("\n最终回答：")
print(response["messages"][-1].content)

```

运行结果

```

最终回答：
根据查询结果，北京今天的天气情况如下：

- **天气状况**：晴
- **温度范围**：-2°C 到 8°C

今天北京天气晴朗，但气温较低，早晚温差较大。建议您：
1. 白天可以适当外出活动，享受阳光
2. 早晚时段要注意保暖，特别是早上温度较低
3. 建议穿着厚外套或羽绒服，注意防寒

如果您需要了解未来几天的天气预报，请告诉我！
```

通过运行结果可以看到 AI 的完整思考过程：判断需要调用 WeatherQuery 工具 → 传入参数 “北京” → 工具返回结果 → AI 整理成自然语言回答。

**这就是工具调用的完整流程啦！**

到这里，我们其实就明白，大模型调用工具，本质是思考→行动→反馈，在回答过程中调用工具获得结果后再返回。

### 3.2.2 高级技巧：自定义工具（@tool装饰器）

通过前面的案例，我们发现LangChain提供了@tool装饰器，这样可以让我们能快速创建自己的工具，非常方便。

#### 3.2.2.1 自定义工具的核心要求

要让AI能正确使用你的工具，必须满足三个要求：

- 写清楚文档字符串（docstring）：告诉AI这个工具能做什么、需要什么参数；
- 明确参数类型：用类型注解（比如int、str）指定参数类型，AI才知道该传什么类型的值；
- 返回结果清晰：工具执行后的返回值要简洁明了，方便AI整理成回答。

#### 3.2.2.2 学习案例：温度单位转换

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# ======================
# 1. 环境变量
# ======================
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
)

# ======================
# 2. 参数模型
# ======================
class TemperatureConvertInput(BaseModel):
    temperature: float = Field(description="需要转换的温度值，例如37.0")
    from_unit: str = Field(description="原始温度单位，只能是celsius或fahrenheit")

# ======================
# 3. 工具
# ======================
@tool(args_schema=TemperatureConvertInput)
def temperature_converter(temperature: float, from_unit: str) -> str:
    """温度单位转换工具"""
    
    if from_unit not in ["celsius", "fahrenheit"]:
        return f"错误：单位'{from_unit}'不合法"

    if from_unit == "celsius":
        fahrenheit = temperature * 9/5 + 32
        return f"{temperature}摄氏度 = {fahrenheit:.2f}华氏度"
    else:
        celsius = (temperature - 32) * 5/9
        return f"{temperature}华氏度 = {celsius:.2f}摄氏度"


tools = [temperature_converter]

system_prompt = """
你是一名专业温度转换助手，只能使用temperature_converter工具完成计算。
"""

# ======================
# 4. 创建 Agent
# ======================
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=True
)

# ======================
# 5. 运行
# ======================
if __name__ == "__main__":

    query = "将37摄氏度转换为华氏度"

    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    print("\n最终结果：")
    print(response["messages"][-1].content)

```

运行结果

```
最终结果：
37摄氏度等于98.60华氏度。
```

可以看到模型正确调用@tool工具，然后得到了需要的结果。

我们来看看@tool工具的源码：

```python
(function)
def tool(
    *,
    description: str | None = None,          # 工具功能描述（Agent判断是否调用的核心依据）
    return_direct: bool = False,             # 是否直接返回工具结果（False：LLM整理后返回）
    args_schema: ArgsSchema | None = None,   # 参数校验模型（Pydantic BaseModel）
    infer_schema: bool = True,               # 是否自动从类型注解推导参数schema
    response_format: Literal['content', 'content_and_artifact'] = "content",  # 返回格式
    parse_docstring: bool = False,           # 是否解析docstring生成描述/参数信息
    error_on_invalid_docstring: bool = True  # docstring解析失败时是否抛异常
)
```

做一个简单的介绍：

1. `description: str | None = None`

- **核心作用**：手动指定工具的功能描述，是**Agent 判断 “何时调用该工具” 的核心依据**（Agent 会把用户问题和 description 匹配，决定是否调用）。
- **默认行为**：若不传，会优先通过`parse_docstring`（解析函数 docstring）或函数名推导简单描述。

2. `return_direct: bool = False`

- 核心作用：控制工具执行结果的返回逻辑：
  - `False`（默认）：工具结果会传给 LLM，由 LLM 整理成自然语言回答后返回；
  - `True`：跳过 LLM 二次处理，直接返回工具的原始执行结果。


3. `args_schema: ArgsSchema | None = None`

- **核心作用**：通过 Pydantic `BaseModel`（`ArgsSchema`是其别名）定义参数的**类型、描述、校验规则**（如取值范围、必填项、枚举值），强制约束 Agent 传参的正确性。

4. `infer_schema: bool = True`

- 核心作用：控制是否自动从函数的类型注解推导参数 schema：

  - `True`（默认）：自动解析函数注解，生成基础的参数类型校验；
  - `False`：禁用自动推导，仅使用`args_schema`（若传）或无参数校验。

5. `response_format: Literal['content', 'content_and_artifact'] = "content"`

  - 核心作用：控制工具返回结果的格式：

    - `"content"`（默认）：仅返回工具函数的返回值（如字符串、数值），简洁高效；
- `"content_and_artifact"`：返回字典`{"content": 工具结果, "artifact": 元信息}`，可携带执行耗时、请求 ID 等额外数据。

6. `parse_docstring: bool = False`

- 核心作用：控制是否解析函数的docstring，来自动推导`description`（工具描述）和参数信息：

  - `True`：优先从 docstring 提取工具描述、参数说明，替代手动传`description`/`args_schema`；
  - `False`（默认）：忽略 docstring，仅使用手动传的参数或自动注解推导。

7. `error_on_invalid_docstring: bool = True`

  - 核心作用：当`parse_docstring=True`但 docstring 格式不合法（如参数描述缺失、格式混乱）时，控制程序行为：
    - `True`（默认）：抛出异常，终止工具创建；
    - `False`：忽略解析错误，使用默认的工具描述 / 参数规则。

### 3.2.3 其他常用内置工具

除了自定义工具，LangChain 提供了很多**内置工具**，可直接用于 Agent，无需重复设计。

**文件操作工具（FileManagementToolkit）**

- `ReadFile`：读取本地文件（txt、docx 等）
- `WriteFile`：写入本地文件
- `ListDirectory`：查看文件夹内容
- `DeleteFile`（部分版本）：删除文件

还有其他的一些官方工具，大家可以根据自己的需要去使用。

> 运行注意事项：注意文件路径权限，Windows 路径用`\\`或`/`，避免权限不足导致操作失败。

#### 3.2.3.1 学习案例：创建文件

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from dotenv import load_dotenv
import os

# -------------------
# 1. 初始化环境
# -------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
)

# -------------------
# 2. 创建文件管理工具
# -------------------
toolkit = FileManagementToolkit(root_dir=".")
tools = toolkit.get_tools()

# -------------------
# 3. 创建 Agent（最新版）
# -------------------
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 打开调试，显示模型思考和工具调用过程
)

# -------------------
# 4. 执行任务
# -------------------
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "请创建一个名为 llm诗词.txt 的文件，并在文件中写入一首原创七言绝句，主题围绕科技与人文的融合。"}
    ]
})

print("\n任务执行完成！文件已写入。")
print("Agent最终输出：\n", response["messages"][-1].content)

```

运行结果

```
任务执行完成！文件已写入。
Agent最终输出：
 文件已成功创建！我在"llm诗词.txt"文件中写入了一首原创的七言绝句《智联古今》。

这首诗的主题围绕科技与人文的融合：
- 第一句"硅基算力织云锦"：以硅基芯片的算力比喻织造云锦，体现科技之美
- 第二句"代码如诗绘古今"：将编程代码比作诗歌，描绘古今文明
- 第三句"人文科技相辉映"：点明人文与科技相互辉映的主题
- 第四句"智能时代共知音"：表达在人工智能时代，科技与人文成为知音伙伴

这首诗符合七言绝句的格律要求，每句七个字，共四句，押韵工整，意境深远。
```

可以看到大模型成功创建了文件，并写了一首诗。

Wooo~ 是不是看起来很酷，和那种对话类的智能体感觉就不一样了~

当然，LangChain内置的工具除了文件操作类以外，还有计算类、搜索类等工具，这里就不一一列举，实际使用时可以再进行相关的搜索。

## 3.3 综合实践：把Memory和Tool组合起来

前面我们分别学了Memory（记忆）和Tool（工具），但实际应用中，我们需要把它们组合起来，才能构建出更强大的智能应用。比如“带记忆的实时信息查询助手”——既能记住你的查询历史，又能帮你查天气、查新闻。

前置安装，在本节的实践案例中引入数学计算工具，需要额外安装 \`langchain-experimental\` 包。
```bash
pip install langchain-experimental
```

### 3.3.1 实践1：带记忆的对话机器人

我们都知道大语言模型在数学计算上的能力比较差，因此智能体可以引入计算器来增强大模型的计算能力。

目标：构建一个能进行数学计算的对话助手，实现连贯的助手对话。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI  # 补充LLM定义
from langchain_experimental.tools import PythonREPLTool    # 数学计算工具
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import re
import os

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化LLM模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3  # 降低随机性，保证输出稳定
)

# 初始化数学计算工具（PythonREPL）
calc_tool = PythonREPLTool()
# 窗口记忆大小：保留最近2轮对话（每轮=用户+助手消息）
WINDOW_SIZE = 2

# -------------------------- 2. 定义提示词模板（适配窗口记忆+工具调用） --------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名友好的个人助手，规则如下：
    1. 能记住最近{window_size}轮对话内容，用简单语言解答问题；
    2. 如果问题包含数学计算（如加减乘除、公式、数值运算），先调用计算工具得到结果，再用自然语言解释；
    3. 非计算问题直接回答，记得结合历史对话上下文。"""),
    MessagesPlaceholder(variable_name="chat_history"),  # 窗口记忆注入点
    ("human", "{input}")  # 用户新问题
])

# -------------------------- 3. 工具调用逻辑（判断是否需要计算） --------------------------
def judge_and_calc(inputs):
    """
    核心逻辑：
    1. 检测用户问题是否包含数学计算需求
    2. 是：调用PythonREPLTool计算，再结合LLM生成回答
    3. 否：直接用LLM回答
    """
    user_input = inputs["input"]
    chat_history = inputs["chat_history"]
    
    # 简单的计算意图检测（可根据需求扩展）
    calc_pattern = r"(\+|\-|\×|\*|÷|/|=|计算|求和|求差|平方|立方|多少|等于)"
    is_calc_needed = bool(re.search(calc_pattern, user_input))
    
    if is_calc_needed:
        # 步骤1：调用计算工具执行运算
        try:
            # 提取计算表达式（简化版：取数字和运算符部分）
            calc_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", user_input)
            if not calc_expr:
                calc_result = "未识别到可计算的表达式"
            else:
                calc_result = calc_tool.run(calc_expr)
        except Exception as e:
            calc_result = f"计算出错：{str(e)}"
        
        # 步骤2：构造包含计算结果的提示，让LLM生成自然语言回答
        enhanced_input = f"""
        用户问题：{user_input}
        计算过程/结果：{calc_result}
        请结合计算结果，用简单易懂的语言回答用户问题，同时参考历史对话：{chat_history}
        """
        inputs["input"] = enhanced_input
    return inputs

# -------------------------- 4. 窗口记忆实现（仅保留最近N轮） --------------------------
# 会话存储：key=session_id，value=InMemoryChatMessageHistory
window_memory_store = {}

def get_window_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取带窗口限制的会话历史，自动截断超出长度的消息"""
    # 初始化会话记忆（无则创建）
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    
    history = window_memory_store[session_id]
    # 截断逻辑：保留最近WINDOW_SIZE轮（每轮2条消息：Human+AI）
    total_messages = len(history.messages)
    if total_messages > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE:]  # 只保留最后N轮
    
    return history

# -------------------------- 5. 构建完整的LCEL链（记忆+工具+LLM） --------------------------
# 核心链：参数传递 → 计算判断 → 提示词拼接 → LLM生成
chain = (
    RunnableLambda(judge_and_calc)
    | prompt
    | llm
)

# 注入窗口记忆功能
chain_with_window_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_window_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

# -------------------------- 6. 多轮对话测试 --------------------------
if __name__ == "__main__":
    session_id = "student_001"  # 每个用户独立会话ID，记忆隔离
    print("===== 带窗口记忆的数学计算智能助手 =====")
    print("支持：多轮对话、仅保留最近2轮记忆、自动数学计算")
    print("输入'退出'结束对话\n")
    
    while True:
        user_input = input("你：")
        if user_input in ["退出", "quit", "q"]:
            print("助手：再见！有问题随时问我～")
            break
        
        # 调用带窗口记忆的智能体
        response = chain_with_window_memory.invoke(
            {"input": user_input, "window_size": WINDOW_SIZE},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 输出回答（并将对话存入记忆）
        print(f"助手：{response.content}\n")
```

> 【LCEL优势】用管道符组合组件，逻辑非常清晰，后续要加工具的话，只需要在链路上再加一个`| 工具组件`，不用重构整个代码，非常灵活。
>
> 需要注意的是，PythonREPLTool 可执行任意代码，生产环境需限制可执行的表达式（如仅允许加减乘除）。

运行结果

```
你：168*12等于多少
Python REPL can execute arbitrary code. Use with caution.
助手：小熊，我帮你算了一下：

可以这样理解：
168 × 10 = 1680，再加上 168 × 2 = 336，
1680 + 336 = 2016。

有其他问题随时告诉我哦！ 😊

你：你：2开方是多少
助手：小熊，2的平方根约等于 **1.414**（更精确值是无限不循环小数）。
可以这样理解：
如果一个正方形的面积是2，它的边长就是√2 ≈ 1.414。

需要算其他数的话，随时告诉我哦！ 😊

你：我是谁
助手：小熊，根据我们之前的对话，我只知道你是正在和我聊天的用户，但你没有告诉我你的名字或身份呀～

如果你想让我记住你的称呼或信息，可以告诉我哦！我会尽力在接下来的对话中留意。 😊

你：668开方是多少
助手：小熊，668的平方根约等于 **25.8457**（保留四位小数）。

可以这样理解：
如果一个正方形的面积是668，它的边长就是√668 ≈ 25.85左右。

和之前算的√2类似，这也是一个无限不循环小数，通常取近似值使用。

需要算其他数的话，继续喊我～ 😊
```

### 3.3.2 实践2：带记忆的文件夹操作助手

ToolMessage 是工具调用的结果消息，会被加入对话历史，让 LLM 后续能参考工具执行结果回答问题。

目标：构建一个能查看文件、创建文件、写入文件的文件助手。核心组合：`Memory + Tool + Agent + LLM`

```python
# =========================
# 1. 基础依赖
# =========================
import os
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import AIMessage, ToolMessage

# =========================
# 2. 环境变量 & 模型
# =========================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="deepseek-chat",
    temperature=0.3
)

# =========================
# 3. 窗口记忆
# =========================
WINDOW_SIZE = 3
memory_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()

    history = memory_store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE:]
    return history

# =========================
# 4. 定义工具（@tool）
# =========================
@tool
def list_files(path: str = ".") -> str:
    """查看指定目录下的文件列表"""
    try:
        if not os.path.exists(path):
            return f"路径不存在：{path}"

        items = os.listdir(path)
        if not items:
            return "目录为空"

        result = []
        for item in items:
            full = os.path.join(path, item)
            if os.path.isfile(full):
                result.append(f"文件：{item}（{os.path.getsize(full)} 字节）")
            else:
                result.append(f"文件夹：{item}")
        return "\n".join(result)
    except Exception as e:
        return f"查看失败：{e}"

@tool
def create_file(path: str, content: str = "") -> str:
    """创建文件，并可写入初始内容"""
    try:
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"文件已创建：{path}"
    except Exception as e:
        return f"创建失败：{e}"

@tool
def write_file(path: str, content: str, append: bool = True) -> str:
    """向文件写入内容，支持追加或覆盖"""
    try:
        if not os.path.exists(path):
            return f"文件不存在：{path}"

        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        return f"写入成功（{'追加' if append else '覆盖'}）"
    except Exception as e:
        return f"写入失败：{e}"

@tool
def delete_file(path: str) -> str:
    """删除文件或空文件夹"""
    try:
        if not os.path.exists(path):
            return f"路径不存在：{path}"

        if os.path.isfile(path):
            os.remove(path)
            return f"文件已删除：{path}"

        if os.path.isdir(path):
            if os.listdir(path):
                return "文件夹非空，无法删除"
            os.rmdir(path)
            return f"文件夹已删除：{path}"

        return "无效路径"
    except Exception as e:
        return f"删除失败：{e}"

tools = [list_files, create_file, write_file, delete_file]

# =========================
# 5. Prompt（告诉模型：你可以用工具）
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个文件操作智能助手。"
     "当用户请求涉及文件或目录操作时，你可以自主决定是否调用工具。"
     "如果不需要工具，直接回答用户。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# =========================
# 6. 构建 Tool-Calling Agent
# =========================
agent = prompt | llm.bind_tools(tools)

agent_with_memory = RunnableWithMessageHistory(
    runnable=agent,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    session_id = "tool_agent_demo"

    print("===== 🧠 Tool Calling 文件 Agent =====")
    print("示例：")
    print(" - 查看当前文件夹")
    print(" - 创建文件 test.txt 内容 Hello")
    print(" - 写入文件 test.txt 内容 World 追加")
    print(" - 删除文件 test.txt")
    print("输入 q 退出\n")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["q", "quit", "退出"]:
            print("助手：再见 👋")
            break

        # ===== 第一次：模型思考 =====
        result = agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        history = get_session_history(session_id)

        print("\n🧠【模型输出】")
        if result.content:
            print(result.content)

        # ===== 模型决定调用工具 =====
        if isinstance(result, AIMessage) and result.tool_calls:
            print("\n🔧【模型决定调用工具】")
            for call in result.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]

                print(f"➡️ 工具名：{tool_name}")
                print(f"➡️ 参数：{tool_args}")

                tool_func = next(t for t in tools if t.name == tool_name)
                observation = tool_func.invoke(tool_args)

                print("\n📦【工具执行结果】")
                print(observation)

                history.add_message(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=str(observation)
                    )
                )

            print("\n✅【本轮结束：工具执行完成】\n")
            continue  # 回到 while True 等用户输入

        # ===== 最终回答（没有工具调用） =====
        print("\n✅【最终回答】")
        print(result.content, "\n")

```

运行结果

```
示例：
 - 查看当前文件夹
 - 创建文件 test.txt 内容 Hello
 - 写入文件 test.txt 内容 World 追加
输入 q 退出

你：查看当前文件夹

🧠【模型输出】
我来查看当前文件夹的内容。

🔧【模型决定调用工具】
➡️ 工具名：list_files
➡️ 参数：{'path': '.'}

📦【工具执行结果】
文件夹：easy-langent
文件：llm诗词.txt（157 字节）
✅【本轮结束：工具执行完成】

你：创建一个叫test.txt的文件 并在里面写一首诗

🧠【模型输出】
我来创建一个名为test.txt的文件，并在里面写一首诗。

🔧【模型决定调用工具】
➡️ 工具名：create_file
➡️ 参数：{'path': 'test.txt', 'content': '《静夜思》\n李白\n\n床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。\n\n《春晓》\n孟浩然\n\n春眠不觉晓，\n处处闻啼鸟。\n夜来风雨声，\n花落知多少。'}

📦【工具执行结果】
文件已创建：test.txt


你：查看当前文件夹

🧠【模型输出】
我来查看当前文件夹的内容，确认test.txt文件是否已创建成功。

🔧【模型决定调用工具】
➡️ 工具名：list_files
➡️ 参数：{'path': '.'}

📦【工具执行结果】
文件夹：easy-langent
文件：llm诗词.txt（157 字节）
文件：test.txt（214 字节）

✅【本轮结束：工具执行完成】

你：你是谁

🧠【模型输出】
我是一个文件操作智能助手。我可以帮助您进行文件相关的操作，比如：

1. 查看目录下的文件列表
2. 创建文件并写入内容
3. 向文件写入内容（支持追加或覆盖模式）

刚才我已经按照您的要求创建了一个名为test.txt的文件，并在里面写了两首经典的中国古诗：李白的《静夜思》和孟浩然的《春晓》。

您可以看到当前文件夹中有两个文件：llm诗词.txt和test.txt。如果您需要对这些文件进行其他操作，比如查看内容、修改内容或删除文件，我都可以帮助您处理。

✅【最终回答】
我是一个文件操作智能助手。我可以帮助您进行文件相关的操作，比如：

1. 查看目录下的文件列表
2. 创建文件并写入内容
3. 向文件写入内容（支持追加或覆盖模式）
4. 删除文件或空文件夹

刚才我已经按照您的要求创建了一个名为test.txt的文件，并在里面写了两首经典的中国古诗：李白的《静夜思》和孟浩然的《春晓》。

您可以看到当前文件夹中有两个文件：llm诗词.txt和test.txt。如果您需要对这些文件进行其他操作，比如查看内容、修改内容或删除文件，我都可以帮助您处理。
```

Wooo~ 是不是看起来很酷，这是你自己实现的带有记忆功能的智能体，它能自主决定使用哪些工具。在实际的企业中，会有很多路由和工具模块，这样能进一步提高模型的能力。

### 3.3.3 组合实践总结

通过这两个实践案例，我们能总结出组件组合的核心思路——“模块化拆分+流水线组合”：

- 拆分：把应用拆成独立的“小模块”——记忆（管状态）、工具（管行动）、提问模板（管输入）、LLM（管思考）；
- 组合：用LCEL的管道符（|）把模块串成流水线，比如“用户输入 → 记忆提取 → 提问拼接 → Agent决策 → 工具调用 → LLM回答 → 记忆保存”；
- 扩展：要加新功能，只需加新模块（比如加翻译功能就加翻译工具），不用改整个流水线，开发效率很高。

## 3.4 本章小结

到这里，本章的内容就全部学完了。我们来回顾一下核心知识点：

1. Memory组件：解决LLM“健忘”的问题，重点掌握三种常用组件（全量、窗口、摘要）的适用场景和用法；
2. Tool组件：给LLM装“手脚”，让它能调用外部工具，重点掌握内置工具的使用和自定义工具的规范；
3. 组件组合：用LCEL实现模块流水线，重点掌握“记忆+工具+Agent”的组合思路，能构建带记忆、能行动的复杂应用。

接下来的学习重点：Agent的高级用法（比如多工具协同、复杂任务规划），这是构建智能应用的核心能力。大家可以先试着扩展今天的实践案例，巩固一下本章的知识！

## 3.5 本章练习

1. 复现本章核心案例：记忆模块、工具模块，确保所有案例均可成功运行并输出预期结果；
2. 综合实践选择 1 个案例在本地复习，确保所有案例可成功运行并输出预期结果；
3. 思考企业落地优化方向：
   - 记忆层：用 Redis/PostgreSQL 替代内存存储，实现记忆持久化；
   - 工具层：增加权限控制、重试机制、异常捕获（如文件写入失败提示）；
   - 性能层：缓存高频工具调用结果、限制记忆长度控制 Token 消耗；
   - 体验层：优化工具调用话术，让回答更自然。