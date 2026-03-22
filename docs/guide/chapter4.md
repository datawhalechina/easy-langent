#  第四章 LangChain应用级系统设计与RAG实践

## 前言

从这一章开始，我们就要从LangChain的基础用法正式“升级打怪”，进入应用级系统设计的领域啦！

如果说之前的学习是在“认识零件”，那这一章就是教大家如何把这些零件组装成能解决实际问题的“智能机器”。

核心重点是两个：链式工作流（把复杂任务拆成流水线）和RAG（给大模型装一个“实时知识库”）。

全程搭配实操代码，跟着敲就能上手，还会穿插很多“踩坑指南”，让大家少走弯路～

## 4.1 链式工作流设计：复杂任务的拆解与流转

先问大家一个问题：如果让你用大模型完成“从一篇新闻稿中提取关键信息，再生成摘要，最后写成社交媒体文案”的任务，你会怎么做？直接丢给大模型一句话让它搞定？大概率会得到一个不伦不类的结果——要么信息提取不全，要么文案风格不对。

这就像让一个厨师同时完成“买菜、洗菜、做菜、摆盘”，步骤越多越容易出错。

而链式工作流，就是把这些复杂任务拆成一个个小步骤，让它们按顺序（或按条件）依次执行，最终拿到满意结果。

### 4.1.1 链式工作流核心认知

#### 4.1.1.1 为什么需要结构化链式工作流？

咱们先搞懂“为什么要这么折腾”。直接调用大模型不好吗？还真不好，主要有三个痛点：

- **复杂任务hold不住**：大模型对多步骤任务的逻辑连贯性处理得不好，比如让它同时做“数据分析+结论生成+报告撰写”，很容易遗漏关键步骤；
- **调试和优化难**：如果直接出问题了，你根本不知道是哪个环节错了，就像拆盲盒一样；
- **灵活性差**：不同步骤可能需要不同的模型或工具（比如提取信息用轻量模型，写文案用创意模型），直接调用无法实现“混搭”。

而结构化链式工作流就解决了这些问题：把大任务拆成小步骤（每个步骤一个“组件”），每个组件专注做一件事，步骤之间可以传递数据，出问题了能精准定位，还能根据需求灵活替换某个环节的模型或工具。就像工厂的流水线，每个工位只负责一道工序，最终组装出合格产品。

#### 4.1.1.2 常见链式工作流类型与适用场景

LangChain 0.1.x 版本后，官方重构了核心架构，**旧版 SequentialChain/RouterChain/ParallelChain 已标记为遗留组件**，推荐使用基于 `Runnables` 的全新组件实现链式工作流。

| 组件                           | 核心逻辑                                                 | 适用场景                                                     |
| ------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| RunnableSequence / `|`         | `|` 运算符（推荐）                                       | 多个组件按固定顺序执行，前一个组件的输出作为后一个组件的输入 |
| RunnableBranch                 | 根据输入内容或条件进行判断，将请求动态分发到不同处理分支 | 多场景路由任务，如智能客服（订单问题 / 售后问题）、多主题问答（历史 / 科技 / 医疗） |
| RunnableParallel / RunnableMap | 对同一输入并行执行多个子组件，最终返回结构化结果（dict） | 多维度分析任务，如从“情感、逻辑、文采”等多个角度评价同一文本 |

>  小提示：在实际应用中，RunnableSequence（线性流转） 与 RunnableBranch（动态路由） 是最常用的两种模式。
>  新版 Runnables 体系的优势在于：组件更轻量、支持流式与异步执行、可自由组合嵌套，更适合构建复杂的应用级 AI 系统。

### 4.1.2 基于 RunnableSequence / `|` 运算符

线性链就像“多米诺骨牌”，一个倒下带动下一个，核心是“顺序执行、数据流转”。

咱们从简单的开始练手。

#### 4.1.2.1 核心原理：Runnables 的线性流转逻辑

核心逻辑超简单：把“提示词模板（PromptTemplate）+ 大模型（LLM）”看作一个个独立的 Runnable 组件，用 `|` 运算符按顺序串联，第一个组件的输出自动作为第二个组件的输入，以此类推。就像接力赛，每个运动员跑完自己的棒，把接力棒交给下一个。

举个例子：“提取新闻关键信息→生成新闻摘要”。第一步（组件1）：PromptTemplate（提取信息）+ LLM，输入新闻原文，输出关键信息；第二步（组件2）：PromptTemplate（生成摘要）+ LLM，输入关键信息，输出新闻摘要。用 `|` 串联两个组件，顺序执行就能拿到最终结果。

#### 4.1.2.2 基础实践：单输入输出线性流转

先做一个简单任务：给一段产品介绍，先提取核心卖点，再根据卖点写一段营销话术。咱们一步步来，先准备环境，再写代码。

第一步：环境准备

【前置准备】所有案例需先完成环境配置：

```bash
# 安装完整依赖（新手必执行）
pip install langchain langchain-core langchain-openai python-dotenv
```

 **.env 文件**示例

```
 API_KEY=你的deepseek-api-key
```

**第二步：编写代码**

```python
# =========================
# 1. 基础依赖
# =========================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# =========================
# 2. 环境变量 & 模型
# =========================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# =========================
# 3. 组件1：提取核心卖点
# =========================
sell_point_prompt = PromptTemplate(
    input_variables=["product_intro"],
    template="请从以下产品介绍中提取3个核心卖点，用简洁的语言列出：{product_intro}"
)

sell_point_chain = sell_point_prompt | llm

# =========================
# 4. 中间结果结构化（LangChain 风格）
# =========================
extract_sell_points = RunnableLambda(
    lambda msg: {"sell_points": msg.content}
)

# =========================
# 5. 组件2：生成营销话术
# =========================
marketing_prompt = PromptTemplate(
    input_variables=["sell_points"],
    template="请根据以下产品核心卖点，写一段吸引消费者的营销话术（适合朋友圈发布）：{sell_points}"
)

marketing_chain = marketing_prompt | llm

# =========================
# 6. 线性串联（Sequential Runnable）
# =========================
overall_chain = (
    sell_point_chain
    | extract_sell_points
    | marketing_chain
)

# =========================
# 7. 执行
# =========================
product_intro = """这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟，支持高清通话；续航长达30小时，充电10分钟可使用2小时；机身采用亲肤硅胶材质，佩戴舒适，防水防汗，适合运动使用。"""

result = overall_chain.invoke({"product_intro": product_intro})

print("\n最终营销话术：")
print(result.content)

```

**第三步：代码解释与运行结果**

- **组件封装**：直接用 `PromptTemplate | llm` 封装成可执行组件，简洁高效；
- **串联方式**：使用 `|` 运算符串联组件，直观体现“流水线”式的执行逻辑；
- **执行与输出**：统一用 `invoke()` 方法执行组件，模型输出为 AIMessage 对象，需通过`.content` 属性获取文本内容；
- **数据适配**：可通过 lambda 函数转换数据格式（如将 AIMessage 转为字典），保障组件间数据流转顺畅。

运行结果示例：

```text
最终营销话术： 【运动新宠｜告别卡顿与电量焦虑】✨

戴上它，世界瞬间清晰！
✅ 蓝牙5.3芯片加持，通话如面对面般稳定流畅，再也不用担心运动时通话断断续续～
✅ 狂飙30小时超长续航，充电10分钟畅听2小时，电量焦虑？不存在的！
✅ 亲肤硅胶轻盈贴耳，狂汗暴雨也不怕，运动暴走一整天，舒适感依然在线！

🏃‍♂️💦 无论是晨跑、健身还是通勤，让它成为你随时在线的“舒适能量耳伴”！
👇点击体验，把高清畅听与自由运动装进口袋～
```

> 踩坑指南：
>
> 1. 若提示 API 密钥错误，先检查 .env 文件密钥是否正确，再用 `print(api_key)` 验证环境变量是否加载成功；
>
> 2. 若提示“输入变量缺失”，检查组件间数据格式是否匹配（比如用 lambda 函数转换 AIMessage 为字典）；
> 3. 替换其他厂商模型（如通义千问、文心一言）时，只需修改`base_url` 和 `model` 参数，核心逻辑不变。

#### 4.1.2.3 进阶实践：多输入多输出复杂线性任务

基础实践的 `|` 串联适合单输入单输出场景，若任务更复杂（比如“输入产品介绍和目标人群，先提取卖点，再根据卖点和人群写营销话术”），就需要用到 `RunnableSequence` 结合 `RunnablePassthrough` 实现多输入多输出

**编写代码**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from dotenv import load_dotenv
import os

# 1. 初始化模型
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# 2. Prompt 定义
sell_point_prompt = PromptTemplate(
    input_variables=["product_intro"],
    template="从以下产品介绍中提取3个核心卖点，简洁列出：{product_intro}"
)

marketing_prompt = PromptTemplate(
    input_variables=["sell_points", "target_audience"],
    template="针对{target_audience}，结合以下核心卖点，写一段朋友圈营销话术：{sell_points}"
)

# 3. 多输入多输出线性链（教学标准版）
overall_chain = (
    # Step 1：生成卖点 + 透传原始输入
    RunnableMap({
        "sell_points": sell_point_prompt | llm | (lambda x: x.content),
        "target_audience": RunnablePassthrough(),
    })
    # Step 2：营销话术生成
    | marketing_prompt
    | llm
)

# 4. 执行
input_data = {
    "product_intro": "这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟...",
    "target_audience": "大学生群体（喜欢运动、预算有限、注重性价比）"
}

result = overall_chain.invoke(input_data)

print("营销话术：")
print(result.content)
```

**代码解释与运行结果**

- `RunnableLambda` 是 LangChain 中用于将**任意自定义函数 / 逻辑**包装成 `Runnable` 接口的组件，让普通函数可以无缝接入 LangChain 的链式调用体系中。简单来说，它就是「普通函数」和「LangChain 可运行组件」之间的「适配器」。
- `RunnableMap` 是用于**并行执行多个 Runnable 组件**的容器，它接收一个字典（key 为自定义名称，value 为 Runnable 组件），执行时会并行调用所有子组件，并将每个组件的输出以「key: 输出结果」的形式汇总返回。
- `RunnablePassthrough` 是一个「透传组件」，它的核心作用是**原样传递输入数据**（或基于输入生成新数据后透传），常用来在链式调用中保留原始输入、补充数据或调整数据结构。

运行结果示例：

```text
营销话术：
【运动党必备！性价比逆天的蓝牙耳机来啦！】🎧

跑步总被耳机线缠到崩溃？健身时耳机滑落社死现场？电话打到一半突然断连？——你的运动耳机该升级了！

✨ 蓝牙5.3芯片加持，连接稳如泰山！操场狂奔、球场激战，音乐不断连，通话清晰无延迟，团战开黑再也不怕坑队友！

⚡ 续航狂魔登场！30小时超长续航，图书馆泡一整天都不用充电。支持快充，早上洗漱时间充10分钟，就能撑完两节体育课+晚间夜跑！

🏃‍♂️ 亲肤硅胶耳翼+IPX5防水，狂汗如雨也不滑落！轻若无物的佩戴感，跳绳撸铁毫无负担，下雨天晨跑照样戴～

学生价直接打穿底线！少喝两杯奶茶，就能拥有专业运动耳机，宿舍开黑/图书馆自习/操场暴汗全适配！

👇点击立抢专属学生优惠，跑赢新学期的第一圈！
```

是不是发现，针对特定人群的话术更精准了？

这就是多输入线性链的价值——能结合多个维度的信息完成任务。

## 4.2 路由链设计与异常处理：动态任务分发与系统稳健性

线性链适合“流程固定”的任务，但如果遇到“一个入口、多个场景”的情况就不行了。比如智能客服，用户可能问“查订单”“退货款”“问保修”，不同问题需要不同的处理流程。这时候就需要路由链——它像一个“智能分流员”，先判断用户需求，再把任务分配给对应的处理链。

另外，不管是线性链还是路由链，运行时都可能出问题（比如API调用失败、输入格式错误），所以异常处理也很重要，能让你的系统“不轻易崩溃”。

### 4.2.1 RouterChain（路由链）核心原理

#### 4.2.1.1 基于条件判断的动态任务分发机制

路由链的核心逻辑是“判断+分发”，就像学校的教务处：学生来办事，先问清楚“你要办什么事？”（判断），再指引到对应的科室（分发）——办成绩证明去A科，办学籍异动去B科。

在新版LangChain中，路由链基于Runnable范式实现，核心组成仍为三部分：

1. 目标链：多个处理不同场景的Runnable（如查订单链、退货款链）；
2. 路由选择器：通过条件判断或大模型理解，匹配输入对应的目标链；
3. 默认链：输入无法匹配任何目标链时，用于兜底处理的Runnable。

#### 4.2.1.2 关键组件：路由选择器工作机制

路由选择器是路由链的“大脑”，新版LangChain推荐用`RunnableBranch`实现条件路由，或用大模型驱动的路由判断。其核心工作流程：

1. 接收用户输入（通常是包含query的字典）；
2. 根据预设规则（硬编码条件或大模型判断）匹配目标链；
3. 将输入转发至匹配的目标链执行，无匹配则触发默认链。

新版中最常用的是“大模型+RunnableBranch”组合：利用大模型的自然语言理解能力解析用户需求，输出标准化标识后，由RunnableBranch完成分发，无需编写复杂的条件判断逻辑。

### 4.2.2 RouterChain 实践案例

#### 4.2.2.1 多场景需求的动态路由匹配（如客服咨询分类处理）

我们做一个简单的智能客服路由链：预设三个场景（查订单、退货款、保修政策），用户输入问题后，自动分发到对应的链处理，无法匹配则用默认链回复。采用新版LangChain写法（基于Runnable范式，使用ChatOpenAI模型）。

**第一步：环境准备**

和之前一样，确保安装了依赖包并配置了API密钥。

**第二步：编写代码**

```python
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

# 1. 加载环境变量与初始化模型（新版推荐用ChatOpenAI，支持聊天模型）
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# 2. 定义各场景的提示词模板与目标链（新版用RunnableSequence组合Prompt+LLM+Parser）
# 场景1：查订单链
order_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，负责解答用户的订单查询问题。"),
    ("human", "用户问题：{query}\n请引导用户提供订单号，并告知查询流程：1. 提供订单号；2. 系统验证；3. 反馈订单状态。")
])
order_chain = order_prompt | llm | StrOutputParser()  # 新版Runnable流水线写法

# 场景2：退货款链
refund_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，负责解答用户的退货款问题。"),
    ("human", "用户问题：{query}\n请说明退款流程：1. 申请退款（订单页面点击退款）；2. 等待审核（1-3个工作日）；3. 退款到账（原路返回，3-5个工作日）。如果用户问退款进度，引导提供退款申请单号。")
])
refund_chain = refund_prompt | llm | StrOutputParser()

# 场景3：保修政策链
warranty_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，负责解答产品保修政策问题。"),
    ("human", "用户问题：{query}\n请说明保修政策：本产品保修期限为1年，保修范围包括质量问题（非人为损坏），保修流程：1. 联系客服；2. 提供购买凭证；3. 寄回检测维修。")
])
warranty_chain = warranty_prompt | llm | StrOutputParser()

# 3. 定义路由判断逻辑（大模型解析需求，输出场景标识）
# 路由提示词：让大模型输出标准化的场景名称，用于后续分支匹配
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是路由选择器，需根据用户问题判断所属场景，仅输出以下标准化标识之一：
- order：订单查询相关（含订单状态、订单号）
- refund：退货款相关（含退款进度、退款申请）
- warranty：保修相关（含维修、售后保障）
- default：以上均不匹配
无需输出任何其他内容，仅返回标识字符串。
"""),
    ("human", "用户问题：{query}")
])

# 路由解析链：输入query，输出场景标识
router_chain = router_prompt | llm | StrOutputParser()

# 4. 定义默认链（兜底处理）
default_prompt = PromptTemplate(
    input_variables=["query"],
    template="抱歉，我无法解答你的问题'{query}'。请你重新描述问题，或者联系人工客服（工作时间：9:00-18:00）。"
)
default_chain = default_prompt | llm | StrOutputParser()

# 5. 构建完整路由链（核心：RunnableBranch实现条件分发）
# 逻辑：先通过router_chain获取场景标识，再由RunnableBranch分发到对应目标链
full_router_chain = RunnableLambda(lambda x: x) | (
    # 分支1：匹配order场景
    RunnableBranch(
        (lambda x: x["scene"] == "order", order_chain),
        (lambda x: x["scene"] == "refund", refund_chain),
        (lambda x: x["scene"] == "warranty", warranty_chain),
        default_chain  # 默认分支
    )
).with_config(run_name="full_router_chain")

# 6. 封装调用函数（整合场景解析与路由分发）
def process_query(query: str):
    # 第一步：获取场景标识
    scene = router_chain.invoke({"query": query})
    # 第二步：将query和scene传入完整路由链，执行分发处理
    return full_router_chain.invoke({"query": query, "scene": scene})

# 7. 测试不同场景输入
test_queries = [
    "我的订单什么时候发货？",
    "怎么申请退款呀？",
    "这个产品保修多久？",
    "你们家有什么新品？"  # 无法匹配，触发默认链
]

for query in test_queries:
    print(f"\n用户问题：{query}")
    print("客服回复：", process_query(query))

```

**代码解释与运行结果**

新版写法核心优势：基于Runnable范式，通过`|`（流水线）组合组件，逻辑更清晰，可扩展性更强。关键部分说明：

1. 目标链实现：用`ChatPromptTemplate | llm | StrOutputParser()`流水线替代旧版LLMChain，简洁且符合新版规范；
2. 路由分发核心：`RunnableBranch`是新版推荐的条件路由组件，通过“（条件判断函数，对应链）”的元组定义分支，无需手动编写分发逻辑；
3. 灵活性提升：路由解析链与业务处理链完全解耦，后续新增场景时，只需新增分支和目标链，无需修改整体架构。

运行结果示例：

```text
用户问题：我的订单什么时候发货？
客服回复： 您好！为了帮您查询订单的发货时间，请您提供一下订单号。查询流程如下：  
1. **提供订单号**：请告知您的订单号（通常可在订单确认邮件或账户订单页面找到）。
2. **系统验证**：我会根据订单号核实订单信息。
3. **反馈订单状态**：验证后，我会为您提供具体的发货进度或预计发货时间。

请提供订单号，我会尽快为您处理！ 📦

用户问题：怎么申请退款呀？
客服回复： 您好，申请退款的操作很简单，具体流程如下：

1. **提交申请**：请您进入【我的订单】页面，找到需要退款的订单，点击【申请退款】按钮，根据提示填写退款原因并提交。

2. **等待审核**：提交后，我们会在 **1-3个工作日** 内完成审核，审核结果会通过短信或站内消息通知您。

3. **退款到账**：审核通过后，退款会按原支付路径自动退回，一般 **3-5个工作日** 内到账（具体到账时间以银行或支付平台为准）。

如果您已经提交申请，想查询退款进度，可以告诉我您的 **退款申请单号**，我会帮您跟进处理。

用户问题：这个产品保修多久？
客服回复： 本产品提供**1年保修服务**，具体政策如下：  

**保修范围**：
- 产品在正常使用情况下出现的**质量问题**（非人为损坏）。

**保修流程**：
1. **联系客服**：通过官方渠道联系客服人员。
2. **提供凭证**：提供有效的购买凭证（如订单截图、发票等）。
3. **寄回检测**：按客服指引将产品寄回指定地址进行检测与维修。

如有其他疑问，可随时告知，我会为您进一步解答！ 😊

用户问题：你们家有什么新品？
客服回复： 您好！我理解您想了解新品信息，但作为AI助手，我无法获取实时产品目录或库存信息。不过，我可以为您提供几种快速获取信息的途径：

1. **官方渠道**
   📱 **官网/小程序**：访问品牌官方网站或微信小程序，通常会有“新品推荐”或“最新上市”专栏。
   📞 **客服热线**：直接致电官方客服（如您提到的工作时间9:00-18:00），他们能提供最准确的实时信息。

2. **线上平台**
   🛒 **电商旗舰店**：在淘宝、京东等平台的品牌旗舰店中，搜索“新品”或按“上新时间”筛选商品。

3. **其他建议**
   ✨ **订阅通知**：若您已注册会员，可关注品牌邮件或短信通知，新品常会优先推送。
   📍 **线下门店**：如有实体店，可咨询店员获取最新到店产品信息。

如果您需要我帮您查找特定品类的产品信息（如电子产品、美妆等），或整理新品选择的注意事项，我很乐意提供通用建议！请随时告诉我您的需求哦~ 🌟
```

> 小技巧：小技巧：若需提升路由判断精度，可在router_prompt中增加场景示例（如“例：用户问‘订单号在哪找’→输出order”），或调整提示词，进一步提升自然语言理解能力。

### 4.2.3 链式工作流的错误处理机制

不管是线性链还是路由链，运行时都可能遇到各种“意外”：API调用超时、输入格式错误、大模型返回空结果……新版LangChain基于Runnable范式提供了更优雅的错误处理方案，核心组件包括`RunnableRetry`（重试）、`RunnableWithFallback`（降级），结合Python原生异常捕获，构建完整的“安全防护网”。

#### 4.2.3.1 常见错误类型与触发场景

先搞清楚链式工作流中最容易遇到的错误：

| 错误类型     | 触发场景                                                     | 示例                                                         |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| API相关错误  | API密钥错误、网络中断、调用频率超限、模型服务宕机            | openai.AuthenticationError（密钥错误）、requests.exceptions.ConnectionError（网络中断） |
| 输入格式错误 | 输入缺少必要变量、格式不符合Prompt要求（如需字典却传入字符串） | MissingInputError（缺少输入变量）、ValueError（输入格式错误） |
| 输出解析错误 | 大模型返回结果不符合解析器要求（如需字符串却返回空值）       | OutputParserException（解析失败）                            |
| 业务逻辑错误 | 输入内容无法被业务逻辑处理（如查订单时输入非数字订单号）     | 订单号格式错误、无此订单记录                                 |

#### 4.2.3.2 工程化解决方案：重试机制、异常捕获与分支降级

针对上述错误，结合新版LangChain组件，提供三种核心解决方案：

**1. 重试机制（解决临时API错误）**

对于网络波动、API临时不可用等临时错误，使用`RunnableRetry`组件自动重试，支持自定义重试次数、间隔时间和重试条件。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# 1️⃣ Prompt
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "请简洁总结以下文本核心内容，不超过50字。"),
    ("human", "{text}")
])

# 2️⃣ 基础链
base_chain: Runnable = summary_prompt | llm | StrOutputParser()

# 3️⃣ 重试链（官方推荐：直接 with_retry）
retry_chain = base_chain.with_retry(
    stop_after_attempt=3,          # 最多重试 3 次
    wait_exponential_jitter=True,  # 指数退避 + 抖动（推荐）
    retry_if_exception_type=(
        ConnectionError,
        TimeoutError,
    ),
)

# 4️⃣ 调用
try:
    result = retry_chain.invoke({
        "text": "LangChain是一个用于构建大模型应用的框架，提供了丰富的Runnable组件，支持重试、降级等工程化能力。"
    })
    print("总结结果：", result)

except OutputParserException as e:
    # ❗ 解析错误通常是逻辑问题，不建议重试
    print("输出解析失败：", e)

except Exception as e:
    print("最终失败（已达到最大重试次数）：", e)

```

> 代码解释：`with_retry`是新版Runnable的内置方法，通过`RunnableRetry`配置重试规则，相比旧版`RetryWithErrorOutputChain`更灵活，可精准控制重试范围（如仅对临时API错误重试，避免无效重试）。

**2. 异常捕获（解决可预知的错误）**

对于输入格式错误、解析错误等可预知问题，结合Python原生`try-except`和LangChain异常类型，捕获错误后返回友好提示。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)
# 1️⃣ 定义多变量 Prompt 链（营销话术生成示例）
marketing_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据产品卖点和目标人群，撰写一句营销话术。"),
    ("human", "产品卖点：{sell_points}，目标人群：{target_audience}")
])

# 2️⃣ 构建链
marketing_chain: Runnable = marketing_prompt | llm | StrOutputParser()

# 3️⃣ 调用并捕获异常（官方推荐风格）
inputs = {
    "sell_points": "无线耳机续航30小时",
    # "target_audience" 故意缺失，用于演示 MissingInputError
}

try:
    result = marketing_chain.invoke(inputs)
    print("营销话术：", result)

except ValueError as e:
    # 解析异常信息，提取缺失的变量名
    if "Missing required variables" in str(e):
        missing_vars = str(e).split("Missing required variables: ")[-1].strip("[]'")
        print(f"错误提示：缺少必要输入变量 [{missing_vars}]，请检查输入数据。")
    else:
        print(f"输入参数错误：{e}")
except OutputParserException as e:
    # ✅ 官方推荐：逻辑解析错误不重试
    print(f"解析失败：{e}，请确认 Prompt 与输出格式匹配。")

except Exception as e:
    # ❗ 兜底捕获未知异常
    print(f"未知错误：{e}，请联系开发者排查。")

```

运行结果：`错误提示：缺少必要输入变量 'target_audience'，请检查输入数据是否完整。`

>  优势：新版LangChain定义了明确的异常类型（如`MissingInputError`、`OutputParserException`），可精准捕获特定错误，避免“一刀切”的异常处理，便于调试和用户理解。

**3. 分支降级（错误时切换备用方案）**

核心链失败时，使用`RunnableWithFallbacks`自动切换到备用链（如小模型、预设模板），保证系统可用性。相比旧版自定义函数，新版写法更简洁、可复用。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithFallbacks 
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
import os
from dotenv import load_dotenv

load_dotenv()

# 🔑 使用 OpenAI 的 API 密钥（从环境变量读取）
api_key = os.getenv("OPENAI_API_KEY")   # 在 .env 中设置 OPENAI_API_KEY=sk-xxxxx
# 注意：不需要指定 base_url，默认就是 https://api.openai.com

# 1️⃣ 核心链（性能高但可能不稳定）
core_llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4",              
    temperature=0.7
)
core_prompt = ChatPromptTemplate.from_messages([
    ("system", "用专业语言详细解答用户问题。"),
    ("human", "{query}")
])
core_chain = core_prompt | core_llm | StrOutputParser()

# 2️⃣ 降级链（稳定但精度略低）
fallback_llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.5
)
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "用简洁语言解答用户问题，保证信息准确。"),
    ("human", "{query}")
])
fallback_chain = fallback_prompt | fallback_llm | StrOutputParser()

# 3️⃣ 构建带降级的链
chain_with_fallback: RunnableWithFallbacks = core_chain.with_fallbacks(
    fallbacks=[fallback_chain],
    exceptions_to_handle=(ConnectionError, TimeoutError),# ✅ 官方推荐：只捕获临时错误或网络错误
)

# 4️⃣ 调用链并捕获异常
try:
    result = chain_with_fallback.invoke({"query": "什么是RAG技术？"})
    print("解答：", result)
except OutputParserException as e:
    print(f"解析失败：{e}")
except Exception as e:
    print(f"最终失败：{e}")

```

代码解释：如果核心链（gpt-4）因为API错误、超时而失败，就会自动切换到降级链（gpt-3.5-turbo-instruct），保证用户能得到解答，只是可能在回答质量上略有差异。这在工程实践中非常重要，能极大提升系统的可用性。

> 踩坑指南：异常捕获要精准，不要用“except Exception”捕获所有错误，尽量先捕获具体的错误类型（比如KeyError、AuthenticationError），再用通用异常捕获未知错误，这样方便后续调试。

## 4.3 RAG 核心原理与应用价值

讲完了链式工作流，我们来学习本章的另一个核心——RAG。

如果说链式工作流是“让大模型按步骤做事”，那RAG就是“让大模型有东西可依”。

不知道大家有没有遇到过这种情况：问大模型“我们公司2025年的产品规划是什么？”，大模型要么说“不知道”，要么胡编乱造一个答案。

这就是大模型的原生痛点——知识滞后、事实性差。

### 4.3.1 大模型原生痛点：知识滞后与事实性错误的根源

大模型的两个致命缺点，让它在企业级应用中“寸步难行”：

1. **知识滞后**：大模型的知识截止于训练数据的时间，比如2023年训练的模型，不知道2024年之后发生的事。就像一个人很久没上网，不知道最新的热点新闻；
2. **事实性错误（幻觉）**：大模型会“一本正经地胡说八道”，比如把张三的事安到李四身上，甚至编造不存在的文献、数据。新华社的报道中就提到，有大模型把文物“青铜利簋”的铸造者认错，还伪造了学术观点和作者信息。更严重的是，“语料污染”还会让大模型的错误率飙升，比如不法分子通过大量发布虚假信息，让大模型误判为可信信息，进而用于“AI杀猪盘”等诈骗活动。

为什么会这样？因为大模型的知识是“死”的，存在于训练参数中，无法实时更新，也无法验证事实的准确性。而RAG，就是给大模型装上一个“可更新的知识库”和“事实校验器”。

### 4.3.2 RAG 核心逻辑：检索增强的本质是“知识补充”与“事实校验”

RAG的全称是Retrieval-Augmented Generation（检索增强生成），核心逻辑超简单：在大模型生成答案之前，先从外部知识库中检索相关的真实、最新信息，把这些信息作为“参考资料”传给大模型，让大模型基于参考资料生成答案。

就像考试时，你遇到一个不会的问题，监考老师给你一本参考书，让你基于参考书的内容答题——这样既能保证答案的准确性，又能回答你原本不会的问题。RAG的流程可以总结为三步：

1. 检索：用户提问后，从外部知识库（比如公司文档、最新新闻、行业报告）中检索和问题相关的信息；
2. 增强：把检索到的信息和用户问题一起，作为提示词的一部分传给大模型；
3. 生成：大模型基于检索到的参考资料，生成准确、有依据的答案。

> 小提示：RAG的本质不是替代大模型，而是“赋能”大模型——让大模型能利用外部的、实时的、准确的知识来生成答案，解决知识滞后和事实性错误的问题。

### 4.3.3 RAG 的核心价值：提升应用准确性、拓展知识边界、降低微调成本

相比直接使用大模型，RAG有三个核心价值，也是企业为什么愿意用RAG的原因：

1. **提升准确性**：答案基于真实的参考资料，大大减少事实性错误。比如问公司内部政策，RAG会从公司文档中检索准确信息，而不是让大模型瞎编；
2. **拓展知识边界**：大模型可以获取训练数据之外的知识，包括最新信息（比如2025年的行业数据）和私有信息（比如公司内部文档、客户资料）；
3. **降低成本**：更新知识不需要重新训练大模型（微调成本很高，动辄几十万），只需要更新外部知识库（比如添加新的文档），成本低、效率高。

### 4.3.4 RAG 适用场景与不适用场景辨析

RAG不是万能的，要根据场景选择是否使用。下面列出适用和不适用的场景，帮大家避坑：

| 适用场景                                                     | 不适用场景                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 1. 企业内部知识库问答（比如员工手册、产品文档、技术手册）；2. 行业报告/政策文件问答（需要准确的事实和数据）；3. 最新信息问答（比如新闻资讯、赛事结果、股市行情）；4. 私有数据问答（比如客户资料、合同文档、内部会议纪要）。 | 1. 创造性任务（比如写诗歌、编故事、设计广告语，不需要真实参考资料）；2. 逻辑推理任务（比如数学题、编程题，核心是逻辑计算，不是外部知识）；3. 简单闲聊（比如“今天天气怎么样”，不需要复杂的知识库检索）。 |

举个例子：做一个“公司产品手册问答机器人”，适合用RAG；做一个“儿童故事生成器”，不适合用RAG。

## 4.4 RAG 系统构建全流程实操

这部分是实操重点，我们将一步步构建一个完整的RAG系统。

一个标准的RAG系统，从数据准备到最终问答，需要经过4个核心步骤：文档加载→文本分割→向量存储→检索与生成。

咱们逐个环节拆解，每个环节都配实操代码。

**重要前置说明**：LangChain 新版已将所有第三方文档加载器迁移至 `langchain-community` 库（核心库仅保留基础接口），因此先确保安装必备依赖： `pip install langchain-core langchain-community`

### 4.4.1 文档加载（Document Loading）

第一步是“文档加载”——把我们的知识库（比如PDF、Word、Markdown、TXT文件）加载到LangChain中，转换成统一的Document对象（LangChain定义的文档格式，包含文本内容和元数据）。

4.4.1.1 常见文档格式适配（PDF、Word、Markdown、TXT）

不同格式的文档，加载方式不同。LangChain提供了大量的文档加载器，覆盖主流格式。下面列出常见格式的加载器和依赖包：

| 文档格式        | 官方推荐加载器                                             | 需要安装的依赖包                                             |
| --------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| TXT             | TextLoader                                                 | 无需额外安装（langchain-community 内置）                     |
| PDF             | PyPDFLoader（基础款）/ PDFPlumberLoader（复杂款）          | 基础款：`pip install pypdf`；复杂款：`pip install pdfplumber` |
| Word（.docx）   | Docx2txtLoader                                             | `pip install docx2txt`                                       |
| Markdown（.md） | MarkdownLoader（轻量款）/ UnstructuredFileLoader（通用款） | 轻量款：`pip install python-markdown`；通用款：`pip install unstructured` |

#### 4.4.1.2 实操：多格式文档加载代码实现

光说不练假把式！下面咱们逐个实现不同格式文档的加载，代码里都加了详细注释，跟着敲就行。先统一说明：所有文档都放在一个叫「knowledge_base」的文件夹里，建议大家先创建这个文件夹，再把测试文档放进去（比如放一个test.txt、test.pdf、test.docx、test.md）。

**1. TXT文档加载（最简单的入门案例）**

TextLoader 是官方推荐的TXT专属加载器，轻量无冗余依赖，支持自定义编码，适配中文文档。

```python
# 导入TXT加载器（新版路径：langchain_community.document_loaders）
from langchain_community.document_loaders import TextLoader
import os

# 定义文档路径（请替换成你自己的路径）
txt_path = os.path.join("knowledge_base", "test.txt")

# 初始化加载器并加载文档
loader = TextLoader(txt_path, encoding="utf-8")  # 指定编码，避免中文乱码
txt_docs = loader.load()  # load()返回Document对象列表（即使只有一个文档）

# 查看加载结果
print("TXT文档加载结果：")
print(f"文档数量：{len(txt_docs)}")
print(f"文档内容：{txt_docs[0].page_content[:200]}...")  # 打印前200个字符
print(f"文档元数据：{txt_docs[0].metadata}")  # 元数据包含文档路径等信息
```

> 踩坑指南：如果加载中文TXT出现乱码，大概率是编码问题！把encoding参数改成"gbk"试试（有些旧TXT是gbk编码）。

**2. PDF文档加载（企业场景最常用）**

官方推荐基础场景用 PyPDFLoader（轻量高效），复杂场景（需保留表格、精准页码、格式）用 PDFPlumberLoader。两者均会按页拆分文档，生成独立Document对象，方便后续处理。

```python
# 方案1：基础款（官方推荐，仅提取文本，轻量）
from langchain_community.document_loaders import PyPDFLoader
import os

# 定义PDF路径
pdf_path = os.path.join("knowledge_base", "test.pdf")

# 初始化加载器并加载（按页拆分）
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load_and_split()  # load_and_split()直接按页拆分，更易用

# 查看结果
print("\nPDF文档加载结果（基础款）：")
print(f"PDF总页数：{len(pdf_docs)}")
print(f"第1页内容：{pdf_docs[0].page_content[:200]}...")
print(f"第1页元数据：{pdf_docs[0].metadata}")  # 元数据包含页码、文档路径

# 方案2：复杂款（需保留表格/格式时用）
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader(pdf_path)
pdf_docs_adv = loader.load()
print("\nPDF文档加载结果（复杂款）：")
print(f"第1页表格/格式保留情况：{pdf_docs_adv[0].page_content[:200]}...")
```

**3. Word文档加载（适配.docx格式）**

在当前常见的 Python 教学环境中（例如 `langchain_community==0.4.1`），可直接使用 `Docx2txtLoader` 加载 `.docx` 文档。运行前需先安装 `docx2txt` 依赖。

```python
# 导入Word加载器（兼容 langchain_community 0.4.1）
from langchain_community.document_loaders import Docx2txtLoader
import os

# 定义Word路径
docx_path = os.path.join("knowledge_base", "test.docx")

# 加载文档（需提前安装 docx2txt：pip install docx2txt）
loader = Docx2txtLoader(docx_path)
docx_docs = loader.load()

# 查看结果
print("\nWord文档加载结果：")
print(f"文档数量：{len(docx_docs)}")
print(f"文档内容：{docx_docs[0].page_content[:200]}...")
print(f"元数据：{docx_docs[0].metadata}")
```

> 踩坑指南：1. 仅支持 .docx 格式，旧版 .doc 需先转成 .docx；2. 若Word含大量图片，图片内容无法直接提取，需额外集成OCR工具。

**4. Markdown文档加载（技术文档常用）**

官方推荐轻量款 MarkdownLoader（保留标题层级、列表等结构），避免使用重量级的 UnstructuredMarkdownLoader（依赖复杂）。若需适配多格式通用场景，可备选 UnstructuredFileLoader。

```python
# 方案1：轻量款（官方推荐，保留MD结构，优先选）
from langchain_community.document_loaders import MarkdownLoader
import os

# 定义MD路径
md_path = os.path.join("knowledge_base", "test.md")

# 加载文档（需提前安装python-markdown）
loader = MarkdownLoader(md_path)
md_docs = loader.load()

# 查看结果
print("\nMarkdown文档加载结果（轻量款）：")
print(f"文档数量：{len(md_docs)}")
print(f"文档内容（保留结构）：{md_docs[0].page_content[:200]}...")
print(f"元数据：{md_docs[0].metadata}")

# 方案2：通用款（适配多格式，含MD/TXT等）
from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(md_path, mode="elements")  # mode="elements"保留结构
md_docs_univ = loader.load()
print("\nMarkdown文档加载结果（通用款）：")
print(f"内容预览：{md_docs_univ[0].page_content[:200]}...")
```

**5. 批量加载多格式文档（实用技巧）**

如果知识库文件夹里有多种格式的文档，一个个加载太麻烦，咱们基于官方推荐加载器写个批量加载函数，自动识别格式并加载：

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, MarkdownLoader
)
import os

def batch_load_documents(folder_path):
    """
    批量加载文件夹内的所有官方支持格式文档（基于新版加载器）
    :param folder_path: 知识库文件夹路径
    :return: 所有文档的Document对象列表
    """
    all_docs = []
    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 跳过文件夹，只处理文件
        if os.path.isdir(file_path):
            continue
        # 根据文件后缀选择对应的官方推荐加载器
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)  # 基础款，复杂场景可替换为PDFPlumberLoader
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".md"):
                loader = MarkdownLoader(file_path)
            else:
                print(f"不支持的文件格式：{filename}")
                continue
            # 加载并添加到文档列表
            docs = loader.load()
            all_docs.extend(docs)
            print(f"成功加载：{filename}，生成{len(docs)}个Document对象")
        except Exception as e:
            print(f"加载失败：{filename}，错误信息：{str(e)}")
    return all_docs

# 测试批量加载
if __name__ == "__main__":
    knowledge_base_path = "knowledge_base"
    # 确保知识库文件夹存在
    if not os.path.exists(knowledge_base_path):
        os.makedirs(knowledge_base_path)
        print(f"已自动创建知识库文件夹：{knowledge_base_path}，请放入测试文档")
    else:
        all_docs = batch_load_documents(knowledge_base_path)
        print(f"\n批量加载完成，总Document对象数：{len(all_docs)}")
        # 打印每个文档的基本信息
        for i, doc in enumerate(all_docs):
            print(f"\n文档{i+1}：")
            print(f"内容预览：{doc.page_content[:100]}...")
            print(f"元数据：{doc.metadata}")
```

这个函数超实用！后续构建自己的RAG系统时，直接调用它就能加载整个知识库的文档，省了很多重复代码。

### 4.4.2 文本分割（Text Splitting）：让检索更精准的关键一步

加载完文档后，下一步要做的是「文本分割」——把大文档切成一个个小的文本片段（叫Chunks）。可能有同学会问：“直接用整个文档检索不行吗？” 还真不行！这里有两个核心原因：

1. **大模型有上下文窗口限制**：比如GPT-3.5-turbo-instruct最多支持4096个token（大概3000个中文字），如果文档太大，根本塞不进大模型的“脑子”里；
2. **检索精度低**：用整个文档去匹配用户问题，就像在一本厚厚的字典里找一个字却不翻索引——范围太大，很容易把不相关的内容也检索出来。而切成小片段后，能精准匹配和问题相关的局部内容。

举个例子：你问“LangChain的链式工作流有哪些类型？”，如果把包含链式工作流、RAG、向量存储的大文档整个拿去检索，可能会把RAG的内容也带出来；但如果切成“链式工作流类型”“RAG核心原理”等小片段，就能精准定位到需要的内容。

#### 4.4.2.1 核心原则：如何科学地分割文本？

分割文本不是“随便切”，要遵循两个核心原则，不然会影响后续检索效果：

- **相关性原则**：同一个片段内的内容要语义相关，不能把一个完整的知识点切成两半（比如把“线性链的定义”和“路由链的定义”混在一个片段，或把“线性链的定义”切成两段）；
- **大小适中原则**：片段不能太大（超过大模型上下文窗口），也不能太小（太小会丢失上下文，比如只切一个词）。一般建议中文片段长度在200-500字，英文在200-500个token。

#### 4.4.2.2 常见分割策略与LangChain推荐实现

LangChain v0.1+ 版本对文本分割器进行了标准化优化，核心推荐 `RecursiveCharacterTextSplitter` 作为默认分割器（适配绝大多数场景），同时保留针对性分割方案。以下是官方推荐的实操方案（兼容最新API）：

**1.默认推荐：RecursiveCharacterTextSplitter（智能语义分割）**

LangChain 官方推荐此分割器作为首选——它通过“分层分隔符优先级”实现语义感知分割，先按大分隔符（如空行）拆分段落，再按小分隔符（如句号、逗号）微调，直到片段长度符合要求，最大程度保证语义完整性。适配TXT、Markdown、PDF等绝大多数文本类型。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader   # LangChain推荐的基础文本加载器
from pathlib import Path  # 官方推荐用pathlib处理路径（比os.path更现代）

# 1. 加载文档（推荐用Path处理路径，避免跨系统兼容问题）
txt_path = Path("knowledge_base") / "test.txt"
loader = TextLoader(txt_path, encoding="utf-8")
txt_docs = loader.load()  # 返回Document对象列表（含内容+元数据）

# 2. 初始化分割器（LangChain推荐参数）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # 中文片段推荐长度：200-500字
    chunk_overlap=50,        # 重叠长度：建议为chunk_size的10%-20%，避免跨片段语义丢失
    length_function=len,     # 中文用len计数，英文可改用tiktoken.count_tokens
    separators=["\n\n", "\n", "。", "！", "？", "，", "；", "、"]  # 中文推荐分隔符优先级
)

# 3. 执行分割（split_documents为官方推荐方法，接收Document列表）
split_docs = text_splitter.split_documents(txt_docs)

# 4. 验证结果
print(f"原始文档数：{len(txt_docs)}")
print(f"分割后片段数：{len(split_docs)}")
print("\n前3个片段示例：")
for i, doc in enumerate(split_docs[:3]):
    print(f"\n片段{i+1}（字符数：{len(doc.page_content)}）：")
    print(doc.page_content.strip())
    print(f"片段元数据：{doc.metadata}")  # 保留原始文档路径等元数据（检索时有用）
    
```

关键说明：

- separators参数：官方建议按“语义颗粒度从大到小”排序，中文场景优先用空行、换行、句末标点，避免拆分完整句子；
- 元数据保留：分割后的每个片段会继承原始Document的元数据（如文件路径），这是LangChain推荐的实践——后续检索时可追溯内容来源；
- 长度计算：若需对接OpenAI等模型，可改用`tiktoken.count_tokens`计算token数（需安装tiktoken包），更贴合模型上下文限制。

**2. 基础备选：CharacterTextSplitter（按字符数分割）**

仅适用于结构极简单的文本（如纯文本日志），LangChain不推荐作为默认方案。核心优势是轻量、速度快，需手动保证语义完整性。

```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader 
from pathlib import Path

# 加载文档
txt_path = Path("knowledge_base") / "test.txt"
loader = TextLoader(txt_path, encoding="utf-8")
txt_docs = loader.load()

# 初始化（官方推荐设置自然分隔符）
text_splitter = CharacterTextSplitter(
    separator="\n\n",        # 优先按空行分割，减少语义破坏
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    keep_separator=False     # 官方默认False，不保留分隔符（避免片段冗余）
)

split_docs = text_splitter.split_documents(txt_docs)

# 验证结果
print(f"分割后片段数：{len(split_docs)}")
```

**3. 针对性分割：MarkdownTextSplitter（保留MD标题层级）**

针对Markdown文档（技术文档、笔记），LangChain推荐用`MarkdownTextSplitter`——它会识别MD标题层级（#、##、###），确保“标题+对应内容”不被拆分，后续检索时能通过标题快速定位主题。搭配`UnstructuredMarkdownLoader`加载文档（保留MD结构元数据）。

```python
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # 官方推荐MD加载器
from pathlib import Path

# 加载MD文档（保留标题层级元数据）
md_path = Path("knowledge_base") / "test.md"
loader = UnstructuredMarkdownLoader(str(md_path), mode="elements")  # Path转字符串兼容所有版本
md_docs = loader.load()

# 3. 初始化MD分割器（保留你原有核心参数，1.x版本完全兼容）
text_splitter = MarkdownTextSplitter(
    chunk_size=500,          # MD有标题引导，片段长度可稍大
    chunk_overlap=50,        # 重叠部分避免标题/内容割裂
    length_function=len,     # 中文用len计数字符
    separators=["\n\n", "# ", "## ", "### "]  # 优先按标题层级分割，保留语义
)

# 4. 执行分割（方法名split_documents在1.x版本无变化）
split_docs = text_splitter.split_documents(md_docs)

# 5. 验证结果（片段会保留MD标题层级，符合预期）
print(f"分割后片段数：{len(split_docs)}")
print("\n前2个MD片段示例：")
for i, doc in enumerate(split_docs[:2]):
    print(f"\n片段{i+1}：")
    print(doc.page_content.strip())
    
```

### 4.4.3 向量存储与嵌入（Vector Storage & Embedding）：给文本做“数字指纹”

分割完文本片段后，下一步要做的是「向量嵌入」和「向量存储」。

这一步是RAG能实现“精准检索”的核心——把文字变成计算机能理解的“数字”，再存到专门的向量数据库里，后续用户提问时，也把问题变成数字，通过比对数字的相似度，找到最相关的文本片段。

用通俗的话讲：每个文本片段就像一个人，向量嵌入就是给每个人做一个“指纹”（一串数字），向量数据库就是存指纹的“数据库”。用户提问时，先给问题做个指纹，再去数据库里找指纹最像的人（文本片段），就是最相关的内容。

#### 4.4.3.1 核心概念：嵌入模型（Embedding Model）

把文字变成向量的工具就是「嵌入模型」。好的嵌入模型能精准捕捉文本的语义——意思越接近的文本，向量越像（数字差值越小）。常见的嵌入模型有：

- 国内模型：qwen3-embedding；
- 开源模型：BERT、Sentence-BERT（可以本地部署，免费，适合隐私敏感场景）。

咱们实操用qwen3-embedding-0.6b，该模型基于通义千问大模型技术研发，轻量高效，支持中文语义精准捕捉，适合中小型RAG项目。

**前置准备**

首先安装最新版本的核心依赖包，确保LangChain相关组件完整性（1.x版本拆分了多个子包，需完整安装）：

```bash
pip install -U langchain langchain-core langchain-community langchain-openai faiss-cpu transformers torch python-dotenv
```

说明：`langchain-core`是LCEL的核心依赖，`langchain-community`提供向量库、本地嵌入模型等社区组件，缺一不可。

#### 4.4.3.2 向量数据库选择：faiss（轻量易上手）

向量需要存在专门的向量数据库里，方便后续快速检索。LangChain支持很多向量数据库，比如FAISS、Chroma、Pinecone等。咱们选FAISS——它是Meta开源的轻量级向量数据库，支持高效的相似性检索，不需要单独部署服务，直接在代码里初始化就能用，同时兼容多种嵌入模型，特别适合初学者和小型项目。

> 如果想进一步了解向量数据库相关知识，可学习datawhale社区《easy-vectordb》相关内容。链接：https://github.com/datawhalechina/easy-vectordb

#### 4.4.3.3 实操：文本嵌入与向量存储完整流程

流程：加载文档→分割文本→初始化嵌入模型→初始化向量数据库→将分割后的文本嵌入并存储。

qwen3-embedding-0.6b是一个相对比较轻量的嵌入模型，因此本节的教程推荐大家使用原生导入。地址：https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-Embedding-0.6B',cache_dir='./models')
```

实践代码

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

txt_path = os.path.join("knowledge_base", "test.txt")
if not os.path.exists(txt_path):
    raise FileNotFoundError(f"文档文件不存在：{txt_path}")

# 加载文本文档
loader = TextLoader(txt_path, encoding="utf-8")
txt_docs: list[Document] = loader.load()

# 文本分割（使用最新的 RecursiveCharacterTextSplitter 配置）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # 每个文本块的大小
    chunk_overlap=50,        # 块之间的重叠长度（提升上下文连续性）
    length_function=len,     # 长度计算函数（中文用len即可）
    is_separator_regex=False # 显式指定非正则分隔符（默认值，增加可读性）
)
split_docs: list[Document] = text_splitter.split_documents(txt_docs)
print(f"分割后的文本片段数：{len(split_docs)}")

# 3. 初始化本地CPU运行的嵌入模型（替换QwenEmbeddings）
embedding_model_name = "./models/Qwen/Qwen3-Embedding-0___6B"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu"  # 强制使用CPU运行，无需GPU
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 4. 构建并持久化FAISS向量库
try:
    # 生成向量并初始化FAISS（本地CPU计算，首次运行会下载模型，需联网）
    vector_db = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )

    # 持久化向量库到本地
    vector_db.save_local(
        folder_path="./faiss_db",
        index_name="local_cpu_faiss_index"  # 索引名改为本地CPU版
    )
    print("向量存储完成！向量数据已保存到 ./faiss_db 文件夹")
except Exception as e:
    raise RuntimeError(f"构建/保存向量库失败：{str(e)}")

# 5. 相似性检索测试
query = "LangChain的链式工作流有哪些类型？"
try:
    # 一次性获取带评分的检索结果
    retrieved_docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    
    print(f"\n与问题「{query}」最相关的3个文本片段：")
    for i, (doc, score) in enumerate(retrieved_docs_with_scores):
        print(f"\n片段{i+1}：")
        print(f"内容：{doc.page_content}")
        print(f"相关性评分（越小越相似）：{round(score, 4)}")
        print(f"来源：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"检索向量库失败：{str(e)}")
```

### 4.4.4 检索器配置与检索策略优化

向量存储好后，下一步就是「检索」——用户提问时，从向量数据库里找到最相关的文本片段。LangChain的「检索器」（Retriever）就是干这个活的，它封装了检索逻辑，还支持多种检索策略，能提升检索精度。

#### 4.4.4.1 基础检索器：FAISSRetriever

直接从FAISS向量数据库创建检索器，最基础的检索方式，支持相似性检索。

```python
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# 本地Qwen嵌入模型路径
embedding_model_name = "./models/Qwen/Qwen3-Embedding-0___6B"

# 初始化本地CPU运行的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu"  # 强制使用CPU运行
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 加载已有的FAISS向量数据库
try:
    vector_db = FAISS.load_local(
        folder_path="./faiss_db",  # 之前的持久化路径
        embeddings=embeddings,
        allow_dangerous_deserialization=True, 
        index_name="local_cpu_faiss_index"
    )
    print("FAISS向量库加载成功！")
except FileNotFoundError:
    raise FileNotFoundError("未找到 ./faiss_db 文件夹，请确认向量库已正确保存")
except Exception as e:
    raise RuntimeError(f"加载FAISS向量库失败：{str(e)}")

# 创建检索器（v0.1+ 规范）
retriever: BaseRetriever = vector_db.as_retriever(
    search_kwargs={"k": 3},  # 每次检索返回3个最相关的片段
    # 可选：按分数阈值检索（按需启用）
    # search_type="similarity_score_threshold",
    # search_kwargs={"k": 3, "score_threshold": 0.5}
)

# 测试检索
query = "LangChain的SequentialChain有什么用？"
try:
    # 核心修改：替换 get_relevant_documents → invoke（Runnable接口标准）
    retrieved_docs: list[Document] = retriever.invoke(query)

    print(f"\n检索到的相关片段（{len(retrieved_docs)}个）：")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n片段{i+1}：")
        print(f"内容：{doc.page_content}")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")

    # 如需获取检索评分（补充完整可运行的评分获取逻辑）
    print("\n===== 带评分的检索结果 =====")
    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\n片段{i+1}（相关性评分：{round(score, 4)}）：")
        print(f"内容：{doc.page_content}")
        
except Exception as e:
    raise RuntimeError(f"检索向量库失败：{str(e)}")
```

检索器的优势：可以直接和LangChain的链结合（比如后续的检索-生成链），不用单独写检索逻辑，简化代码。

#### 4.4.4.2 进阶检索策略：相似性检索 vs MMR检索

基础的相似性检索（Similarity Search）可能会有一个问题：检索到的片段内容太相似，导致信息冗余。

而MMR（Maximum Marginal Relevance，最大边际相关性）检索能解决这个问题——在保证相关性的同时，尽量选择多样化的片段，避免冗余。

比如你问“LangChain的链有哪些？”，相似性检索可能返回3个都讲线性链的片段；而MMR检索会返回1个线性链、1个路由链、1个并行链的片段，信息更全面。

```python
import os
from langchain_community.vectorstores import FAISS  # v0.1+ 正确导入路径
from langchain_huggingface import HuggingFaceEmbeddings  # 本地模型用这个
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# 1. 配置本地Qwen嵌入模型（替换原API版QwenEmbeddings）
embedding_model_name = "./models/Qwen/Qwen3-Embedding-0___6B"

# 初始化本地CPU运行的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu",  # 强制CPU运行，无需GPU
        # 如需加载量化模型，可添加以下配置（按需）
        # "trust_remote_code": True,
        # "load_in_8bit": False
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 2. 加载FAISS向量数据库
try:
    vector_db = FAISS.load_local(
        folder_path="./faiss_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="local_cpu_faiss_index"  # 需和保存时的索引名一致
    )
    print("FAISS向量库加载成功！")
except FileNotFoundError:
    raise FileNotFoundError("未找到 ./faiss_db 文件夹，请确认向量库已正确保存")
except Exception as e:
    raise RuntimeError(f"加载FAISS向量库失败：{str(e)}")

# 3. 创建不同类型的检索器（保留相似性+MMR对比）
# 3.1 相似性检索（默认）
retriever_similar: BaseRetriever = vector_db.as_retriever(
    search_type="similarity",  # 检索类型：纯相似性
    search_kwargs={"k": 3}     # 返回3个最相似的片段
)

# 3.2 MMR检索（最大化边际相关性，兼顾相关性和多样性）
retriever_mmr: BaseRetriever = vector_db.as_retriever(
    search_type="mmr",  # 检索类型：MMR
    search_kwargs={
        "k": 3,         # 最终返回3个片段
        "fetch_k": 10,  # 先检索10个最相关的，再从中选多样化的
        "lambda_mult": 0.5  # 权重：0=只看多样性，1=只看相关性，0.5平衡
    }
)

# 4. 测试对比两种检索方式（适配v0.1+的invoke方法）
query = "LangChain的链有哪些类型？"

# 4.1 相似性检索测试
print("=== 相似性检索结果 ===")
try:
    similar_docs: list[Document] = retriever_similar.invoke(query)  # 替换get_relevant_documents
    for i, doc in enumerate(similar_docs):
        print(f"\n片段{i+1}：{doc.page_content[:100]}...")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"相似性检索失败：{str(e)}")

# 4.2 MMR检索测试
print("\n=== MMR检索结果 ===")
try:
    mmr_docs: list[Document] = retriever_mmr.invoke(query)  # 替换get_relevant_documents
    for i, doc in enumerate(mmr_docs):
        print(f"\n片段{i+1}：{doc.page_content[:100]}...")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"MMR检索失败：{str(e)}")

# 可选：补充MMR检索的评分对比（便于理解差异）
print("\n===== 相似性检索（带评分） =====")
similar_docs_with_score = vector_db.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(similar_docs_with_score):
    print(f"片段{i+1}（评分：{round(score,4)}）：{doc.page_content[:80]}...")
```

>  小技巧：如果你的知识库内容比较单一，用相似性检索就行；如果知识库内容丰富，需要全面的信息，用MMR检索更好。lambda_mult参数可以调，比如0.7更偏向相关性，0.3更偏向多样性。

#### 4.4.4.3 实操案例：检索器参数配置与检索效果验证

咱们通过一个完整案例，验证不同检索器参数对检索效果的影响。

**第一步：环境准备与向量数据库加载**

沿用之前创建的FAISS向量数据库（以qwen-embedding-0.6b为例），先完成环境初始化和数据库加载：

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_name = "./models/Qwen/Qwen3-Embedding-0___6B"

# 初始化本地CPU运行的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu",  # 强制CPU运行，无需GPU
        # 如需加载量化模型，可添加以下配置（按需）
        # "trust_remote_code": True,
        # "load_in_8bit": False
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 加载已有的FAISS向量数据库
vector_db = FAISS.load_local(
    folder_path="./faiss_db",  # 之前存储向量的路径
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
print("向量数据库加载成功！")
```

**第二步：配置不同参数的检索器**

我们配置4种不同参数的检索器，分别验证“返回数量（k值）”“检索类型（相似性/MMR）”“多样性权重（lambda_mult）”对结果的影响：

```python
retriever_similar_k2 = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# 4.2 相似性检索（k=5）
retriever_similar_k5 = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 4.3 MMR检索（偏向相关性，lambda_mult=0.8）
retriever_mmr_high_rel = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.8}
)

# 4.4 MMR检索（偏向多样性，lambda_mult=0.3）
retriever_mmr_high_div = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.3}
)

# 5. 定义测试查询并执行检索（适配v0.1+ invoke方法）
test_query = "RAG系统的核心价值是什么？"
```

**第三步：执行检索并对比结果**

分别调用4个检索器，打印检索结果，对比不同参数的效果差异：

```python
def test_retriever(retriever: BaseRetriever, retriever_name: str):
    """测试检索器并打印结果"""
    try:
        # 核心适配：使用invoke()替代旧的get_relevant_documents()
        results: list[Document] = retriever.invoke(test_query)
        print(f"=== {retriever_name} 检索结果（共{len(results)}条） ===")
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] 内容：{doc.page_content[:120]}...")
            print(f"   来源：{doc.metadata.get('source', '未知')}")
        print("\n" + "-"*80 + "\n")
    except Exception as e:
        raise RuntimeError(f"{retriever_name} 检索失败：{str(e)}")

# 执行所有检索并打印结果
test_retriever(retriever_similar_k2, "相似性检索（k=2）")
test_retriever(retriever_similar_k5, "相似性检索（k=5）")
test_retriever(retriever_mmr_high_rel, "MMR检索（偏向相关性 λ=0.8）")
test_retriever(retriever_mmr_high_div, "MMR检索（偏向多样性 λ=0.3）")

# 可选：补充相似性评分展示（FAISS特有）
print("=== 相似性检索（k=3）带评分结果 ===")
docs_with_scores = vector_db.similarity_search_with_score(test_query, k=3)
for i, (doc, score) in enumerate(docs_with_scores):
    print(f"\n[{i+1}] 评分：{round(score, 4)} | 内容：{doc.page_content[:80]}...")
```

**第四步：检索结果分析与参数选择建议**

运行上述代码后，会得到类似以下的结果（忽视内容部分），也可以根据自己的知识库自己的问题来比较

我们结合结果分析参数的影响：

```text
向量数据库加载成功！

=== 相似性检索（k=2） 检索结果 ===

片段1（字符数：286）：
RAG 的核心价值：提升应用准确性、拓展知识边界、降低微调成本
相比直接使用大模型，RAG有三个核心价值，也是企业为什么愿意用RAG的原因：1. 提升准确性：答案基于真实的参考资料，大大减少事实性错误。比如问公司内部政策，RAG会从公司文档中检索准确信息，而不是让大模型瞎编；2. 拓展知识边界：大模型可以获取训练数据之外的知识...
相似度评分：0.1862（越小越相关）

片段2（字符数：215）：
1. 提升准确性：答案基于真实的参考资料，大大减少事实性错误。比如问公司内部政策，RAG会从公司文档中检索准确信息，而不是让大模型瞎编；2. 拓展知识边界：大模型可以获取训练数据之外的知识，包括最新信息（比如2025年的行业数据）和私有信息（比如公司内部文档、客户资料）；3. 降低成本：更新知识不需要重新训练大模型（微调成本很高，动辄几十万）...
相似度评分：0.1987（越小越相关）

=== 相似性检索（k=5） 检索结果 ===
（前2个片段与上述一致，新增3个相关性较低的片段）
片段3（字符数：198）：
RAG的全称是Retrieval-Augmented Generation（检索增强生成），核心逻辑超简单：在大模型生成答案之前，先从外部知识库中检索相关的真实、最新信息，把这些信息作为“参考资料”传给大模型，让大模型基于参考资料生成答案。
相似度评分：0.2345
...

=== MMR检索（偏向相关性，lambda=0.8） 检索结果 ===

片段1（字符数：286）：
RAG 的核心价值：提升应用准确性、拓展知识边界、降低微调成本
相比直接使用大模型，RAG有三个核心价值，也是企业为什么愿意用RAG的原因：1. 提升准确性：答案基于真实的参考资料，大大减少事实性错误。比如问公司内部政策，RAG会从公司文档中检索准确信息，而不是让大模型瞎编；2. 拓展知识边界：大模型可以获取训练数据之外的知识...

片段2（字符数：198）：
RAG的全称是Retrieval-Augmented Generation（检索增强生成），核心逻辑超简单：在大模型生成答案之前，先从外部知识库中检索相关的真实、最新信息，把这些信息作为“参考资料”传给大模型，让大模型基于参考资料生成答案。

片段3（字符数：182）：
大模型的两个致命缺点，让它在企业级应用中“寸步难行”：1. 知识滞后：大模型的知识截止于训练数据的时间，比如2023年训练的模型，不知道2024年之后发生的事。就像一个人很久没上网，不知道最新的热点新闻；2. 事实性错误（幻觉）：大模型会“一本正经地胡说八道”...

=== MMR检索（偏向多样性，lambda=0.3） 检索结果 ===

片段1（字符数：286）：
RAG 的核心价值：提升应用准确性、拓展知识边界、降低微调成本
相比直接使用大模型，RAG有三个核心价值，也是企业为什么愿意用RAG的原因：1. 提升准确性：答案基于真实的参考资料，大大减少事实性错误。比如问公司内部政策，RAG会从公司文档中检索准确信息，而不是让大模型瞎编；2. 拓展知识边界：大模型可以获取训练数据之外的知识...

片段2（字符数：175）：
RAG适用场景与不适用场景辨析
RAG不是万能的，要根据场景选择是否使用。下面列出适用和不适用的场景，帮大家避坑：适用场景：1. 企业内部知识库问答（比如员工手册、产品文档、技术手册）；2. 行业报告/政策文件问答...

片段3（字符数：168）：
向量存储与嵌入（Vector Storage & Embedding）：给文本做“数字指纹”
分割完文本片段后，下一步要做的是「向量嵌入」和「向量存储」。这一步是RAG能实现“精准检索”的核心——把文字变成计算机能理解的“数字”，再存到专门的向量数据库里...
```

基于结果，总结参数选择建议：

- **k值（返回数量）**：不宜过大或过小。k=2-3适合精准定位核心信息，避免冗余；k=5及以上适合需要全面信息覆盖的场景（比如复杂问题解答），但会引入相关性较低的内容，增加大模型处理负担。实际应用中建议k=3-4。
- **检索类型**：简单问题（如“RAG核心价值是什么”）用相似性检索即可，高效精准；复杂问题（如“对比RAG与微调的优劣”）用MMR检索，能获取多样化信息，避免信息片面。
- **lambda_mult（MMR多样性权重）**：核心问题优先保证相关性，建议lambda_mult=0.7-0.8；需要多角度分析的问题可适当提升多样性，建议lambda_mult=0.4-0.5；不建议低于0.3，否则会引入无关信息。

>  踩坑指南：参数没有“最优值”，需结合知识库规模和业务场景调试！比如小知识库（100个片段以内），k=2即可；大知识库（1000+片段），k可适当增大到4-5，同时用MMR检索保证信息多样性。

### 4.4.5 检索-生成（Retrieval-Generation）全流程整合

前面我们已经完成了“文档加载→文本分割→向量存储→检索”的所有环节，最后一步是“生成”——将检索到的相关片段和用户问题结合，传给大模型生成最终答案。基于LangChain最新推荐的LCEL（LangChain Expression Language）风格，我们采用FAISS向量数据库和本地Qwen3嵌入模型，搭建端到端的RAG问答系统。

#### 4.4.5.1 核心原理：检索与生成的协同逻辑

核心逻辑：用户提问→检索器从FAISS向量数据库获取相关片段→将“用户问题+相关片段”按指定规则拼接→传给大模型→大模型基于相关片段生成答案。

LangChain 1.x版本通过LCEL实现检索与生成的模块化协同，核心优势在于**组件化串联**与**数据流式传递**：无需依赖`create_retrieval_chain`等封装函数，直接通过管道符（|）将检索器、文档格式化函数、提示词模板、大模型、输出解析器等组件串联成链。数据在链中按顺序流转，上一个组件的输出自动作为下一个组件的输入，既简化了代码逻辑，又提升了自定义灵活性（如可随时插入日志、数据过滤等自定义组件）。

#### 4.4.5.2 实操：完整RAG系统代码实现

基于前面构建的FAISS向量数据库和检索器，搭建端到端的RAG问答系统，全程使用本地CPU运行嵌入模型，无需GPU依赖。以下是LangChain 1.x版本官方推荐写法：

**第一步：环境准备与组件初始化**

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # 1.x推荐用ChatOpenAI，适配对话模型，功能更全
from langchain_core.prompts import ChatPromptTemplate  # 替代PromptTemplate，适配LCEL
from langchain_core.runnables import RunnablePassthrough  # LCEL核心组件，传递数据
from langchain_core.output_parsers import StrOutputParser  # 统一输出格式解析
from dotenv import load_dotenv
import os

# 加载并验证环境变量（1.x推荐显式验证，避免配置缺失）
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未找到OPENAI_API_KEY环境变量，请检查.env文件配置")

# 1. 初始化本地CPU运行的Qwen3嵌入模型（参数兼容1.x版本，保持原推荐配置）
embedding_model_path = "./models/Qwen/Qwen3-Embedding-0___6B"
# 验证模型路径有效性
if not os.path.exists(embedding_model_path):
    raise FileNotFoundError(f"Qwen3嵌入模型路径不存在：{embedding_model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={
        "device": "cpu",  # 强制CPU运行，无需GPU
        "trust_remote_code": True,  # Qwen3模型必选配置，信任远程代码
        # 如需加载量化模型，可启用以下配置（降低内存占用）
        # "load_in_8bit": True,
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索相似度计算精度
    }
)

# 2. 加载FAISS向量数据库（1.x版本用法完全兼容，保留原逻辑）
# 注意：首次构建时需用FAISS.from_documents(docs, embeddings)创建并save_local
vector_db = FAISS.load_local(
    folder_path="./faiss_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # 本地开发可用，生产环境需谨慎（存在安全风险）
)

# 3. 初始化检索器（MMR策略，平衡相关性和多样性，参数无变化）
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)

# 4. 初始化大模型（1.x推荐用ChatOpenAI，支持更丰富的对话配置）
llm = ChatOpenAI(
    api_key=api_key,
    temperature=0.3,  # 低温度保证答案精准，减少幻觉
    timeout=30,  # 新增超时配置，避免请求挂起（1.x官方推荐）
    max_retries=2  # 网络波动时自动重试，提升稳定性
)
```

**第二步：用LCEL构建检索-生成链（最新推荐写法）**

通过LCEL管道符（|）串联组件，逻辑更清晰、自定义性更强。核心步骤：格式化检索文档→拼接提示词→大模型生成→解析输出。

```python
# 1. 自定义文档格式化函数（将检索到的多个文档拼接为统一文本，供提示词使用）
def format_docs(docs):
    """格式化检索到的文档片段，用空行分隔"""
    return "\n\n".join([doc.page_content for doc in docs])

# 2. 自定义提示词模板（1.x推荐用ChatPromptTemplate，通过from_messages创建）
# 保持原业务规则：基于参考资料、分点说明、带案例
system_prompt = """你是一个专业的RAG系统问答助手，必须基于以下提供的参考资料（context）回答用户问题。
规则：
1. 答案必须严格基于参考资料，不能编造未提及的信息；
2. 语言简洁明了，分点说明（如果有多个要点）；
3. 每个要点搭配1个简单案例，帮助理解。

参考资料：{context}"""

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # 系统指令
    ("human", "{question}")     # 用户问题（1.x推荐用"question"键，语义更清晰）
])

# 3. 用LCEL构建完整检索-生成链（管道符串联组件，数据流式传递）
rag_qa_chain = (
    # 第一步：并行处理输入（传递用户问题+检索文档）
    {"context": retriever | format_docs, "question": RunnablePassthrough()},
    # 第二步：将格式化数据传入提示词模板
    custom_prompt,
    # 第三步：传入大模型生成答案
    llm,
    # 第四步：解析输出（统一为字符串格式）
    StrOutputParser()
)

# 补充：如需返回检索的源文档（用于验证答案来源），可调整链结构：
rag_qa_chain_with_sources = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "source_documents": retriever  # 保留原始检索文档
    }
    | custom_prompt
    | llm
    | StrOutputParser()
)
```

**第三步：测试RAG系统并验证结果**

1.x版本统一用`invoke`方法执行链，参数为字典格式，获取结果后可直接打印答案和源文档

```python
# 测试问题列表（覆盖不同类型的查询）
test_questions = [
    "RAG系统的核心价值是什么？",
    "RAG和直接使用大模型相比，优势在哪里？",
    "企业内部知识库问答为什么适合用RAG？"
]

# 执行测试并打印结果
for i, question in enumerate(test_questions):
    print(f"\n===== 测试问题{i+1}：{question} =====")
    # 执行RAG链（1.x统一用invoke方法）
    result = rag_qa_chain_with_sources.invoke(question)  # 带源文档的链
    
    # 打印生成的答案
    print("\n生成答案：")
    print(result)
    
    # 打印参考资料（验证答案来源）
    print("\n参考资料：")
    # 注意：源文档从链的输入参数中获取（因链结构中保留了source_documents）
    sources = retriever.invoke(question)  # 重新调用检索器获取源文档（或在链中传递）
    for j, doc in enumerate(sources):
        print(f"\n参考片段{j+1}：")
        print(doc.page_content)
        if doc.metadata:  # 打印文档元数据（如文件名、页码等）
            print(f"元数据：{doc.metadata}")
```

**第四步：运行结果与系统验证**

#### 第五步：运行结果与系统验证

运行代码后，将得到清晰的答案、参考资料及对应关系。核心验证点：

- 答案严格基于检索的参考资料，无编造信息；
- 答案按要求分点说明，每个要点带案例；
- 源文档可正常打印，能追溯答案来源。

> 小技巧：RetrievalQA的chain_type除了“stuff”，还有“map_reduce”“refine”等。如果检索片段较多、较长，建议用“map_reduce”（先分别处理每个片段，再汇总答案），避免提示词超过大模型上下文窗口限制。

### 4.4.6 RAG系统常见问题与优化方向

搭建完基础RAG系统后，实际应用中可能会遇到检索不准、生成答案质量低等问题。下面总结常见问题及对应的优化方向，帮助大家提升系统性能。

#### 4.4.6.1 常见问题排查

| 常见问题             | 可能原因                                                     | 排查与解决方法                                               |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 检索结果不相关       | 1. 文本分割不合理（片段过大/过小，语义不完整）；2. 嵌入模型不适合中文场景；3. 检索参数设置不当（k值过大、MMR多样性过高） | 1. 调整分割参数（chunk_size=200-500字，用RecursiveCharacterTextSplitter）；2. 替换为中文优化的嵌入模型（如通义千问embedding-v2）；3. 减小k值（k=2-3），降低MMR多样性权重（lambda_mult=0.7-0.8） |
| 生成答案遗漏关键信息 | 1. 检索片段未覆盖关键信息；2. 提示词未明确要求“全面回答”；3. 大模型temperature过低，生成过于简洁 | 1. 增大k值（k=4-5），改用相似性检索保证核心信息覆盖；2. 优化提示词，添加“全面覆盖参考资料中的所有关键要点”；3. 适当提高temperature（0.4-0.5） |
| 生成答案包含编造信息 | 1. 提示词未严格限制“基于参考资料”；2. 检索片段相关性低，大模型无足够信息生成答案；3. 大模型temperature过高 | 1. 优化提示词，明确“禁止编造信息，无相关信息则回复‘无法回答’”；2. 提升检索精度（调整分割、嵌入模型、检索参数）；3. 降低temperature（0.2-0.3） |
| 系统响应速度慢       | 1. 检索片段过多（k值过大）；2. 向量数据库未优化（未持久化、片段数量过多）；3. 大模型推理速度慢 | 1. 减小k值（k=2-3）；2. 优化向量数据库（定期清理无效片段、使用更高效的向量数据库如FAISS）；3. 替换为轻量型大模型（如GPT-3.5-turbo-instruct替代GPT-4） |

#### 4.4.6.2 进阶优化方向

如果基础RAG系统满足不了业务需求，可以从以下3个方向进阶优化，提升系统性能：

1. **检索优化**：引入“混合检索”（相似性检索+关键词检索），结合语义相关性和关键词匹配，提升检索精度；使用“检索重排”（如Cross-Encoder），对初步检索结果重新排序，筛选出最相关的片段。
2. **提示词工程优化**：采用“思维链（CoT）”提示词，引导大模型逐步分析参考资料、生成答案；针对特定行业（如医疗、法律）定制领域专属提示词模板，提升答案专业性。
3. **知识库优化**：建立知识库更新机制（定期添加新文档、删除过期文档）；对文档进行预处理（清洗冗余信息、标注核心知识点），提升文本分割和嵌入的效果。

这些进阶优化方向在企业级RAG应用中非常重要，后续章节会结合具体业务场景深入讲解。

## 4.6 RAG 系统的评估与调优方法（选学）

> 以下部分内容，更加的专业化，工程化，适合在职或有工作经验的同学学习。

搭建完RAG系统并不意味着结束，实际应用中需要通过科学的评估发现问题，再针对性调优，才能让系统稳定输出高质量结果。

本节将从核心评估指标、评估方法、调优方向和工程化实践四个维度，完整覆盖RAG系统“评估-调优”的闭环流程。

### 4.6.1 评估核心指标：相关性、准确性、响应速度

评估RAG系统的效果，核心围绕“能否精准找到有用信息”“答案是否正确可用”“用户等待时间是否可接受”三个核心诉求，对应三个关键指标：相关性、准确性、响应速度。每个指标都有明确的定义和可量化的衡量标准。

| 指标名称                      | 核心定义                                                   | 衡量标准与计算方式                                           | 优化目标                                             |
| ----------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| 相关性（Retrieval Relevance） | 检索器返回的文本片段与用户问题的语义关联程度               | 1. 精确率（Precision）：检索到的相关片段数 / 检索到的总片段数； 2. 召回率（Recall）：检索到的相关片段数 / 知识库中所有相关片段数； 3. F1分数：2×(精确率×召回率)/(精确率+召回率)（综合精确率和召回率） | F1分数≥0.8，确保检索结果“既准又全”                   |
| 准确性（Generation Accuracy） | 大模型生成的答案与参考资料（检索片段）的一致性、事实正确性 | 1. 事实一致性（Factuality）：人工或自动化工具判断答案是否符合检索片段中的事实； 2. 信息完整性（Completeness）：答案是否覆盖用户问题所需的所有核心信息； 3. 错误率：包含编造信息、错误关联的答案占比 | 事实一致性≥95%，错误率≤5%，核心信息覆盖率≥90%        |
| 响应速度（Response Speed）    | 用户发起提问到获取最终答案的总耗时，含检索耗时和生成耗时   | 1. 检索耗时：从接收问题到获取相关片段的时间； 2. 生成耗时：从传入检索片段到生成最终答案的时间； 3. 总耗时=检索耗时+生成耗时 | 总耗时≤2秒（普通问答场景），≤5秒（复杂文档问答场景） |

> 小提示：三个指标存在一定权衡关系。比如增大检索片段数量（k值）可能提升召回率，但会增加响应速度；提高大模型temperature可能提升答案丰富度，但可能降低准确性。实际优化需结合业务场景优先级调整。

### 4.6.2 评估方法：人工评估与自动化评估工具

根据评估成本和效率，RAG系统的评估方法分为两类：人工评估（精准但低效，适合小规模验证）和自动化评估（高效但需校准，适合大规模迭代）。

实际项目中通常结合使用，先通过自动化工具快速筛选问题，再用人工评估精准验证核心场景。

#### 4.6.2.1 人工评估：精准验证核心场景

人工评估适合对核心业务场景（如高价值客户咨询、关键政策问答）的效果验证，需要制定清晰的评估标准和流程，避免主观偏差。

**1. 评估流程**

1. 构建测试集：选取100-200个覆盖核心场景的用户问题（含简单问题、复杂问题、边缘问题），并标注每个问题在知识库中对应的正确参考片段；
2. 系统输出采集：将测试集问题输入RAG系统，记录每个问题的检索片段、生成答案、响应耗时；
3. 人工打分：由2-3名评估者按照预设标准对每个指标打分，打分结果取平均值（减少主观偏差）；
4. 结果分析：统计各指标得分，定位薄弱环节（如某类问题检索相关性低、某类答案准确性差）。

**2. 打分标准示例（以准确性为例）**

| 分数等级 | 标准描述                                                     |
| -------- | ------------------------------------------------------------ |
| 5分      | 答案完全符合检索片段事实，核心信息完整，逻辑清晰，无任何编造内容 |
| 3分      | 答案基本符合事实，但遗漏1-2个非关键信息，或表述存在轻微歧义  |
| 1分      | 答案存在事实错误，或编造检索片段中未提及的信息，无法满足用户需求 |

#### 4.6.2.2 自动化评估工具：高效迭代优化

对于大规模迭代（如每日优化后的效果验证），人工评估成本过高，需借助自动化工具快速评估。常用的自动化评估工具分为两类：LangChain原生工具和第三方专用工具（如RAGAS、LLM-as-a-Judge）。

**1. LangChain原生评估工具**

LangChain 1.x提供了检索评估专用的`RetrievalEvaluator`（迁移至`langchain.evaluation.retrieval`子模块）和答案评估专用的`QAEvalChain`，可直接评估检索相关性和答案准确性，无需额外开发。

```python
# 1. 评估检索相关性（精确率、召回率、F1分数）
from langchain.evaluation.retrieval import RetrievalEvaluator  # 1.x 正确导入路径
from langchain_community.vectorstores import FAISS  # 替换：导入FAISS（原Chroma）
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未找到OPENAI_API_KEY环境变量")

# 初始化Qwen3本地嵌入模型
embedding_model_path = "./models/Qwen/Qwen3-Embedding-0___6B"
# 验证模型路径有效性
if not os.path.exists(embedding_model_path):
    raise FileNotFoundError(f"Qwen3嵌入模型路径不存在：{embedding_model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={
        "device": "cpu",  # 强制CPU运行，无需GPU
        "trust_remote_code": True,  # Qwen3模型必选配置，信任远程代码
        # 如需加载量化模型，可启用以下配置（降低内存占用）
        # "load_in_8bit": True,
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索相似度计算精度
    }
)

# 初始化向量数据库和检索器（替换：Chroma→FAISS，适配FAISS本地加载方式）
faiss_db_path = "./faiss_db"  # FAISS数据库存储路径
if not os.path.exists(faiss_db_path):
    raise FileNotFoundError(f"FAISS向量数据库路径不存在：{faiss_db_path}")

vector_db = FAISS.load_local(
    folder_path=faiss_db_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # 本地开发允许加载，生产环境需谨慎
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 辅助函数：FAISS通过document_id获取标注文档（FAISS无直接get方法，需自定义）
def get_doc_by_id(vector_db, doc_id):
    for doc in vector_db.docstore._dict.values():
        if doc.metadata.get("document_id") == doc_id:
            return doc
    raise ValueError(f"未找到document_id为{doc_id}的文档")

# 定义测试集（关键修正：relevant_documents需为人工标注的正确文档，而非动态检索）
# 注意：需提前为FAISS中的文档添加document_id元数据，确保评估真实性
test_cases = [
    {
        "query": "RAG系统的核心价值是什么？",
        # 示例：人工从FAISS向量库中筛选的2篇正确相关文档（真实场景需手动标注）
        "relevant_documents": [
            get_doc_by_id(vector_db, "doc1"),  # 通过自定义函数获取标注文档
            get_doc_by_id(vector_db, "doc2")
        ]
    },
    {
        "query": "LangChain的SequentialChain有什么用？",
        "relevant_documents": [
            get_doc_by_id(vector_db, "doc3")
        ]
    }
]

# 初始化检索评估器（1.x 用法不变，指标支持完整）
retrieval_evaluator = RetrievalEvaluator(
    metric_names=["precision", "recall", "f1"],  # 要评估的指标
    retriever=retriever
)

# 执行评估
results = retrieval_evaluator.evaluate_batch(
    [case["query"] for case in test_cases],
    [case["relevant_documents"] for case in test_cases]
)

# 打印结果
for i, result in enumerate(results):
    print(f"\n测试用例{i+1}（问题：{test_cases[i]['query']}）")
    print(f"精确率：{round(result['precision'], 4)}")
    print(f"召回率：{round(result['recall'], 4)}")
    print(f"F1分数：{round(result['f1'], 4)}")

# 2. 评估答案准确性（事实一致性、完整性）
from langchain.evaluation import QAEvalChain
from langchain_openai import ChatOpenAI  # 1.x 推荐用ChatOpenAI（替代旧版OpenAI）

# 初始化大模型（1.x 推荐配置：添加超时、重试，提升稳定性）
llm = ChatOpenAI(
    api_key=api_key,
    temperature=0,  # 0温度保证评估标准统一
    timeout=30,
    max_retries=2
)
eval_chain = QAEvalChain.from_llm(llm=llm)

# 定义测试集（问题-正确答案-系统答案）
test_data = [
    {
        "question": "RAG系统的核心价值是什么？",
        "answer": "RAG系统的核心价值包括提升准确性、拓展知识边界、降低微调成本。提升准确性指答案基于真实参考资料，减少事实错误；拓展知识边界指大模型可获取训练数据外的最新信息和私有信息；降低成本指更新知识无需微调大模型，仅需更新知识库。",
        "prediction": "RAG的核心价值是提升答案准确性、拓展知识范围、降低企业使用大模型的成本。准确性方面，依托外部知识库保障事实正确；知识范围可覆盖最新行业数据和内部文档；成本上避免了昂贵的大模型微调。"
    },
    {
        "question": "SequentialChain的作用是什么？",
        "answer": "SequentialChain用于串联多个LLMChain，支持多输入多输出，可指定整个流程的输入变量和输出变量，能结合多个维度的信息完成复杂的多步骤任务。",
        "prediction": "SequentialChain可以把多个链按顺序连接起来，支持多个输入和输出，适合处理需要多步骤的任务，比如先提取产品卖点再写营销话术。"
    }
]

# 执行评估
eval_results = eval_chain.evaluate(
    test_data,
    prediction_key="prediction",  # 系统答案的键名
    question_key="question",      # 问题的键名
    answer_key="answer"           # 正确答案的键名
)

# 打印评估结果
for i, result in enumerate(eval_results):
    print(f"\n测试用例{i+1}评估结果：")
    print(f"事实一致性评分（1-5分）：{result['correctness']}")
    print(f"评估说明：{result['reasoning']}")
```

**2. 第三方专用工具：RAGAS（RAG专用评估框架）**

RAGAS是专门针对RAG系统的评估框架，支持评估相关性、准确性、完整性等核心指标，且无需标注正确答案（仅需问题、检索片段、系统答案），使用更便捷

```python
# 安装依赖：pip install ragas datasets faiss-cpu  # 补充：安装FAISS依赖
from ragas import evaluate
from ragas.metrics import (
    RetrievalPrecision,  # 检索精确率
    RetrievalRecall,     # 检索召回率
    Faithfulness,        # 事实一致性
    AnswerCompleteness   # 答案完整性
)
from datasets import Dataset
from langchain_community.vectorstores import FAISS  # 替换：导入FAISS（原Chroma）
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未找到OPENAI_API_KEY环境变量")

# 1. 初始化RAG系统组件（1.x 官方推荐LCEL链式写法，替代旧版RetrievalQA）
# 初始化Qwen3本地嵌入模型
embedding_model_path = "./models/Qwen/Qwen3-Embedding-0___6B"
# 验证模型路径有效性
if not os.path.exists(embedding_model_path):
    raise FileNotFoundError(f"Qwen3嵌入模型路径不存在：{embedding_model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={
        "device": "cpu",  # 强制CPU运行，无需GPU
        "trust_remote_code": True,  # Qwen3模型必选配置，信任远程代码
        # 如需加载量化模型，可启用以下配置（降低内存占用）
        # "load_in_8bit": True,
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索相似度计算精度
    }
)

# 初始化FAISS向量数据库（替换：Chroma→FAISS）
faiss_db_path = "./faiss_db"
if not os.path.exists(faiss_db_path):
    raise FileNotFoundError(f"FAISS向量数据库路径不存在：{faiss_db_path}")

vector_db = FAISS.load_local(
    folder_path=faiss_db_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # 本地开发允许加载，生产环境需谨慎
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 定义提示词模板
system_prompt = "基于以下上下文回答问题：{context}"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 定义文档格式化函数
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 构建LCEL风格的RAG链
rag_qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()},
    prompt,
    ChatOpenAI(api_key=api_key, temperature=0.3),
    StrOutputParser()
)

# 2. 构建测试数据集（仅需问题）
test_questions = [
    "RAG系统的核心价值是什么？",
    "SequentialChain的作用是什么？",
    "RAG系统的构建流程有哪些步骤？"
]

# 3. 采集系统输出（检索片段、生成答案）
test_data = []
for question in test_questions:
    # 获取生成答案（1.x 统一用invoke方法执行链）
    answer = rag_qa_chain.invoke(question)
    # 获取检索片段（1.x 用invoke替代get_relevant_documents，FAISS完全兼容）
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    test_data.append({
        "question": question,
        "answer": answer,
        "contexts": contexts
    })

# 4. 转换为RAGAS支持的Dataset格式
dataset = Dataset.from_list(test_data)

# 5. 定义要评估的指标
metrics = [RetrievalPrecision(), RetrievalRecall(), Faithfulness(), AnswerCompleteness()]

# 6. 执行评估（关键修正：最新版RAGAS通过llm参数指定大模型，无需直接传api_key）
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ChatOpenAI(api_key=api_key, temperature=0)  # 传入LangChain LLM实例
)

# 7. 打印评估结果（转换为DataFrame更易读）
print("RAG系统自动化评估结果：")
print(results.to_pandas())
```

运行结果示例：

```text
RAG系统自动化评估结果：
   question  retrieval_precision  retrieval_recall  faithfulness  answer_completeness
0  RAG系统的核心价值是什么？               0.85              0.82         0.96                  0.88
1  SequentialChain的作用是什么？               0.90              0.88         0.98                  0.92
2  RAG系统的构建流程有哪些步骤？               0.82              0.79         0.95                  0.85
```

>  踩坑指南：自动化评估工具的结果需人工校准。比如RAGAS的事实一致性评分可能存在偏差，对于核心场景的评估结果，建议抽取10%-20%的样本进行人工复核，确保评估准确性。

### 4.6.3 调优方向：分割策略、向量模型、检索参数优化

根据评估结果定位问题后，需针对性调优。RAG系统的调优核心围绕“检索环节”（检索是生成的基础，检索不准则生成必错），主要包括三个方向：文本分割策略、向量模型选择、检索参数配置。

#### 4.6.3.1 文本分割策略优化

文本分割是RAG的“基础工程”，分割不合理会直接导致检索相关性低。优化需结合文档类型和问题场景调整，核心是保证片段的“语义完整性”和“大小适中”。

| 常见问题                                  | 优化方案                          | 实操建议                                                     |
| ----------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| 片段过大（超过500字），检索时无关信息过多 | 减小chunk_size，优化分割符优先级  | 中文文档：chunk_size=200-300字，分割符顺序：\n\n→\n→。→，；英文文档：chunk_size=200-300token，分割符顺序：\n\n→\n→.→,→ |
| 片段过小（小于100字），丢失上下文信息     | 增大chunk_size，增加chunk_overlap | chunk_size调整为300-400字，chunk_overlap=50-80字，避免跨片段的知识点被切断 |
| 结构化文档（如MD、PDF）分割后破坏层级关系 | 使用针对性分割器，保留层级信息    | MD文档用MarkdownTextSplitter，PDF文档用PyPDFLoader按页分割后再二次细分（按段落） |

#### 4.6.3.2 向量模型选择与优化

向量模型的质量直接决定文本嵌入的语义表征能力，进而影响检索相关性。优化需结合语言类型（中文/英文）、业务场景（通用/专业领域）选择合适的模型，必要时进行模型微调。

| 模型类型     | 代表模型                                                     | 优势                                                         | 适用场景                                                     |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 通用英文模型 | EmbeddingGemma（谷歌，开源）、Yuan-EB 2.0-en（浪潮信息，开源） | 语义表征能力强，多语言支持（覆盖100+种语言），轻量化易部署，部分支持自定义向量维度，兼顾速度与精度 | 英文通用文档问答（如国际新闻、英文技术文档、跨语言信息检索） |
| 通用中文模型 | Qwen3-Embedding（通义千问，开源）、Yuan-EB 2.0-zh（浪潮信息，开源） | 针对中文语义深度优化，检索与排序任务双SOTA，轻量化参数（0.3B-0.6B），支持本地部署，部分具备多模态检索能力 | 中文通用场景（如企业内部中文手册、中文客服问答、中文语义搜索） |
| 专业领域模型 | 医疗领域：Claude Opus 4.5（Anthropic）；法律领域：法衡-R2（东南大学） | 医疗模型支持专业病历处理与医疗数据库接入，合规性强；法律模型具备复杂司法推理与裁判文书深度理解能力，术语表征精准 | 医疗领域（病历解读、临床试验分析、理赔申诉）；法律领域（法条检索、案件争议焦点总结、裁判文书分析）；金融等其他专业领域可基于通用SOTA模型微调 |

> 实操技巧：如果通用模型效果不佳，可尝试“模型对比测试”——用多个模型对同一批测试数据进行嵌入和检索，通过F1分数对比选择最优模型。

#### 4.6.3.3 检索参数配置优化

检索参数的调整直接影响检索结果的“相关性”和“多样性”，需结合评估指标（精确率、召回率、F1分数）动态调整，核心参数包括k值、检索类型、MMR相关参数。

| 参数类型 | 核心参数                                   | 调整逻辑                                                   | 实操建议                                                   |
| -------- | ------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- |
| 基础参数 | k（返回片段数量）                          | 精确率低→减小k值；召回率低→增大k值                         | 通用场景k=3-4；复杂问题k=4-5；简单问题k=2-3                |
| 检索类型 | similarity（相似性）/mmr（最大边际相关性） | 需精准核心信息→similarity；需全面多样化信息→mmr            | 简单问答用similarity；复杂对比、多维度分析用mmr            |
| MMR参数  | lambda_mult（相关性-多样性权重）           | 需优先相关性→增大lambda_mult；需优先多样性→减小lambda_mult | 通用场景lambda_mult=0.7-0.8；多样化需求lambda_mult=0.4-0.5 |

### 4.6.4 工程化调优实践总结

在企业级RAG项目中，调优不是“一次性操作”，而是“评估-调优-验证”的循环过程。结合实际项目经验，总结出以下工程化调优流程，确保调优效果稳定、可复现。

1. **问题定位阶段**：通过自动化评估工具快速扫描全量测试数据，统计各指标的薄弱环节（如“医疗术语相关问题检索相关性低”“长文档问答准确性差”）；对薄弱环节的样本进行人工分析，明确根因（如“分割片段过小导致医疗术语上下文丢失”“向量模型对专业术语表征不足”）。
2. **针对性调优阶段**：根据根因选择调优方向，制定调优方案并执行。例如：根因为“专业术语表征不足”，调优方案为“替换为医疗领域专用嵌入模型”；根因为“片段过小”，调优方案为“调整chunk_size从200字到350字，chunk_overlap从50字到80字”。
3. **效果验证阶段**：将调优后的系统重新运行测试集，通过自动化工具评估指标变化；抽取核心场景样本进行人工复核，确保调优后无新问题（如“调整k值后响应速度是否达标”“替换模型后其他场景相关性是否下降”）。
4. **迭代优化阶段**：将调优方案固化到系统配置中，记录调优前后的指标变化（形成调优日志）；定期（如每月）更新测试集（加入新场景问题），重新评估并调优，确保系统适配业务需求的变化。

工程化技巧：建立“调优配置中心”，将分割参数、向量模型、检索参数等配置项集中管理，支持动态切换配置并对比效果，避免硬编码导致的调优效率低下。

## 4.7 本章小结

本章围绕LangChain的核心应用——链式工作流和RAG系统，从原理、实操、评估调优三个维度展开，覆盖了从“简单链”到“企业级RAG系统”的全链路知识。以下是核心知识点和技术落地要点的总结，帮助大家梳理脉络、巩固重点。

**1. 链式工作流：让大模型按步骤解决复杂问题**

- 核心价值：将复杂任务拆解为多个简单子任务，通过串联或动态分发的方式，让大模型逐步完成，提升任务处理的精准度和可控性。
- 三种核心链类型：  - SimpleSequentialChain：单输入单输出，线性串联多个链，适合简单的多步骤任务（如“文本摘要→关键词提取”）；  - SequentialChain：多输入多输出，支持指定输入输出变量，适合需要结合多个维度信息的任务（如“提取产品卖点→针对特定人群写营销话术”）；  - RouterChain：动态任务分发，通过路由选择器判断用户需求，分发到对应的目标链，适合多场景客服、多任务处理系统。
- 异常处理：通过重试机制（RetryWithErrorOutputChain）、异常捕获（try-except）、分支降级（核心链失败切换备用链）提升系统稳健性。

**2. RAG 系统：解决大模型知识滞后与事实性错误**

- 核心逻辑：检索增强生成，通过“文档加载→文本分割→向量存储→检索→生成”的流程，让大模型基于外部知识库的真实信息生成答案，避免胡编乱造。
- 全流程关键环节：  - 文档加载：适配PDF、Word、MD、TXT等多格式文档，转换为LangChain统一的Document对象；  - 文本分割：用RecursiveCharacterTextSplitter等工具，保证片段语义完整、大小适中；  - 向量存储：通过嵌入模型将文本片段转为向量，存储到Chroma等向量数据库，实现快速相似性检索；  - 检索优化：结合相似性检索和MMR检索，平衡相关性和多样性；  - 检索-生成整合：用RetrievalQA链整合检索器和大模型，支持自定义提示词，实现端到端问答。

## 4.8 本章练习

通过分层练习巩固本章核心知识，从基础的链式工作流实现，到进阶的RAG系统构建与调优，再到拓展的工程化能力提升，逐步提升大模型应用开发实战能力。

### 1.基础练习：基于 SequentialChain 实现多步骤文本处理任务

**练习任务**

实现“新闻文本分类→提取核心事件→生成摘要”的多步骤任务：

1. 输入：新闻文本（news_text）、分类标签列表（category_list，如[“科技”, “财经”, “娱乐”, “体育”]）；
2. 步骤1（链1）：根据分类标签列表，对新闻文本进行分类，输出分类结果（category）；
3. 步骤2（链2）：根据分类结果和新闻文本，提取该类新闻的核心事件（如科技新闻提取“技术突破、产品发布”等，财经新闻提取“政策变化、企业动态”等），输出核心事件（core_event）；
4. 步骤3（链3）：结合分类结果和核心事件，生成100字以内的新闻摘要（summary）；
5. 输出：分类结果、核心事件、新闻摘要。

### 2. 进阶练习：构建支持 PDF 文档的 RAG 问答机器人并完成调优

**练习任务**

1. 文档加载：加载1-2份PDF文档（如企业产品手册、行业报告），转换为Document对象；
2. 文本分割：使用RecursiveCharacterTextSplitter对文档进行分割，调试chunk_size和chunk_overlap参数；
3. 向量存储：选择合适的嵌入模型（如通义千问text-embedding-v2），将分割后的片段存入Chroma向量数据库；
4. 检索-生成整合：构建RetrievalQA链，自定义提示词（要求答案基于检索片段，分点清晰）；
5. 评估与调优：  - 构建10个测试问题（覆盖PDF文档的核心知识点）；  - 用RAGAS工具评估检索精确率、事实一致性等指标；  - 根据评估结果调优（如调整分割参数、检索k值、嵌入模型），使核心指标达标（F1≥0.8，事实一致性≥0.9）。

### 3. 拓展练习：为 RAG 系统添加错误处理与分支降级机制

**练习任务**

提升RAG系统的工程化能力，通过错误处理和分支降级机制，确保系统在异常情况下仍能正常响应。

### 练习任务

1. 添加重试机制：为RAG系统的检索环节和生成环节添加重试逻辑，当API调用超时或临时失败时，自动重试2-3次；
2. 添加异常捕获：捕获API密钥错误、输入格式错误、检索无结果等常见异常，返回友好的错误提示；
3. 添加分支降级：  - 核心链：使用GPT-4作为大模型，通义千问text-embedding-v2作为嵌入模型；  - 降级链：当核心链失败（如API调用失败、指标不达标）时，自动切换为GPT-3.5-turbo-instruct（大模型）和m3e-base（开源嵌入模型）；
4. 测试验证：模拟多种异常场景（如错误API密钥、网络中断、检索无结果），验证系统是否能正常降级并返回有效响应。
