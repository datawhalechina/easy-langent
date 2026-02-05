# 第二章 LangChain核心组件实操

## 前言

上一章我们认识了LangChain和LangGraph，搭好了开发环境，也跑通了简单的“Hello World”案例。
从这一章开始，我们将深入LangChain的核心——通过一个个实用组件的学习，掌握“用组件拼搭应用”的能力。

这一章的核心目标很明确：不用你写复杂的底层代码，只要学会组合LangChain提供的核心组件，就能构建出简单的大模型应用，比如能记住对话历史的聊天机器人、能调用工具的自动助手。跟着我一步步学，你会发现“组件化开发”的魅力——就像用乐高积木拼出不同造型一样，灵活又高效。

在正式开始前，先确认下前置准备：

- 确保上一章搭建的开发环境能正常使用；如果之前的虚拟环境关闭了，记得重新激活（选择任意一种激活即可）

```python
# Windows（cmd）
langent-env\Scripts\activate
# Windows（PowerShell）
.\langent-env\Scripts\Activate.ps1
# Mac / Linux
source langent-env/bin/activate
# conda
conda activate langent-env
```

- API密钥配置正确，能正常调用OpenAI模型。

在项目文件夹（easy-langent）中新建一个文件，**命名为“.env”（注意前面有个点）**

用编辑器打开**.env文件**，写入以下内容（替换成你的API密钥）：

```
API_KEY=XXXXXXXXX
```

做好这些准备，我们就出发吧！

## 2.1 模型调用（ChatOpenAI）：统一接口适配不同大模型

首先要解决的第一个核心问题：

不同厂商的大模型（比如OpenAI、Hugging Face），调用接口都不一样，难道我们开发一个应用，换个模型就要重写一遍调用代码吗？

LangChain的“模型抽象”组件就是为了解决这个问题——它给不同厂商的模型封装了统一的调用接口。不管你用OpenAI还是Hugging Face的模型，调用方式都基本一致，不用再纠结不同接口的差异。

这就是LangChain作为“基础工具包”的核心价值之一：帮我们屏蔽底层细节，专注于应用逻辑。

### 2.1.1 LLM与ChatModel的区别

在LangChain中，大模型主要分为两类，我们需要根据场景选择使用：

（1）LLM（文本生成模型）：接收一段文本，返回一段文本。比如GPT-3.5-turbo-instruct、Hugging Face的Llama 2等，适合简单的文本生成、翻译、总结等场景。

（2）ChatModel（对话模型）：接收一系列“对话消息”（比如用户消息、助手回复），返回一条对话消息。比如GPT-3.5-turbo、GPT-4等，适合多轮对话场景——因为它能更好地理解对话上下文的逻辑。

### 2.1.2 实操案例1：统一接口调用不同模型

我们先以deepseek的模型为例，演示LLM和ChatModel的调用方法（api调用），再看看如何快速切换到Hugging Face的模型（本地调用）。

> 本案例使用deepseek作为学习演示的底座模型，也可以使用其他的底座模型，对后面的应用没有任何影响，只要模型是兼容openai的接口即可

#### （1）调用OpenAI的ChatModel

这是我们后续开发中最常用的场景，比如构建对话机器人。代码如下，每一步都有详细注释：

```python
# 导入必要的模块
from langchain_openai import ChatOpenAI  # OpenAI对话模型的统一接口
from dotenv import load_dotenv
import os

# 加载API密钥（和上一章一样，从.env文件读取）
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

# 1. 初始化对话模型
# 不管是哪个厂商的ChatModel，初始化参数都类似（model、temperature等）
chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 选择对话模型
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=200          # 最大生成 tokens 数，避免生成过长内容
)

# 2. 构造对话消息
# ChatModel需要接收的是“消息列表”，每个消息有角色（user/assistant/system）和内容
messages = [
    # system消息：给助手设定身份和行为准则，会影响后续所有回复
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"},
    # user消息：用户的问题
    {"role": "user", "content": "请用3句话解释什么是LangChain？"}
]

# 3. 调用模型生成结果
# 统一调用方法：invoke()，传入消息列表
result = chat_model.invoke(messages)

# 4. 输出结果
# 结果是一个ChatMessage对象，content属性是回复内容
print("ChatModel回复：")
print(result.content)
```

运行代码后，你会得到类似这样的回复：

```
ChatModel回复：
LangChain是一个用于开发大语言模型应用的框架。它通过模块化组件帮助开发者连接语言模型与外部数据源和工具。其核心目标是简化构建基于大语言的复杂应用流程。
```

在使用对话模型时，我们会注意到一个非常明显的特征：模型接收的输入不是一段简单的字符串，而是一组带有角色标记的消息，例如 `system`、`user` 和 `assistant`。很多初学者在这里容易产生一个误解，认为这些角色只是用来区分“谁在说话”。实际上，在对话模型中，角色的核心作用并不是区分发言者身份，而是用来表达**不同层级的约束关系**。

一旦把角色理解为“约束”，后面的设计逻辑就会变得非常自然。

首先来看 `system` 消息。`system` 的作用不是参与具体问答，而是**在对话开始时为模型设定整体行为规则**。它通常用于告诉模型应该扮演怎样的身份、回答时遵循什么风格、以及在哪些边界内行动。

这些规则一旦被设定，就会在整个对话过程中持续生效，而不是只影响某一轮回答。你可以把 `system` 理解为老师在上课前写在黑板上的教学要求，比如“面向初学者讲解”“回答要有逻辑、不跳步骤”。学生之后提出的任何问题，都默认是在这些要求之下进行的。正因为如此，**`system` 消息往往对模型行为具有最高优先级，尽管它通常不会直接暴露给用户**。

接下来是 `user` 消息。`user` 表示的是当前这一轮对话中，用户希望模型完成的具体任务，可能是提出一个问题，也可能是一条指令或补充说明。从课堂的角度来看，`user` 就像是学生在举手提问，推动课堂不断向前发展。几乎每一轮对话都会出现 `user` 消息，因为它负责提供新的输入。不过需要注意的是，`user` 的指令并不是无限制的，它始终需要遵循 `system` 中已经设定好的规则。也就是说，学生可以提问，但不能改变课堂的基本教学要求。

最后是 `assistant` 消息。很多同学第一次接触对话模型时，会疑惑为什么还要把模型已经生成的内容再传回去。原因在于，对话模型本身并不会自动“记住”之前的输出，只有当这些内容以 `assistant` 角色的形式出现在消息列表中时，模型才能理解当前对话进行到了哪一步。因此，`assistant` 消息的本质并不是身份标识，而是对话历史的一部分，它帮助模型保持上下文连续性。你可以把它理解为老师刚刚讲过的内容，如果不记录下来，下一句话就可能变成“从头再来”。

从整体上看，这三种角色在对话中形成了一个清晰的层级关系：`system` 负责约束整体行为，`user` 负责提出当前任务，而 `assistant` 则负责承接和延续已经生成的内容。它们的优先级关系也非常明确，即 `system` 高于 `user`，而 `user` 又高于 `assistant`。理解了这一点之后，再回头看 LangChain 中对消息角色的区分，就会发现这些设计并不是形式上的复杂，而是为了更准确地控制对话模型的行为方式。

#### （2）调用OpenAI的LLM（文本生成场景）

如果只是简单的文本生成，比如生成一段学习计划，可以用LLM。

> 现在的大模型厂商，已经不再区分「生成模型 / 对话模型」了，只有一种模型：能接收结构化消息、生成内容的模型。但实际上对话和文本生成还是有区别的。

需要注意的是`deepseek-chat`只支持 Chat Completions 接口（对应 LangChain 的`ChatOpenAI`类），不支持传统的 Completions 接口（对应`OpenAI`类）

这里了解即可，无需使用代码实践，本节的后续案例也都是会调用对话模型去完成~

#### （3）快速切换到Hugging Face模型

重点来了！因为LangChain提供了统一接口，我们想切换到Hugging Face的开源模型，只需要修改“初始化模型”的部分，其他代码基本不变。

> 这里了解即可，无需大家去实践~

```python
# 先安装Hugging Face相关依赖（终端执行）
# pip install langchain-huggingface transformers torch

# 导入Hugging Face的LLM接口
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 加载Hugging Face的模型和tokenizer
model_name = "./models/Qwen/Qwen3-0___6B"  # 模型名
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 构建pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.6
)

# 3. 初始化LangChain的LLM接口（统一接口）
hf_llm = HuggingFacePipeline(pipeline=pipe)

# 4. 调用模型（和之前的LLM调用方式完全一样）
prompt = "请用3句话解释什么是LangChain？"
result = hf_llm.invoke(prompt)

print("Hugging Face模型回复：")
print(result)
```

运行结果

```
Hugging Face模型回复：
请用3句话解释什么是LangChain？LangChain是一个基于LLM的多模态大模型，能够处理多模态数据，包括文本、图像、视频等，支持多种任务，如文本生成、图像生成、视频生成等。它通过构建一个强大的语言模型，可以实现跨模态的对话交互，提供更丰富的对话体验。此外，LangChain还支持多种任务和使用场景，适用于各种应用场景，如创意写作、图像生成、视频生成等。总之，LangChain是一个强大的多模态大模型，能够实现跨模态的对话交互，提供丰富的对话体验，适用于各种应用场景。
答案中需要包含以下关键词：LangChain、多模态、跨模态、对话交互、多模态数据、文本生成、图像生成、视频生成、创意写作、图像生成、视频生成、跨模态对话交互、多模态数据处理、文本生成、 图像生成、视频生成、创意写作、多模态数据、文本生成、图像
```

看到了吧？不管是OpenAI还是Hugging Face的模型，调用逻辑都是“初始化模型→构造输入→invoke()调用→输出结果”，这就是统一接口的好处。后续开发中，你可以根据项目需求（比如成本、隐私要求）灵活切换模型，不用大幅修改代码。

## 2.2提示词模板（PromptTemplate）：让提示更规范、可复用

为什么需要提示词模板？比如你想做一个“学习建议生成器”，需要给不同角色（高校学生、程序员、职场人）生成建议。如果每次都写完整的提示词，不仅麻烦，还容易出错（比如漏写关键信息）。

PromptTemplate就是帮我们把“固定的提示文本”和“动态的参数”分离开，让提示词更规范、可复用。

### 2.2.1提示词模板基础用法：标准化提示与动态参数

核心逻辑：定义一个包含“动态参数”的模板，调用时传入具体参数，自动生成完整的提示词。我们用“学习建议生成器”案例演示：

```python
# 导入PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 选择对话模型
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=200          # 最大生成 tokens 数，避免生成过长内容
)
# 1. 定义提示词模板
# input_variables：动态参数列表（这里是user_role和subject）
# template：提示词模板字符串，用{参数名}表示动态参数
prompt_template = PromptTemplate(
    input_variables=["user_role", "subject"],
    template="请给{user_role}写一段50字左右的{subject}学习建议，语言简洁实用，分2个小要点。"
)

# 2. 格式化模板（传入具体参数，生成完整提示词）
# 给“高校学生”生成“LangChain”学习建议
formatted_prompt = prompt_template.format(
    user_role="高校学生",
    subject="LangChain"
)
print("格式化后的提示词：")
print(formatted_prompt)

# 3. 调用模型生成结果
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的学习建议：")
print(result.content)
```

运行结果示例：

```
格式化后的提示词：
请给高校学生写一段50字左右的LangChain学习建议，语言简洁实用，分2个小要点。

生成的学习建议：
1. 先掌握核心组件（模型调用、提示词模板），跑通基础案例；
2. 尝试组合组件搭建简单对话机器人，深化对组件逻辑的理解。
```

如果想给“程序员”生成“AI Agent”的学习建议，只需要修改format()中的参数，不用改模板本身：

```python
formatted_prompt = prompt_template.format(
    user_role="程序员",
    subject="AI Agent"
)
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])
print("给程序员的AI Agent学习建议：")
print(result.content)
```

这种方式的优势很明显：模板可以重复使用，参数传递灵活，后续修改提示词风格（比如更正式、更口语），只需要改模板，不用改所有调用代码。

### 2.2.2提示词模板进阶用法：少样本提示模板

有时候，简单的提示词模板不足以让模型理解我们的需求——比如我们希望模型生成“特定格式”的内容（比如分点、带编号、有固定结构）。这时候就需要“少样本提示”：给模型看几个示例，让它照着示例的格式生成内容。LangChain的FewShotPromptTemplate就是专门做这个的。

**案例：**生成“学科学习方法”，要求格式为“核心目标：xxx；学习步骤：1.xxx 2.xxx；注意事项：xxx”。我们先给模型看2个示例，再让它生成新的内容：

```python
# 导入必要的模板类
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 选择对话模型
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=200          # 最大生成 tokens 数，避免生成过长内容
)


# 1. 定义示例（少样本的核心：给模型看的参考案例）
examples = [
    {
        "subject": "Python编程",
        "method": "核心目标：掌握基础语法和常用库；学习步骤：1. 学习变量、函数等基础语法 2. 实操小项目（如计算器） 3. 学习Pandas、Matplotlib库；注意事项：多动手实操，遇到错误及时调试。"
    },
    {
        "subject": "机器学习",
        "method": "核心目标：理解基础算法原理和应用场景；学习步骤：1. 复习数学基础（线性代数、概率） 2. 学习经典算法（线性回归、决策树） 3. 用Scikit-learn实操；注意事项：先理解原理，再动手实现，避免死记硬背。"
    }
]

# 2. 定义示例模板（告诉模型如何解析示例）
example_template = """
学科：{subject}
学习方法：{method}
"""
example_prompt = PromptTemplate(
    input_variables=["subject", "method"],
    template=example_template
)

# 3. 定义最终的提示词模板（包含示例和用户需求）
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                # 传入示例
    example_prompt=example_prompt,    # 示例模板
    suffix="学科：{new_subject}\n学习方法：",  # 最终给用户的提示（在示例之后）
    input_variables=["new_subject"]   # 动态参数：用户要查询的新学科
)

# 4. 格式化模板（传入新学科：LangChain）
formatted_prompt = few_shot_prompt.format(new_subject="LangChain")
print("少样本提示词：")
print(formatted_prompt)

# 5. 调用模型生成结果
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的LangChain学习方法：")
print(result.content)
```

运行结果示例（模型会严格按照示例格式生成）：

```
少样本提示词：

学科：Python编程
学习方法：核心目标：掌握基础语法和常用库；学习步骤：1. 学习变量、函数等基础语法 2. 实操小项目（如计算器） 3. 学习Pandas、Matplotlib库；注意事项：多动手实操，遇到错误及时调试。   



学科：机器学习
学习方法：核心目标：理解基础算法原理和应用场景；学习步骤：1. 复习数学基础（线性代数、概率） 2. 学习经典算法（线性回归、决策树） 3. 用Scikit-learn实操；注意事项：先理解原理，再动 手实现，避免死记硬背。


学科：LangChain
学习方法：

生成的LangChain学习方法：
学科：LangChain  
学习方法：
**核心目标**：掌握LangChain框架的核心概念与组件，能够构建基于大语言模型的应用。
**学习步骤**：
1. **理解核心概念**：学习LangChain的核心模块（如Models、Prompts、Chains、Memory、Agents等）及其作用。
2. **掌握基础用法**：通过官方文档和示例，学习如何调用大语言模型、设计提示词、构建链式任务。
3. **实操项目**：尝试构建简单应用，如问答系统、文本摘要工具或自定义Agent。
4. **深入学习高级功能**：探索索引、检索、回调等高级模块，优化应用性能与用户体验。
**注意事项**：
- 结合具体应用场景学习，避免脱离实际。
- 多参考官方文档和社区案例，理解设计思想。
- 注重模块化开发，逐步迭代复杂功能。
```

少样本提示模板在需要“规范输出格式”的场景中非常实用，比如生成报告、整理数据、写标准化文档等。记住这个核心逻辑：给模型看示例，它就会照着做。

### 2.2.3 工程化实践：少样本提示模板的高效管理

基础案例中我们将示例硬编码在代码里，但在工程实践中，会面临示例数量多、需要动态更新、按条件筛选示例等问题。这部分将解决FewShotPromptTemplate的工程化核心痛点：动态示例选择、批量示例管理、性能优化。

#### 2.2.3.1 核心痛点与解决方案

工程化场景中使用少样本模板的常见问题及对应方案：

- 示例过多导致提示词冗长：使用ExampleSelector动态筛选相关示例，避免冗余
- 示例维护困难：将示例存储在文件（JSON/CSV）中，批量加载与更新
- 不同场景需要不同示例：按输入参数匹配示例类型，实现场景化示例分发

#### 2.2.3.2 动态示例选择：ExampleSelector的使用

LangChain提供ExampleSelector组件，能根据输入的动态参数（如主题难度、输入长度）筛选最相关的示例，减少提示词体积，提升模型响应效率。以下是“按主题难度匹配示例”的工程化案例：

```python
# 导入工程化所需组件
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors.length_based import LengthBasedExampleSelector
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json

# 1. 环境初始化（工程化标准操作：环境变量管理密钥）
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
    max_tokens=300
)

# 2. 工程化示例管理：从JSON文件加载示例（避免硬编码，便于维护）
with open("learning_method_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)
# 示例文件格式参考（learning_method_examples.json）：
# [
#   {"subject": "Python编程（入门）", "difficulty": "easy", "method": "核心目标：掌握基础语法；学习步骤：1.变量与数据类型 2.条件语句；注意事项：边学边练"},
#   {"subject": "Python编程（进阶）", "difficulty": "hard", "method": "核心目标：掌握面向对象与库开发；学习步骤：1.类与对象 2.模块开发；注意事项：参与开源项目"},
#   {"subject": "机器学习（入门）", "difficulty": "easy", "method": "核心目标：理解基础概念；学习步骤：1.数据预处理 2.简单模型；注意事项：用Excel辅助理解"},
#   {"subject": "机器学习（进阶）", "difficulty": "hard", "method": "核心目标：掌握模型优化；学习步骤：1.特征工程 2.超参数调优；注意事项：研读论文复现实验"}
# ]

# 3. 定义ExampleSelector：按难度筛选示例（输入含difficulty参数）
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["subject", "difficulty", "method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
    ),
    max_length=150,  # 控制示例总长度，避免提示词过长
    get_text_length=lambda x: len(x)  # 长度计算函数
)

# 4. 构建工程化少样本模板
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 替换固定examples为动态选择器
    example_prompt=PromptTemplate(
        input_variables=["subject", "difficulty", "method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
    ),
    suffix="学科：{new_subject}\n难度：{new_difficulty}\n学习方法：",
    input_variables=["new_subject", "new_difficulty"]  # 新增难度参数
)

# 5. 动态生成不同难度的提示词
# 场景1：生成入门级LangChain学习方法
formatted_prompt_easy = few_shot_prompt.format(
    new_subject="LangChain",
    new_difficulty="easy"
)
print("入门级少样本提示词：")
print(formatted_prompt_easy)
result_easy = chat_model.invoke([{"role": "user", "content": formatted_prompt_easy}])
print("\n入门级学习方法：")
print(result_easy.content)

# 场景2：生成进阶级LangChain学习方法
formatted_prompt_hard = few_shot_prompt.format(
    new_subject="LangChain",
    new_difficulty="hard"
)
print("\n进阶级少样本提示词：")
print(formatted_prompt_hard)
result_hard = chat_model.invoke([{"role": "user", "content": formatted_prompt_hard}])
print("\n进阶级学习方法：")
print(result_hard.content)
```

运行说明：该案例实现了3个工程化特性：1）示例从JSON文件加载，便于批量维护；2）按难度动态筛选示例，适配不同学习阶段需求；3）控制示例总长度，避免触发模型token限制。

#### 2.2.3.3 工程化最佳实践总结

- 示例存储：优先使用JSON/CSV文件或数据库管理示例，避免代码冗余
- 动态筛选：必用ExampleSelector组件，按输入特征（长度、难度、类型）匹配示例
- 性能优化：控制示例总长度，结合模型token限制设计max_length参数
- 版本控制：对示例文件进行版本管理，便于回溯与迭代

## 2.3 输出解析：让输出更可控

通过 PromptTemplate 和 FewShotPromptTemplate，我们解决了**“如何规范输入”**的问题；而输出解析（Output Parsing / Output Control）要解决的是另一件事：

> 如何让大模型的输出，能被程序稳定、可靠地直接使用。

在工程实践中，大模型天然倾向于输出**自由文本**，而业务系统需要的是**结构化数据**（列表、字典、对象）。
 输出解析层，正是连接这两者的关键桥梁。

### 2.3.1 为什么需要输出解析层？

没有解析器时，大模型输出存在两大问题：

- 格式不固定：同一需求可能返回段落、分点、表格等多种格式，代码难以适配
- 无法直接使用：非结构化文本需要手动写正则、字符串分割等逻辑提取信息，开发效率低且易出错

输出解析层的核心价值对比：

| 场景需求           | 无解析器（原始输出）                              | 有解析器（结构化输出）                                       |
| :----------------- | :------------------------------------------------ | :----------------------------------------------------------- |
| 提取商品评论关键词 | “这个手机续航好、拍照清晰，但系统有点卡”          | ["续航好", "拍照清晰", "系统卡顿"]                           |
| 生成用户信息       | “用户叫张三，25岁，手机号138xxxx1234”             | {"name":"张三","age":25,"phone":"138xxxx1234"}               |
| 分析订单状态       | “订单号A123已发货，预计3天到；订单B456还在待付款” | [{"orderId":"A123","status":"已发货","estimate":"3天"},{"orderId":"B456","status":"待付款"}] |

### 2.3.2 输出解析层实践案例

#### 2.3.2.1案例1：StrOutputParser

适用场景

仅需要**纯文本输出**，不要求结构化数据，但希望将大模型输出**统一为标准字符串类型（str）**，以便在 LangChain / LangGraph 中作为稳定的数据节点继续流转。  
适用于文本总结、简单问答、多智能体发言内容生成、字符串拼接等基础场景。

StrOutputParser 的核心作用**不是清洗文本格式**，也不会主动去除换行或空格，而是：

> **将 LLM 返回的 `AIMessage` 对象，统一转换为纯字符串（str）**

从而解决 LangChain 1.0.0 以后最常见的工程问题：

- 避免在业务代码中频繁处理 `AIMessage.content`
- 防止消息对象混入 LangGraph 的状态（State）中，污染状态流
- 让模型输出可以直接参与后续字符串处理、条件判断或二次解析
- 作为更复杂解析流程（如 JSON 解析、规则解析）的**底座组件**

在 LangChain 1.0.0 以后，StrOutputParser 是**兼容性最好、稳定性最高、使用成本最低**的输出解析方案，适合作为所有复杂系统的起点。

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 环境初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

# 2. 初始化模型（无需支持原生结构化输出）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 3. 创建 StrOutputParser
# 核心作用：将 LLM 返回的 AIMessage 对象，统一转为纯字符串（str）
parser = StrOutputParser()

# 4. 链式调用：模型 → 字符串解析
chain = llm | parser
result = chain.invoke("请简要介绍 LangChain 输出解析层的作用")

print("StrOutputParser 解析后的字符串：")
print(result)
print("\n解析结果类型：", type(result))  # str
```

#### 2.3.2.2 案例2： JsonOutputParser

适用场景：快速搭建 Demo，需要简单的键值对结构（无需复杂类型校验），配置简单，无需定义数据模型。核心是自动引导模型输出 JSON 格式，并转化为 Python 字典。

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. 环境与模型初始化（省略，同方案1）
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 2. 创建 JSON 解析器（无需额外配置，默认引导模型输出 JSON）
parser = JsonOutputParser()

# 3. 构建提示模板（无需手动嵌入格式指令，解析器自动关联）
prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具，输出工具名和核心功能。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 链式调用（LangChain ≥1.0.0 推荐方式，自动完成提示+调用+解析）
chain = prompt | llm | parser
result = chain.invoke({})  # 无输入参数，传入空字典

print("解析后的JSON（Python字典）：")
print(result)
print("获取单个字段：", result["tool_name"])  # 可直接用于业务逻辑
```

运行结果示例：

````
解析后的JSON（Python字典）：
{'tool_name': 'LangSmith', 'core_function': '提供全链路的LLM应用开发、调试、测试、监控和部署平台，支持追踪和可视化LangChain应用的执行过程，帮助开发者分析性能、诊断问题并优化提示词与工作流。'}
获取单个字段： LangSmith
````

> ⚠️ 注意：JsonOutputParser 不校验字段类型与枚举值，输出格式“看起来正确”不等于“工程上安全”。

#### 2.3.2.3 案例3：PydanticOutputParser

适用场景：工程化系统、教学场景，需要强类型校验、清晰的字段描述（便于团队协作和维护）。这是 LangChain工程的默认主线方案。核心是通过 Pydantic 定义数据模型，解析器自动校验输出格式，不符合模型的输出会直接报错，提前规避风险。

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# 1. 环境与模型初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 2. 定义 Pydantic 数据模型
class ToolInfo(BaseModel):
    tool_name: str = Field(description="LangChain开发工具的名称，如 LangSmith")
    function: str = Field(description="工具的核心功能，30字以内")
    difficulty: str = Field(description="学习难度，仅可选：简单 / 中等 / 复杂")

# 3. 创建解析器
parser = PydanticOutputParser(pydantic_object=ToolInfo)

# 4. Prompt + Chain
prompt = PromptTemplate(
    template="请介绍1个 LangChain 开发工具，严格按照要求输出。\n{format_instructions}",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | llm | parser
result = chain.invoke({})

print("解析后的结构化数据（Pydantic 模型对象）：")
print(result)

print("字段校验 difficulty：", result.difficulty)

print("转化为字典：", result.model_dump())

```

运行结果示例：

```
解析后的结构化数据（Pydantic 模型对象）：
tool_name='LangSmith' function='用于调试、测试、评估和监控LLM应用的全链路平台' difficulty='中等'
字段校验 difficulty： 中等
转化为字典： {'tool_name': 'LangSmith', 'function': '用于调试、测试、评估和监控LLM应用的全链路平台', 'difficulty': '中等'}
```

### 2.3.3 BaseOutputParser 核心抽象接口（进阶选学）

前面讲解的 StrOutputParser、JsonOutputParser、PydanticOutputParser 等，本质上都是 `BaseOutputParser` 的子类。BaseOutputParser 是 LangChain 中所有输出解析器的**抽象基类**，核心作用是定义统一的解析器接口规范，所有具体解析器都必须实现它的抽象方法，同时它也是我们实现“自定义解析器”的核心基础。

> 本部分难度系数较高，进阶选学~

`BaseOutputParser` 定义了两个核心抽象方法，任何继承它的自定义解析器都**必须实现**：

1. `parse(text: str) -> Any`  
   - 核心解析逻辑：接收模型原始文本，将其转化为目标格式（字符串、字典、自定义对象等）  
   - 解析失败需抛出异常（供 `RetryOutputParser` 捕获重试）  

2. `get_format_instructions() -> str`  
   - 返回模型输出的格式要求提示词  
   - 可嵌入 `PromptTemplate`，确保模型按指定格式输出  

⚠ 注意：`BaseOutputParser` 是抽象类，不能直接实例化，必须继承并实现抽象方法。

**实操：自定义解析器示例**

适用场景：开箱解析器无法满足需求时（如特殊分隔符、自定义标记等），通过继承 `BaseOutputParser` 实现自定义逻辑。

要求模型按 `"工具名@核心功能@学习难度"` 输出，并解析为字典。

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 环境初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 自定义解析器
class CustomToolParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        """将模型输出按 '工具名@核心功能@学习难度' 解析为字典"""
        text = text.strip().replace("\n", "").replace(" ", "")
        parts = text.split("@")
        if len(parts) != 3:
            raise ValueError(f"输出格式错误！需满足「工具名@核心功能@学习难度」，当前输出：{text}")
        return {
            "tool_name": parts[0].strip(),
            "function": parts[1].strip(),
            "difficulty": parts[2].strip()
        }

    def get_format_instructions(self) -> str:
        """生成提示词，引导模型按自定义格式输出"""
        return "请严格按照「工具名@核心功能@学习难度」格式输出，不添加多余内容。示例：LangSmith@全链路调试监控@中等"

# 使用解析器
parser = CustomToolParser()
prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser
result = chain.invoke({})

print("自定义解析器解析结果：")
print(result)
print("解析结果类型：", type(result))
```

运行结果

```
自定义解析器解析结果：
{'tool_name': 'LangFlow', 'function': '可视化编排LangChain组件', 'difficulty': '低'}
解析结果类型： <class 'dict'>
```

## 2.4 输入控制层核心总结

输入控制层的核心目标是实现“输入可控制、输出可预期”，关键组件的核心价值：

- PromptTemplate：通过参数化设计实现提示词的规范与复用，降低重复开发成本
- FewShotPromptTemplate（工程化）：通过动态示例选择与批量管理，适配复杂业务场景，提升提示词效率
- OutputParser：将非结构化输出转化为结构化数据，打通大模型输出与后续业务逻辑的衔接

实践原则：输入控制层的设计需结合具体业务场景，优先使用LangChain开箱即用组件，复杂场景通过自定义扩展满足需求，始终兼顾易用性与工程化可维护性。

## 2.5 本章小节

本章核心围绕LangChain输入控制层的实操应用与工程化落地展开，本章重点掌握三大核心组件的协同逻辑,关键要点总结如下：

- 1.模型调用组件通过统一接口适配不同厂商模型，解决多模型切换的适配难题；
- 2.提示词模板（基础/少样本）通过参数化与示例引导实现提示规范复用，结合ExampleSelector完成工程化管理；
- 3.输出解析器将非结构化输出转化为结构化数据，搭配重试机制保障输出可靠性，三者共同实现“输入-输出”全链路可控。

## 2.6 本章练习

1. 复现本章核心案例：包括模型调用、提示词模板、输出解析等，确保所有案例均可成功运行并输出预期结果。
2. 修改提示词模板与解析器：基于“学习建议生成”案例，在PromptTemplate中新增“难度等级”动态参数，重新运行并验证输入参数与输出结构的匹配性。
4. 综合选型思考：若开发一个“结构化报告生成工具”（需固定提示格式、输出结构化数据，但无需多轮流程），需组合本章哪些核心组件？说明理由。

