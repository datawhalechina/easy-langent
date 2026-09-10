# NovelCreationMultiAgent

## 项目简介

基于 LangGraph + LangChain 构建的多智能体小说创作系统。用户输入创作需求后，系统通过多个 LLM 节点依次完成基础设定生成、大纲规划、逐章正文生成，每个关键阶段均支持人工审核与修改，最终将完整小说保存为本地文本文件。

---

## 核心功能

- **需求收集**：交互式输入小说创作需求（类型、主角、情节要求等）
- **基础设定生成**：自动生成小说题目、主要角色、情节概述，支持人工修改，最多重试 3 次
- **大纲与章节规划**：生成整体大纲及至少 8 章的章节结构，支持人工修改，最多重试 3 次
- **逐章正文生成**：按章节结构逐一调用 LLM 生成正文，实时打印进度
- **自动保存**：完整小说以「小说名称.txt」保存到本地

---

## Python 版本

```
Python >= 3.14
```

---

## 依赖包安装

项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，执行以下命令安装所有依赖：

```bash
uv sync
```

若未安装 uv，可先安装：

```bash
pip install uv
```

或手动通过 pip 安装核心依赖：

```bash
pip install langchain langchain-openai langgraph python-dotenv
```

---

## 环境变量配置

在项目根目录创建 `.env` 文件，填入智谱 AI 的 API 密钥与接口地址：

```
ZHIPU_API_KEY=你的API密钥
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

---

## 运行步骤

**1. 克隆项目并进入目录**

```bash
cd Easy_langent
```

**2. 安装依赖**

```bash
uv sync
```

**3. 配置环境变量**

在项目根目录创建 `.env` 文件，填入上方的 API 密钥配置。

**4. 运行程序**

```bash
uv run python easy_langent/NovelCreationMultiAgent.py
```

**5. 按提示交互**

- 输入小说创作需求，例如：`科幻类型，主角是计算机专业大学生，要有AI相关的反转情节`
- 系统生成基础设定后，输入 `YES` 确认或 `NO` 修改
- 系统生成大纲与章节后，输入 `YES` 确认或 `NO` 修改
- 确认后自动逐章生成正文，完成后保存为 `小说名称.txt`

---

## 注意事项

- 基础设定与大纲各最多支持 **3 次**人工修改，超出次数后流程自动终止
- 生成的小说文件保存在运行命令时的当前目录下
- 使用模型为 `glm-5.2`，需确保账户有足够的 API 调用额度
