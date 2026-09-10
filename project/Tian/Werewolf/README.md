# Werewolf —— AI 多智能体狼人杀

> **作者**：田一博
>
> **完整学习笔记**：<https://scnj8i8shsj3.feishu.cn/wiki/XHU1w2uhjixsGZkXUIOcA3rRnDe?fromScene=spaceOverview>

## 项目简介

一个基于 **LangGraph + LangChain** 构建的全自动 AI 狼人杀游戏。8 个由大语言模型（LLM）驱动的 AI 玩家分别扮演狼人、村民、预言家、女巫、猎人，自动完成角色分配、夜晚行动、白天发言、投票，直到分出胜负。整个游戏无需人工干预，可在终端观察每个 AI 玩家的决策、发言和推理过程。

项目使用智谱 AI（GLM）作为底层大模型，通过 LangGraph 的状态图（StateGraph）编排游戏流程，通过精心设计的角色 Prompt 让 AI 表现出符合各自身份的策略行为（狼人伪装、预言家引导、女巫用药等）。

---

## 核心功能

- **多智能体博弈**：8 个 AI 玩家独立决策，各自拥有隔离的私有记忆
- **完整角色系统**：狼人、村民、预言家、女巫、猎人，各具技能与策略
- **昼夜循环流程**：夜晚（狼人击杀 → 女巫用药 → 预言家查验）+ 白天（发言 → 投票）
- **信息不对称**：公开历史所有人可见，私有记忆各自独立，还原真实狼人杀
- **技能连锁处理**：女巫解药/毒药、猎人开枪等连锁反应有序结算
- **自动胜负判定**：狼人全灭（好人胜）/ 屠边（狼人胜）
- **全程可观察**：终端实时打印每个阶段、每个 AI 的发言与推理

---

## Python 版本

```
Python >= 3.14
```

---

## 依赖包安装

项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，在项目根目录执行：

```bash
uv sync
```

若未安装 uv，可先安装：

```bash
pip install uv
```

或手动通过 pip 安装核心依赖：

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

---

## 环境变量配置

在项目根目录创建 `.env` 文件，填入智谱 AI 的 API 密钥与接口地址：

```
ZHIPU_API_KEY=你的API密钥
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

> 获取密钥：登录 [智谱开放平台](https://open.bigmodel.cn) → API Keys 页面创建。

---

## 运行步骤

**1. 进入你的项目目录**

```bash
cd <你的项目目录>
```

**2. 安装依赖**

```bash
uv sync
```

**3. 配置环境变量**

在项目根目录创建 `.env` 文件，填入上方的 API 密钥配置。

**4. 运行游戏**

```bash
uv run python <Werewolf.py 所在路径>/Werewolf.py
```

**5. 观察游戏进程**

终端会依次打印：
- 角色分配结果（上帝视角）
- 每一夜的狼人击杀、女巫用药、预言家查验
- 每一天的玩家发言、投票结果
- 死亡公告与最终胜负

---

## 自定义配置

在 `Werewolf.py` 的 `__main__` 中修改 `create_initial_state` 的参数即可调整局型：

```python
init_state = create_initial_state(
    total_players=8,
    role_distribution={
        "werewolf": 2,   # 狼人
        "villager": 3,   # 村民
        "seer": 1,       # 预言家
        "witch": 1,      # 女巫
        "hunter": 1,     # 猎人
    },
)
```

注意：`role_distribution` 中各角色数量之和必须等于 `total_players`，否则会报错。

---

## 注意事项

- 一局游戏会调用大量 LLM（每个玩家每轮发言/投票 + 夜晚行动），运行时间较长，请耐心等待
- 已配置单次请求 60 秒超时 + 失败自动重试 3 次，应对网络波动
- 若遇到 `APIConnectionError`，多为网络问题或 API 额度不足，请检查网络与账户余额
- 建议先用小配置（如 4-5 人局）快速验证流程，再运行完整 8 人局
- 详细的代码设计说明见同目录下的 `Werewolf_教程.md`
