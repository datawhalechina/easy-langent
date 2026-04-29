# 谁是卧底多智能体毕业设计

## 项目简介
本项目基于 `LangChain + LangGraph` 实现了 chapter8 对应的“谁是卧底”多智能体游戏引擎，采用上帝视角运行：4 个智能体自动完成词语生成、角色分配、发言、投票和胜负判断，适合作为课程综合实践或毕业设计演示项目。

## 核心功能
- 自动生成一组平民词与卧底词。
- 随机分配 1 个卧底和 3 个平民。
- 根据身份和历史上下文生成多智能体发言。
- 根据发言内容自动进行投票。
- 基于 LangGraph 串联完整游戏流程。
- 提供 `pytest` 测试，保证核心逻辑可验证。

## 技术栈
- Python 3.11+
- LangChain
- LangGraph
- langchain-openai
- python-dotenv
- pytest

## 项目结构
```text
who_is_the_spy_graduation/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── Readme.md
├── src/
│   └── game/
│       ├── __init__.py
│       ├── state.py
│       ├── model.py
│       ├── logic.py
│       ├── graph.py
│       └── main.py
└── tests/
    ├── test_game_state.py
    ├── test_nodes.py
    └── test_graph.py
```

## 环境配置
先复制环境变量模板，再填写自己的模型密钥：

```bash
cp .env.example .env
```

`.env` 示例：

```env
API_KEY=your_api_key_here
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

注意：`.env` 不要提交到仓库。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 运行项目
在项目根目录执行：

```bash
python -m game.main
```

如果你的 Python 环境没有自动识别 `src` 布局，也可以使用：

```bash
PYTHONPATH=src python -m game.main
```

## 运行测试
```bash
pytest -q
```

## 示例输出
```text
==================================================
谁是卧底 · 多智能体多轮策略版
==================================================
平民词：牙刷
卧底词：牙膏
总轮次：1
淘汰顺序：['agent4']
胜利方：平民
```

## 设计说明
- 使用 `TypedDict` 保存整局游戏状态，方便节点之间共享数据。
- 使用 `LLMAdapter` 隔离真实模型调用与测试替身，保证测试不依赖外部网络。
- 使用 LangGraph 将“词语生成 -> 角色分配 -> 发言 -> 投票 -> 裁决 -> 总结”串成状态图。
- 对模型输出异常、非法投票等情况增加了兜底逻辑，确保流程稳定运行。

## 提交说明
本项目符合 chapter8 的基础提交要求：
- 包含可运行核心代码；
- 包含项目说明文档；
- 包含测试代码；
- 不提交 `.env`，避免泄露密钥。
