# WhoIsTheSpy

## 项目简介

本项目是 Easy-Langent 第八章综合实践作业：基于 LangGraph 构建“谁是卧底”多智能体游戏。

当前版本采用教程的基础路线：用户作为旁观者，4 个智能体自动完成词语生成、角色分配、发言、投票、淘汰和胜负判断。

## 核心功能

- 自动生成平民词和卧底词。
- 随机分配 4 个智能体角色。
- 每轮由未淘汰智能体依次发言。
- 每轮根据发言自动投票。
- 根据投票结果淘汰玩家并判断胜负。
- 使用 LangGraph 的节点、边、条件跳转组织游戏流程。
- 增加模型调用失败时的兜底逻辑，避免程序中途崩溃。

## Python 版本

推荐使用 Python 3.11 或更高版本。本地测试环境：Python 3.13。

## 依赖安装

如果使用本课程项目根目录下的虚拟环境，通常已安装相关依赖。

如需重新安装，可在项目根目录执行：

```bash
pip install langchain langgraph langchain-openai langchain-core python-dotenv
```

## 环境变量配置

请在项目根目录创建 `.env` 文件，不要把 `.env` 放入本作业目录，也不要提交 `.env`。

示例：

```text
API_KEY=你的模型API密钥
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
```

如果使用 LM Studio 本地模型，可将 `BASE_URL` 改为 LM Studio 提供的 OpenAI Compatible API 地址，并将 `MODEL` 改为 LM Studio 当前加载的模型名。

## 运行步骤

在项目根目录执行：

```powershell
cd D:\ai\easy-langent
.\langent-env\Scripts\python.exe .\project\WhoIsTheSpy\who_is_undercover.py
```

或者进入本目录后运行：

```powershell
D:\ai\easy-langent\langent-env\Scripts\python.exe .\who_is_undercover.py
```

## 文件说明

```text
who_is_undercover.py   核心代码文件
Readme.md             项目说明文档
run_result.txt        本地运行输出记录，可选提交
```

## 参考资料

- Easy-Langent 第八章：综合实践：构建“谁是卧底”游戏智能体
- 项目仓库：https://github.com/datawhalechina/easy-langent

## 注意事项

- 禁止提交 `.env` 文件，避免 API 密钥泄露。
- 如果模型返回的不是严格 JSON，程序会自动使用兜底逻辑继续运行。
- 本项目完成的是基础版“上帝视角”，用户只旁观，不参与发言和投票。
