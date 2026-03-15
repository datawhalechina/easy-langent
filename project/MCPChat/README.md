# MCPChat - AI Agent with MCP Integration

基于 LangChain 1.1 和 MCP (Model Context Protocol) 的智能 AI Agent 应用，支持流式对话、工具管理和会话历史。

## 项目简介

MCPChat 是一个可扩展的 AI Agent 平台，通过 MCP (Model Context Protocol) 协议集成各种外部工具，让 AI 具备调用外部服务的能力。

### 核心特性

- **流式对话**：支持实时流式响应，提供流畅的交互体验
- **MCP 工具管理**：动态添加、删除、启用/禁用 MCP 工具
- **AI 智能推荐**：根据用户需求推荐合适的 MCP 工具
- **会话历史**：自动保存对话记录，支持多会话管理
- **内置工具**：集成天气查询、联网搜索等常用工具

### MCP 工具推荐说明

AI 工具推荐功能仅在 [`mcp_registry.json`](backend/mcp_registry.json) 中定义的工具范围内进行搜索匹配。如需支持更多工具推荐，请编辑该文件添加新的工具定义。

当前内置工具：
- `time` - 时间查询和时区转换
- `sqlite` - SQLite 数据库操作
- `amap-maps` - 高德地图服务（地点搜索、路径规划、天气）

## 📁 项目结构

```
MCPChat/
├── backend/                    # 后端服务
│   ├── server.py              # FastAPI 主服务
│   ├── agent.py               # LangChain Agent 构建逻辑
│   ├── mcp_manager.py         # MCP 工具管理器
│   ├── history.py             # 会话历史管理
│   ├── tools.py               # 内置工具
│   ├── mcp_registry.json      # MCP 工具注册表
│   ├── pyproject.toml         # uv 项目配置
│   └── uv.lock                # uv 锁定文件
│
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── components/        # React 组件
│   │   ├── utils/             # 工具函数
│   │   ├── types/             # TypeScript 类型
│   │   └── App.tsx            # 主应用组件
│   ├── package.json
│   └── vite.config.ts
│
└── .gitignore
```

## 🚀 快速开始

### 前置要求

- **Node.js** >= 18.0.0
- **Python** >= 3.10
- **uv** >= 0.5.0

### 安装 uv

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/MacOS
# 使用 curl 安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用 wget 安装
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 后端设置

```bash
cd backend

# 安装 Python 并同步依赖
uv python install 3.11
uv sync

# 配置环境变量
cp .env.template .env
# 参考注释说明，编辑 .env 填入 API Keys
```

### 启动后端

```bash
uv run python server.py
```

服务默认在 `http://localhost:8002` 启动。

### 前端设置

```bash
cd frontend
pnpm install
pnpm dev
```

前端默认在 `http://localhost:5173` 启动。

## ⚙️ uv 常用命令

```bash
# 同步依赖
uv sync

# 添加依赖
uv add <package-name>

# 运行脚本
uv run python server.py

# 更新依赖
uv sync --upgrade
```

## 📄 许可证

[MIT License](LICENSE)
