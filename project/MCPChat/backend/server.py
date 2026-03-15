# server.py
import uvicorn
import os
import json
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# --- 导入本地模块 ---
# 请确保 agent.py 中有 build_agent_with_mcp 函数
# 请确保 history.py, mcp_manager.py 文件存在且最新
from history import HistoryManager
from mcp_manager import MCPManager
from agent import build_dynamic_agent

# 1. 加载环境变量
load_dotenv(override=True)

# 2. 初始化 FastAPI
app = FastAPI(title="Creative Agent Backend", version="4.0 (MCP Edition)")

# 3. 配置跨域 (CORS) - 允许前端所有访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 初始化核心管理器
# history_manager 在每次请求时动态实例化
mcp_manager = MCPManager()

print(f"🚀 Server 初始化完成")
print(f"📂 历史记录路径: {os.path.abspath('chat_history')}")
print(f"🛠️ MCP 配置文件: {os.path.abspath('mcp_config.json')}")

# ==========================================
#        Pydantic 数据模型定义
# ==========================================

# --- 聊天相关 ---
class ChatRequest(BaseModel):
    query: str
    session_id: str

class RenameRequest(BaseModel):
    title: str

class SessionItem(BaseModel):
    id: str
    title: str
    updated_at: int

# --- MCP 管理相关 ---
class MCPSearchRequest(BaseModel):
    query: str  # 用户输入的自然语言需求

class MCPInstallRequest(BaseModel):
    """用于安装或修改单个 MCP 工具"""
    name: str
    description: str
    type: str       # "stdio" 或 "sse"
    config: dict    # 包含 command, args, url, headers 等
    
class MCPBatchInstallRequest(BaseModel):
    """用于批量安装 (AI推荐列表勾选后一键安装)"""
    tools: List[MCPInstallRequest]

class MCPToggleRequest(BaseModel):
    active: bool

class MCPTestRequest(BaseModel):
    """用于连接测试"""
    name: str
    type: str
    config: dict

# ==========================================
#           API 模块 1: 会话管理
# ==========================================

@app.get("/sessions", response_model=List[SessionItem])
async def get_sessions():
    """获取会话列表"""
    return HistoryManager.get_all_sessions()

@app.post("/sessions")
async def create_session():
    """创建新会话"""
    import uuid
    import time
    return {
        "id": str(uuid.uuid4()),
        "title": "新对话",
        "updated_at": int(time.time())
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    HistoryManager.delete_session(session_id)
    return {"status": "success"}

@app.patch("/sessions/{session_id}/title")
async def rename_session(session_id: str, request: RenameRequest):
    """重命名会话"""
    if HistoryManager.rename_session(session_id, request.title):
        return {"status": "success", "new_title": request.title}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """获取会话详情"""
    return HistoryManager(session_id).get_full_history()

# ==========================================
#           API 模块 2: MCP 工具管理
# ==========================================

@app.get("/mcp/list")
async def list_installed_mcp():
    """
    [功能 1] 获取当前载入的 MCP 工具列表
    包含：名字、介绍、激活状态、完整配置JSON
    """
    return mcp_manager.list_installed_tools()

@app.post("/mcp/search_ai")
async def search_mcp_ai(req: MCPSearchRequest):
    """
    [功能 4] AI 智能搜索
    输入需求 -> 输出推荐列表
    """
    return await mcp_manager.ai_recommend_tools(req.query)

@app.post("/mcp/install")
async def install_mcp_tool(req: MCPInstallRequest):
    """
    [功能 2 & 3] 安装/修改 单个工具
    前端'保存配置'或'新增工具'时调用
    """
    mcp_manager.save_tool(
        name=req.name,
        description=req.description,
        type=req.type,
        config_dict=req.config
    )
    return {"status": "success", "message": f"工具 {req.name} 已保存"}

@app.post("/mcp/install_batch")
async def install_mcp_batch(req: MCPBatchInstallRequest):
    """
    [功能 4 补充] 批量安装
    用户在推荐列表中勾选多个后，一键调用此接口
    """
    count = 0
    for tool in req.tools:
        mcp_manager.save_tool(
            name=tool.name,
            description=tool.description,
            type=tool.type,
            config_dict=tool.config
        )
        count += 1
    return {"status": "success", "message": f"已批量添加 {count} 个工具"}

@app.post("/mcp/test_connection")
async def test_mcp_connection(req: MCPTestRequest):
    """
    [功能 2 & 3] 测试连接
    在保存之前验证配置是否有效
    """
    success, msg = await mcp_manager.test_tool_connection(
        name=req.name,
        type=req.type,
        config_dict=req.config
    )
    return {"success": success, "message": msg}

@app.post("/mcp/toggle/{tool_name}")
async def toggle_mcp(tool_name: str, req: MCPToggleRequest):
    """
    [功能 5] 激活/禁用 工具
    """
    mcp_manager.toggle_tool(tool_name, req.active)
    return {"status": "success", "active": req.active}

@app.delete("/mcp/{tool_name}")
async def uninstall_mcp(tool_name: str):
    """
    [功能 2] 删除工具
    """
    mcp_manager.delete_tool(tool_name)
    return {"status": "success"}

# ==========================================
#           API 模块 3: 核心流式对话
# ==========================================

# SSE 格式化辅助函数
def format_sse(event_type: str, data: dict):
    """
    将数据封装为 SSE 格式。
    增加了 default=str 参数，防止 ToolRuntime 或 datetime 等对象导致序列化报错。
    """
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False, default=str)}\n\n"

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """
    核心对话接口
    每次请求都会动态重新构建 Agent，以确保加载最新的 MCP 工具配置
    """
    
    # 1. 准备历史上下文
    history_mgr = HistoryManager(request.session_id)
    history_messages = history_mgr.load_messages(limit=40)
    input_messages = history_messages + [HumanMessage(content=request.query)]
    
    # 2. 动态构建 Agent (关键步骤)
    # 这会读取 mcp_config.json 并连接所有激活的 MCP Server
    try:
        current_agent = await build_dynamic_agent()
    except Exception as e:
        # 如果 Agent 构建失败（比如某个 MCP 连不上），返回错误流
        async def error_gen():
            yield format_sse("error", {"message": f"Agent 初始化失败: {str(e)}"})
            yield format_sse("finish", {"status": "error"})
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 3. 定义流生成器
    async def event_generator():
        final_answer = ""
        try:
            print(f"🔄 [Server] Session {request.session_id} 开始处理...")
            
            async for event in current_agent.astream_events(
                {"messages": input_messages},
                version="v2"
            ):
                kind = event["event"]
                name = event.get("name", "")

                # --- Token 流 ---
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        final_answer += content
                        yield format_sse("token", {"content": content})

                # --- 工具开始 ---
                elif kind == "on_tool_start":
                    print(f"🛠️ [Tool Start] {name}")
                    
                    # 1. 获取原始输入
                    raw_input = event["data"].get("input")
                    clean_input = {}

                    # 2. 数据清洗逻辑
                    if isinstance(raw_input, dict):
                        for k, v in raw_input.items():
                            # [关键步骤] 剔除 LangChain/MCP 的内部注入参数
                            # runtime: 包含巨大历史记录
                            # state: 包含 Agent 状态
                            if k in ["runtime", "state", "callbacks"]:
                                continue
                            
                            # [可选] 对剩余参数进行截断（防止用户输入超长文本刷屏）
                            str_v = str(v)
                            if len(str_v) > 200: # 限制每个参数值最多显示 200 字符
                                clean_input[k] = str_v[:200] + "..."
                            else:
                                clean_input[k] = v
                    else:
                        # 如果 input 本身不是 dict (很少见)，直接转字符串并截断
                        clean_input = str(raw_input)[:200] + "..."

                    # 3. 发送清洗后的数据
                    yield format_sse("tool_start", {
                        "tool_name": name,
                        "input": clean_input
                    })

                # --- 工具结束 ---
                elif kind == "on_tool_end":
                    print(f"✅ [Tool End] {name}")
                    raw = event["data"].get("output")
                    # 鲁棒性转换
                    output_str = str(raw)
                    if hasattr(raw, "content"): output_str = raw.content
                    elif isinstance(raw, (dict, list)): output_str = json.dumps(raw, ensure_ascii=False)
                    
                    yield format_sse("tool_end", {
                        "tool_name": name,
                        "output": output_str
                    })

            # 保存历史记录
            if final_answer:
                history_mgr.save_interaction(request.query, final_answer)
            
            yield format_sse("finish", {"status": "success"})

        except Exception as e:
            import traceback
            print(f"❌ [Stream Error] {traceback.format_exc()}")
            yield format_sse("error", {"message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

