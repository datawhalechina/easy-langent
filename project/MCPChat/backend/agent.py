# agent.py
import asyncio
from typing import List

# 1. 引入 LangChain 和 DeepSeek
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.tools import BaseTool

# 2. 引入 MCP 官方适配器
from langchain_mcp_adapters.client import MultiServerMCPClient

# 3. 引入本地模块
from tools import get_tools as get_builtin_tools  # 获取天气和搜索
from mcp_manager import MCPManager

# 全局实例化 Manager，用于读取配置（但不保存状态，状态在 json 文件里）
mgr = MCPManager()

async def build_dynamic_agent():
    """
    [核心工厂函数]
    每次对话前调用。动态组装【内置工具】+【已激活 MCP 工具】，并生成动态 Prompt。
    """
    
    # ==========================================
    # Step 1: 收集所有工具 (Tools Assembly)
    # ==========================================
    
    # 1.1 获取内置工具 (Weather, Tavily)
    tools: List[BaseTool] = get_builtin_tools()
    
    # 1.2 获取当前激活的 MCP 配置
    mcp_config = mgr.get_active_config()
    
    mcp_tools: List[BaseTool] = []
    
    # 1.3 如果有激活的 MCP，建立连接并获取工具
    if mcp_config:
        try:
            # 这里的 client 仅仅用于获取工具定义，不会长期持有连接
            # LangChain Agent 执行时会再次通过这个定义去调用
            client = MultiServerMCPClient(mcp_config)
            mcp_tools = await client.get_tools()
            print(f"🔌 [Agent Factory] 已动态挂载 MCP 工具: {[t.name for t in mcp_tools]}")
        except Exception as e:
            print(f"⚠️ [Agent Factory] MCP 挂载失败，将降级运行: {e}")
            
    # 合并工具列表
    all_tools = tools + mcp_tools

    # ==========================================
    # Step 2: 动态构建系统提示词 (Dynamic Prompting)
    # ==========================================
    
    # 2.1 提取工具名称和描述，生成清单
    tool_descriptions = []
    for t in all_tools:
        tool_descriptions.append(f"- **{t.name}**: {t.description}")
    
    tools_str = "\n".join(tool_descriptions)

    # 2.2 编写动态 Prompt
    # 这里的关键是把 tool_str 嵌入进去，让模型知道它拥有哪些能力
    system_prompt = f"""
    你是一个功能强大的全能 AI 智能体。
    
    ### 🛠️ 你当前拥有的工具能力：
    {tools_str}
    
    ### 🧠 思考与行动指南：
    1. **优先使用工具**：如果用户的请求可以通过上述工具解决，请务必调用工具，不要仅凭记忆瞎编。
    2. **内置工具规则**：
       - 查询天气 -> 必须使用 `get_weather`。
       - 搜索新闻/实时信息 -> 必须使用 `search_tool` (Tavily)。
    3. **MCP 工具规则**：
       - 请仔细阅读上述工具列表的描述。
       - 如果用户请求涉及文件操作、数据库查询、精确时间或地图服务，请在列表中寻找对应的 MCP 工具。
    4. **多步推理**：如果一个任务需要多个步骤（例如：先查天气，再发邮件），请按逻辑顺序连续调用工具。
    5. **语言**：始终使用简体中文回答用户，保持友善和专业。
    
    现在，请根据用户的输入，灵活选择工具开始工作。
    """

    # ==========================================
    # Step 3: 创建并返回 Agent
    # ==========================================
    
    model = ChatDeepSeek(model="deepseek-chat", temperature=0)
    
    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt
    )
    
    return agent