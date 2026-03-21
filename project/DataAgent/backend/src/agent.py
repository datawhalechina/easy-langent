import os
from langchain_deepseek import ChatDeepSeek
# [关键] 使用 LangChain 1.1 标准 API 和 中间件工具
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

# 导入工具和数据管理器
from src.tools import python_inter, fig_inter
from src.data_manager import get_data_info

# =============================================================================
# 1. 初始化模型
# =============================================================================
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com"
)

tools = [python_inter, fig_inter]

# =============================================================================
# 2. 定义动态 System Prompt 中间件
# =============================================================================

@dynamic_prompt
def dataset_context_middleware(request) -> str:
    """
    [中间件] 动态提示词生成器
    LangChain 会在每次调用 Agent 之前自动运行这个函数。
    
    request: 包含了当前的 ModelRequest (如 input, messages 等)
    返回: 最新的 System Prompt 字符串
    """
    
    # 1. 实时去内存获取最新的数据摘要（包含文件名、行列数、列名等）
    # 只要用户上传了新文件，get_data_info() 的返回结果就会立刻变化
    data_context = get_data_info()
    
    # 2. 返回完整的 System Prompt
    return f"""你是一名精通 Python 的数据分析专家 DataAgent。

【当前数据集实时状态】
{data_context}

【你的职责】
1. 使用 `python_inter` 执行 Pandas 分析，或 `fig_inter` 进行绘图。
2. 变量 `df` 已内置，直接使用即可。
3. 绘图时请将对象赋值给变量，并调用绘图工具。
4. 遇到问题请先尝试分析报错信息。
"""

# =============================================================================
# 3. 创建 Agent (带中间件)
# =============================================================================

# 使用 LangChain 1.1 的 create_agent
# 注意：我们不再直接传 system_prompt="..." 字符串，
# 而是通过 middleware 参数传入动态生成器。
graph = create_agent(
    model=model,
    tools=tools,
    middleware=[dataset_context_middleware], # 注入中间件
)