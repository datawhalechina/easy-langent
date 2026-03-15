import json
import os
import sys
import asyncio
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# 1. 引入 Pydantic 用于定义结构化输出
from pydantic import BaseModel, Field

# 2. 引入 LangChain 和 DeepSeek 组件
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# 3. 引入 MCP 官方适配器 (用于测试连接)
from langchain_mcp_adapters.client import MultiServerMCPClient

# 加载环境变量
load_dotenv(override=True)

# 文件路径定义
REGISTRY_FILE = "mcp_registry.json"
CONFIG_FILE = "mcp_config.json"

# ==========================================
#   Pydantic 数据模型 (用于 AI 结构化输出)
# ==========================================

class ToolRecommendation(BaseModel):
    """单个工具的推荐信息"""
    name: str = Field(description="推荐的工具名称，必须与知识库中的name字段完全一致")
    reason: str = Field(description="推荐理由，解释为什么这个工具适合用户的需求")

class RecommendationList(BaseModel):
    """最终返回的推荐列表容器"""
    recommendations: List[ToolRecommendation] = Field(description="推荐的工具列表，如果没有合适的工具则为空列表")

# ==========================================
#            MCP 管理器核心类
# ==========================================

class MCPManager:
    def __init__(self):
        self.config = self._load_config()
        self.registry = self._load_registry()
        
        # 初始化 DeepSeek 模型
        # 注意：temperature 设置为 0.1 以确保输出的稳定性
        self.llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)

    # --- 数据加载 ---
    def _load_registry(self) -> List[Dict]:
        if not os.path.exists(REGISTRY_FILE): return []
        try:
            with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []

    def _load_config(self) -> Dict:
        if not os.path.exists(CONFIG_FILE):
            return {"tools": {}}
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {"tools": {}}

    def _save_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    # --- 核心功能 1: AI 智能推荐 (结构化输出版) ---

    async def ai_recommend_tools(self, user_query: str) -> List[Dict]:
        """
        使用 DeepSeek + Pydantic 结构化输出，智能推荐工具
        """
        # 1. 准备知识库摘要
        registry_text = json.dumps([
            {"name": t["name"], "desc": t["description"], "category": t["category"]} 
            for t in self.registry
        ], ensure_ascii=False)

        # 2. 定义 Prompt
        prompt = ChatPromptTemplate.from_template("""
        你是一个专业的 MCP 工具推荐助手。
        
        用户的需求是："{query}"
        
        目前的工具知识库如下：
        {registry}
        
        请分析用户需求，从知识库中挑选出最能解决问题的工具（最多推荐 3 个）。
        如果用户需求模糊或没有匹配工具，请返回空列表。
        """)

        # 3. 绑定结构化输出 (LangChain 1.0 核心特性)
        # 这会自动处理 JSON Schema 转换和解析，替代了繁琐的 parser 代码
        structured_llm = self.llm.with_structured_output(RecommendationList)
        
        chain = prompt | structured_llm

        try:
            # 4. 执行推理
            result: RecommendationList = await chain.ainvoke({
                "query": user_query,
                "registry": registry_text
            })
            
            # 5. 回填详细信息给前端
            final_results = []
            for item in result.recommendations:
                # 在 registry 中找到原始完整数据
                original_tool = next((t for t in self.registry if t["name"] == item.name), None)
                if original_tool:
                    is_installed = item.name in self.config.get("tools", {})
                    
                    final_results.append({
                        **original_tool, 
                        "recommend_reason": item.reason,
                        "installed": is_installed
                    })
            return final_results

        except Exception as e:
            print(f"❌ AI 推荐发生错误: {e}")
            return []

    # --- 核心功能 2: 基础列表查询 ---

    def list_registry(self) -> List[Dict]:
        """列出知识库所有工具"""
        return self.registry

    def list_installed_tools(self) -> List[Dict]:
            """列出已安装工具（带配置详情）"""
            results = []
            for name, data in self.config.get("tools", {}).items():
                # 获取原始配置字典
                raw_config = data.get("config", {})
                
                results.append({
                    "name": name,
                    "description": data.get("description", ""),
                    "active": data.get("active", True),
                    "type": data.get("type"),
                    
                    # 1. 保留原始对象 (为了后端逻辑完整性)
                    "config": raw_config,
                    
                    # 2. ✨ 新增: 格式化后的 JSON 字符串 (专门给前端弹窗编辑器用)
                    # ensure_ascii=False 保证中文不乱码
                    # indent=2 保证显示漂亮的缩进格式
                    "config_json": json.dumps(raw_config, ensure_ascii=False, indent=2)
                })
            return results

    # --- 核心功能 3: 保存/修改工具 ---

    def save_tool(self, name: str, description: str, type: str, config_dict: Dict):
        """
        保存或更新工具配置 (包含智能拆包逻辑 + 读写安全)
        """
        try:
            # 🚨 步骤 0: 先强制重载最新配置，防止覆盖其他进程的修改
            self.config = self._load_config()

            # --- 1. 智能拆包 ---
            real_config = config_dict
            real_type = type.strip().lower()

            # 处理用户粘贴完整 JSON 的情况
            if isinstance(config_dict, dict) and "config" in config_dict and "type" in config_dict:
                print(f"📦 [Save] 检测到嵌套配置，正在自动清洗...")
                real_type = config_dict["type"].strip().lower()
                real_config = config_dict["config"]
                if "name" in config_dict:
                    name = config_dict["name"]
            
            # --- 2. 类型清理 ---
            if real_type == "sse":
                if "command" in real_config:
                    del real_config["command"]

            # --- 3. 路径修正 ---
            if real_type == "stdio" and real_config.get("command") == "python":
                real_config["command"] = sys.executable

            # --- 4. 更新内存 ---
            # 确保 tools 键存在
            if "tools" not in self.config:
                self.config["tools"] = {}

            self.config["tools"][name] = {
                "type": real_type,
                "description": description,
                "active": True,
                "config": real_config
            }

            # --- 5. 写入文件 ---
            self._save_config()
            print(f"✅ 工具 [{name}] 配置已清洗并保存 (Type: {real_type})")

        except Exception as e:
            print(f"❌ 保存工具失败: {str(e)}")
            # 向上抛出异常，让 Server 返回 500，而不是让前端傻等
            raise e

    def install_from_registry(self, registry_name: str):
        """从知识库安装标准模板"""
        target = next((t for t in self.registry if t['name'] == registry_name), None)
        if not target:
            raise ValueError(f"知识库中找不到工具: {registry_name}")
        
        self.save_tool(
            name=target['name'],
            description=target['description'],
            type=target['type'],
            config_dict=target['default_config']
        )

    # --- 核心功能 4: 删除与开关 ---

    def delete_tool(self, name: str):
        """删除工具"""
        if name in self.config["tools"]:
            del self.config["tools"][name]
            self._save_config()

    def toggle_tool(self, name: str, active: bool):
        """激活/禁用工具"""
        if name in self.config["tools"]:
            self.config["tools"][name]["active"] = active
            self._save_config()

    # --- 核心功能 5: 连接测试 (使用 MultiServerMCPClient) ---

    async def test_tool_connection(self, name: str, type: str, config_dict: Dict) -> Tuple[bool, str]:
        """
        测试工具连接 (包含智能拆包逻辑，兼容用户粘贴完整JSON的情况)
        """
        # --- 1. 智能拆包 (防呆逻辑) ---
        # 检查 config_dict 是否包含了嵌套的定义 (用户可能粘贴了整个 JSON)
        real_config = config_dict
        real_type = type.strip().lower()
        
        # 如果 config_dict 里竟然包含 'config' 和 'type' 字段，说明是套娃
        if isinstance(config_dict, dict) and "config" in config_dict and "type" in config_dict:
            print(f"📦 [Debug] 检测到嵌套配置，正在自动提取...")
            real_type = config_dict["type"].strip().lower()
            real_config = config_dict["config"]
            
            # 顺便更新一下名字，如果有的话
            if "name" in config_dict:
                name = config_dict["name"]

        print(f"🧪 [Debug] 修正后的请求 - Name: {name}, Type: '{real_type}'")
        # print(f"📦 [Debug] 修正后的 Config: {real_config}")

        # --- 2. 针对 SSE 类型的检查 ---
        if real_type == "sse":
            if "url" not in real_config:
                return False, "❌ SSE 配置缺失 'url' 字段"
            # 清理可能残留的 stdio 参数
            if "command" in real_config:
                del real_config["command"]

        # --- 3. 针对 Stdio 类型的路径修正 ---
        # 必须使用 copy 防止修改原引用
        test_client_config = real_config.copy()
        
        if real_type == "stdio" and test_client_config.get("command") == "python":
            test_client_config["command"] = sys.executable

        # --- 4. 构建最终 Client 配置 ---
        final_mcp_config = {
            name: {
                "transport": real_type,
                **test_client_config
            }
        }
        
        # 再次强制覆写 transport，确保万无一失
        final_mcp_config[name]["transport"] = real_type

        print(f"🚀 [Debug] 最终 Client 配置: {json.dumps(final_mcp_config, ensure_ascii=False)}")

        try:
            # 增加超时机制 (10秒，因为远程连接可能慢)
            # 注意：这里直接实例化 MultiServerMCPClient
            client = MultiServerMCPClient(final_mcp_config)
            
            # 获取工具列表
            tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
            
            tool_names = [t.name for t in tools]
            msg = f"✅ 连接成功！发现 {len(tools)} 个功能: {', '.join(tool_names[:3])}..."
            print(msg)
            return True, msg

        except asyncio.TimeoutError:
            err = "❌ 连接超时 (10s)。请检查 URL 是否可访问。"
            print(err)
            return False, err
        except Exception as e:
            err = f"❌ 连接错误: {str(e)}"
            print(err)
            return False, err

    # --- 核心功能 6: 生成运行时配置 ---

    def get_active_config(self) -> Dict:
        """
        生成给 Agent 使用的最终配置字典
        每次调用时强制重新读取配置文件，确保 Agent 能拿到最新安装的工具
        """
        # 🚨 关键修复：每次获取配置前，强制从磁盘重载
        self.config = self._load_config()

        final_config = {}
        for name, data in self.config.get("tools", {}).items():
            if data.get("active", True): 
                cfg = data["config"].copy()
                
                # 修正本地 python 路径
                if data["type"] == "stdio" and cfg.get("command") == "python":
                    cfg["command"] = sys.executable
                
                final_config[name] = {
                    "transport": data["type"],
                    **cfg
                }
        return final_config

