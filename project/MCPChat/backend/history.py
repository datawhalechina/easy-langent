# history.py
import os
import json
import time
import uuid
from typing import List, Dict, Optional
from langchain_core.messages import messages_from_dict, messages_to_dict, HumanMessage, AIMessage

HISTORY_DIR = "chat_history"
INDEX_FILE = os.path.join(HISTORY_DIR, "index.json")

# 确保目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 确保索引文件存在
if not os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

class HistoryManager:
    """
    负责管理具体的会话消息以及全局的会话列表索引
    """
    def __init__(self, session_id: str = None):
        # 如果没有提供 session_id，则生成一个新的
        self.session_id = session_id if session_id else str(uuid.uuid4())
        self.file_path = os.path.join(HISTORY_DIR, f"{self.session_id}.json")

    # --- 1. 消息级别的操作 (原有功能增强) ---

    def load_messages(self, limit: int = 50):
        """加载当前会话的消息对象"""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return messages_from_dict(data)[-limit:]
        except Exception:
            return []

    def get_full_history(self) -> List[Dict]:
        """获取当前会话的全量消息"""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_interaction(self, user_query: str, ai_response: str):
        """保存交互并更新索引"""
        # A. 保存消息内容
        current_data = []
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
            except: pass
        
        new_messages = [HumanMessage(content=user_query), AIMessage(content=ai_response)]
        updated_data = current_data + messages_to_dict(new_messages)
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        # B. 更新索引信息 (更新时间戳，如果是新会话则生成标题)
        self._update_index(user_query)

    # --- 2. 会话级别的操作 (新增功能) ---

    @staticmethod
    def get_all_sessions() -> List[Dict]:
        """获取所有会话列表（用于前端侧边栏）"""
        if not os.path.exists(INDEX_FILE):
            return []
        try:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
                # 按 updated_at 倒序排列，最近的在上面
                sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
                return sessions
        except:
            return []

    @staticmethod
    def delete_session(session_id: str):
        """删除指定会话"""
        # 1. 删除具体文件
        file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # 2. 从索引中移除
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            sessions = [s for s in sessions if s["id"] != session_id]
            
            with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)

    @staticmethod
    def rename_session(session_id: str, new_title: str) -> bool:
        """重命名指定会话的标题"""
        if not os.path.exists(INDEX_FILE):
            return False
            
        try:
            # 1. 读取索引
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            # 2. 查找并修改
            found = False
            for session in sessions:
                if session["id"] == session_id:
                    session["title"] = new_title
                    found = True
                    break
            
            # 3. 如果找到了，保存回文件
            if found:
                with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                    json.dump(sessions, f, ensure_ascii=False, indent=2)
                return True
            return False
            
        except Exception as e:
            print(f"Error renaming session: {e}")
            return False

    def _update_index(self, first_query_text: str):
        """更新 index.json，如果会话不存在则创建，并自动生成标题"""
        if not os.path.exists(INDEX_FILE):
            sessions = []
        else:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                sessions = json.load(f)

        # 检查当前 session_id 是否已存在
        session = next((s for s in sessions if s["id"] == self.session_id), None)
        
        current_timestamp = int(time.time())

        if session:
            # 已存在，只更新时间
            session["updated_at"] = current_timestamp
        else:
            # 不存在，创建新条目
            # 简单策略：用用户第一句话的前20个字作为标题
            title = first_query_text[:20] + "..." if len(first_query_text) > 20 else first_query_text
            new_session = {
                "id": self.session_id,
                "title": title,
                "created_at": current_timestamp,
                "updated_at": current_timestamp
            }
            sessions.append(new_session)

        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)