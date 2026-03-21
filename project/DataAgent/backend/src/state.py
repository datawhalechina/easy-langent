from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # 消息列表，add_messages reducer 会自动处理追加逻辑
    messages: Annotated[Sequence[BaseMessage], add_messages]