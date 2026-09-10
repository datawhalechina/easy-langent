"""定义游戏状态与基础常量。"""

from typing import Dict, List, Literal, Tuple, TypedDict

AGENTS = ["agent1", "agent2", "agent3", "agent4"]
FALLBACK_WORD_PAIRS: List[Tuple[str, str]] = [
    ("奶茶", "果汁"),
    ("牙刷", "牙膏"),
    ("米饭", "面条"),
    ("手机", "平板"),
    ("篮球", "足球"),
    ("咖啡", "红茶"),
]

Role = Literal["平民", "卧底"]
GameStatus = Literal["running", "end"]
Winner = Literal["", "civilian", "undercover"]


class GameState(TypedDict):
    """保存整局游戏运行中需要共享的状态。"""

    civilian_word: str  # 平民拿到的词语。
    undercover_word: str  # 卧底拿到的词语。
    role_assignment: Dict[str, Tuple[Role, str]]  # 每个玩家的身份与词语。
    speeches: Dict[str, str]  # 当前轮每个存活玩家的发言。
    history_speeches: List[Dict[str, str]]  # 每一轮的发言历史。
    speech_reasoning: Dict[str, str]  # 发言背后的策略说明。
    votes: Dict[str, str]  # 当前轮每个存活玩家投给了谁。
    vote_reasoning: Dict[str, str]  # 投票理由。
    game_status: GameStatus  # 游戏是否结束。
    winner: Winner  # 获胜方。
    eliminated: List[str]  # 已淘汰玩家顺序。
    round: int  # 当前轮次，从 1 开始。


def init_game_state() -> GameState:
    """创建一份干净的初始状态，供图执行入口调用。"""

    return {
        "civilian_word": "",
        "undercover_word": "",
        "role_assignment": {},
        "speeches": {},
        "history_speeches": [],
        "speech_reasoning": {},
        "votes": {},
        "vote_reasoning": {},
        "game_status": "running",
        "winner": "",
        "eliminated": [],
        "round": 1,
    }
