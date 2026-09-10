import os
import re
import random
from typing import TypedDict, List, Dict, Literal, Optional, Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-v4-pro",
    temperature=0.7,
    timeout=60,        # 单次请求超时（秒），避免网络卡死
    max_retries=3,     # 失败自动重试次数
)

# 定义状态
# 1.不可变配置
class GameConfig(TypedDict):
    """游戏开始时确定的静态配置"""
    total_players: int                              # 玩家人数
    role_distribution: Dict[str,int]                # 角色分配(按人数)，如{"wolf":2,"villager":3,"seer":1,"witch":1,"hunter":1}

# 2.玩家静态身份
class PlayerState(TypedDict):
    """每个玩家的真实身份(仅系统知晓，AI根据角色控制信息可见性)"""
    role: str                                       # 玩家角色
    team: Literal['wolf','good']                    # 所属阵营

    # 永久死因（存活时为 None，死亡时写入，永不修改）
    dead_reason: Optional[Literal['wolf_kill', 'poison', 'vote', 'hunter_shot']]

    # 角色技能使用标记
    witch_antidote_used: bool                       # 女巫解药是否已用（默认：False）
    witch_poison_used: bool                         # 女巫毒药是否已用（默认：False）
    hunter_can_shoot: bool                          # 猎人是否还有能开枪（默认：True）

# 4.实时动态数据
class LiveGameData(TypedDict):
    """每一轮都会频繁变化的数据"""
    round: int                                      # 当前游戏轮数（天数）
    phase: Literal['night','daytime']               # 游戏阶段（天黑/天亮）

    alive_players: List[int]                        # 当前存活玩家ID列表
    dead_players: List[int]                         # 已死亡玩家列表

    # 投票相关
    votes: Dict[int,int]                            # 投票记录：投票人 ID -> 被投票人 ID
    vote_count: Dict[int,int]                       # 票数统计：被投票人ID -> 总票数

    #【临时】当前轮次待处理的死亡（狼杀/毒杀），由结算节点消费后清空
    pending_deaths: Dict[int, str]                  # key: 玩家ID, value: 死因（wolf_kill / poison）

    # ----- 游戏结果 -----
    game_result: Optional[Literal['wolf_win', 'good_win']]  # 游戏结束时写入

# 5.历史记录
class GameHistory(TypedDict):
    """所有已发生的事件日志（供 AI 推理和系统回溯）"""
    speech_log: List[Dict[str, Any]]                # 发言记录 [{"speaker": id, "content": str, "round": int}]
    system_announcements: List[Dict[str, Any]]      # 系统公告 [{"round": int, "phase": str, "content": str}]
    vote_results_log: List[Dict[str, Any]]          # 每轮投票完整记录
                                                    # [{"round": int,
                                                    #   "votes": {voter_id: target_id},
                                                    #   "eliminated": int or None,  # 被淘汰玩家ID，平票时为None
                                                    #   "result": str}]             # 结果描述

# 6.AI玩家专属记忆（彼此隔离）（可选）
class AgentMemory(TypedDict):
    """每个 AI 玩家对游戏的主观认知"""
    # ----- 对他人的推测 -----
    suspects: Dict[int, str]     # 怀疑的玩家：{玩家ID: 怀疑原因}（敌对阵营）
    confirmed: Dict[int, str]    # 确信的玩家：{玩家ID: 确信原因}（同阵营）
    # ----- 预言家专用：真正查验过的玩家（客观事实，与主观怀疑区分）-----
    checked: Dict[int, str]      # 已查验玩家：{玩家ID: "狼人"/"好人"}（非预言家恒为空）
    # ----- 思考记录 -----
    personal_notes: List[str]  # 自己的思考笔记（可用于 LLM 提示增强）

# 7.顶级全局状态
class WolfGameState(TypedDict):
    """LangGraph 使用的唯一顶级状态"""
    config: GameConfig                      # 不可变配置
    players: Dict[int, PlayerState]         # 所有玩家的完整状态（key: 玩家ID）
    live: LiveGameData                      # 实时动态数据
    history: GameHistory                    # 历史记录
    memories: Dict[int, AgentMemory]        # 每个 AI 的专属记忆（key: 玩家ID）

#----------------------------------------------待实现------------------------------------------------
# # 标准狼人杀配置表（屠边规则）
# # key: 总人数, value: 角色配额字典
# STANDARD_CONFIGS = {
#     6: {"werewolf": 2, "villager": 2, "seer": 1, "witch": 1},           # 6人简易局
#     7: {"werewolf": 2, "villager": 3, "seer": 1, "witch": 1},           # 7人局
#     8: {"werewolf": 3, "villager": 2, "seer": 1, "witch": 1, "hunter": 1},  # 8人局
#     9: {"werewolf": 3, "villager": 3, "seer": 1, "witch": 1, "hunter": 1},  # 9人标准局
#     10: {"werewolf": 3, "villager": 3, "seer": 1, "witch": 1, "hunter": 1, "guard": 1},  # 10人局（带守卫）
#     12: {"werewolf": 4, "villager": 4, "seer": 1, "witch": 1, "hunter": 1, "guard": 1},  # 12人标准局
# }
#---------------------------------------------------------------------------------------------------

# 设置初始状态
def create_initial_state(
        total_players: int,
        role_distribution: Dict[str, int],
        player_ids: Optional[List[int]] = None,
) -> WolfGameState:
    """
    创建狼人杀游戏的空白初始状态，不分配角色。
    角色分配由 assign_roles_node 节点完成。

    Args:
        total_players: 总玩家人数
        role_distribution: 角色配额，如 {"werewolf": 2, "villager": 3, "seer": 1, "witch": 1}
        player_ids: 玩家 ID 列表，默认从 1 到 total_players
    """
    if sum(role_distribution.values()) != total_players:
        raise ValueError(f"角色总数({sum(role_distribution.values())})与玩家人数({total_players})不匹配！")

    if player_ids is None:
        player_ids = list(range(1, total_players + 1))
    else:
        if len(player_ids) != total_players:
            raise ValueError(f"玩家列表长度({len(player_ids)})与总人数({total_players})不匹配！")

    # 初始化空白玩家状态（角色留空，由 assign_roles_node 填入）
    players: Dict[int, PlayerState] = {}
    memories: Dict[int, AgentMemory] = {}
    for pid in player_ids:
        players[pid] = {
            "role": "",
            "team": "good",
            "dead_reason": None,
            "witch_antidote_used": False,
            "witch_poison_used": False,
            "hunter_can_shoot": True,
        }
        memories[pid] = {
            "suspects": {},
            "confirmed": {},
            "checked": {},
            "personal_notes": [],
        }

    live: LiveGameData = {
        "round": 0,
        "phase": "night",
        "alive_players": player_ids.copy(),
        "dead_players": [],
        "votes": {},
        "vote_count": {},
        "pending_deaths": {},
        "game_result": None,
    }

    history: GameHistory = {
        "speech_log": [],
        "system_announcements": [],
        "vote_results_log": [],
    }

    config: GameConfig = {
        "total_players": total_players,
        "role_distribution": role_distribution,
    }

    return {
        "config": config,
        "players": players,
        "live": live,
        "history": history,
        "memories": memories,
    }


# ========== 工具函数 ==========
# 打印函数
def print_stage(title: str, char: str = "="):
    """打印节点阶段的醒目分隔标题，便于在终端观察执行到哪个节点"""
    print(f"\n{char * 50} {title} {char * 50}")

# 格式化历史上下文
def format_history_context(history: dict) -> str:
    """格式化完整历史上下文（系统公告 + 投票记录 + 发言记录），供 Prompt 使用"""
    result = ""

    # 系统公告
    if history.get("system_announcements"):
        result += "【系统公告】\n"
        for ann in history["system_announcements"]:
            result += f"  第{ann['round']}轮 {ann['phase']}：{ann['content']}\n"
        result += "\n"

    # 历史投票记录
    if history.get("vote_results_log"):
        result += "【历史投票记录】\n"
        for vote in history["vote_results_log"]:
            eliminated = vote.get("eliminated")
            eliminated_str = f"玩家{eliminated}号被淘汰" if eliminated else "平票无人淘汰"
            result += f"  第{vote['round']}轮投票：{eliminated_str}（{vote.get('result', '')}）\n"
            for voter, target in vote.get("votes", {}).items():
                result += f"    玩家{voter}号 → 玩家{target}号\n"
        result += "\n"

    # 历史发言记录
    if history.get("speech_log"):
        result += "【历史发言记录】\n"
        for log in history["speech_log"]:
            result += f"  第{log['round']}轮 白天 发言人：{log['speaker']}号，发言内容：{log['content']}\n"

    return result

# 格式化玩家记忆上下文
def format_memory_context(mem: dict, current_round: int) -> str:
    """格式化玩家记忆上下文，供 Prompt 使用"""
    result = f"【当前是第{current_round}轮】\n"
    if mem.get("suspects"):
        result += f"我目前怀疑是敌对阵营的玩家：{mem['suspects']}\n"
    if mem.get("confirmed"):
        result += f"我确信是同阵营的玩家：{mem['confirmed']}\n"
    if mem.get("personal_notes"):
        notes_str = "\n  ".join(mem["personal_notes"])
        result += f"我的历史思考笔记：\n  {notes_str}\n"
    return result

# ================================

# 定义节点
# 1.分配角色节点
def assign_roles_node(state: WolfGameState) -> dict:
    """
    角色分配节点（图的入口，只执行一次）：
    从 config 读取角色配额，随机打乱后写入每个玩家的 role/team，
    同时初始化 AI 记忆（狼人互相知晓队友）。
    """
    config = state["config"] # 可从中获取总玩家人数、角色
    players = state["players"]
    player_ids = list(players.keys())

    # 生成角色池并随机打乱
    role_pool = []
    for role, count in config["role_distribution"].items():
        role_pool.extend([role] * count)
    random.shuffle(role_pool)

    # 写入每个玩家的角色和阵营
    new_players = {}
    for pid, role in zip(player_ids, role_pool):
        new_players[pid] = {
            **players[pid], # 解包
            "role": role,
            "team": "wolf" if role == "werewolf" else "good",
        }

    # 初始化 AI 记忆：狼人互相知晓队友
    wolf_ids = [pid for pid, p in new_players.items() if p["role"] == "werewolf"]
    new_memories = {}
    for pid in player_ids:
        mem = dict(state["memories"][pid])
        if new_players[pid]["role"] == "werewolf":
            mem["confirmed"] = {w: "狼人队友" for w in wolf_ids if w != pid}
        new_memories[pid] = mem

    # 写入系统公告（公开角色分布，让所有玩家知道场上有几个狼人/特殊角色）
    role_dist_str = "、".join([f"{role} {count}人" for role, count in config["role_distribution"].items()])
    announcement_content = (
        f"游戏开始，共{config['total_players']}名玩家。"
        f"角色分布：{role_dist_str}。"
        f"胜利条件：狼人全部死亡则好人阵营胜利；所有神职（预言家/女巫/猎人）全部死亡或所有平民全部死亡则狼人阵营胜利。"
    )
    new_history = dict(state["history"])
    new_history["system_announcements"] = list(state["history"]["system_announcements"]) + [{
        "round": 1,
        "phase": "night",
        "content": announcement_content,
    }]

    print_stage("角色分配完成")
    print(f"  {announcement_content}")
    for pid, p in new_players.items():
        print(f"  玩家{pid}：{p['role']}（{p['team']}阵营）")

    return {
        "players": new_players,
        "memories": new_memories,
        "history": new_history,
    }

# 2.夜间狼人行动节点
def night_wolf_node(state: WolfGameState) -> dict:
    """
    夜间：狼人依次选择击杀目标，少数服从多数决定最终击杀对象。
    每个狼人可以选择一个目标、或是跟随上一个狼人的选择，或选择0（不杀人）。
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]
    current_round = live["round"]

    wolf_ids = [pid for pid in alive if players[pid]["role"] == "werewolf"]
    good_ids = [pid for pid in alive if players[pid]["role"] != "werewolf"]

    print_stage("天黑了请闭眼，狼人请睁眼，请选择你们今晚要猎杀的目标：")
    print_stage(f"第{current_round}轮 黑夜 狼人行动阶段")

    # 构建狼人公共上下文
    wolf_context = (
        f"当前存活的狼人玩家：{wolf_ids}\n"
        f"当前存活的非狼人玩家：{good_ids}\n"
    )
    history_context = format_history_context(state["history"])

    # 夜间击杀 Prompt
    night_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的身份是狼人。现在是第{round}轮夜晚，狼人秘密选择击杀目标。

【狼人夜间行动规则】
1. 你可以选择击杀一名存活的非狼人玩家（输出该玩家编号）
2. 你可以选择0，表示本夜不杀人（通常用于掩护自己或队友的身份）
3. 若前一个狼人已经选择了目标，你可以跟随他的选择，也可以独立选择其他目标
4. 最终击杀目标由所有狼人投票，少数服从多数，票数相同时随机选一个
5. 如果为了游戏胜利，需要搅乱局势，可以选择击杀自己的队友（如果不到迫不得已，不得这么做）

【策略建议】
- 优先击杀预言家、女巫等特殊职业（通过发言判断）
- 若队伍身份较安全，可以空刀（选0）迷惑好人
- 尽量和队友保持一致，避免分票
-【高级策略】自刀（击杀自己的狼队友）—— 可以骗取女巫的解药，让狼队友获得"银水"身份，从而在白天博取信任，带队抗推好人。

【游戏上下文】
{wolf_context}
{prev_choice_hint}

【历史记录（发言/投票/公告）】
{history_context}

【基于历史记录的分析要点】
- 发言分析：谁的发言逻辑严密、指向性强（可能是预言家）；谁在暗示怀疑狼人（可能威胁队伍安全）
- 投票分析：谁在投票时针对狼人队友（是潜在威胁）；谁在保护好人（可能是女巫或预言家）
- 优先击杀：对狼人威胁最大的玩家（如上轮投狼人票的、发言中点名怀疑狼人的）

请只输出一个整数：目标玩家编号（存活非狼人玩家编号之一），或 0 表示不杀人。
不要输出任何解释，只输出数字。"""),
        ("user", "你是玩家{pid}，请决定你的击杀目标：")
    ])

    chain = night_prompt | llm | StrOutputParser()

    # 遍历狼人，依次决策
    vote_results: Dict[int, int] = {}  # {狼人pid: 投票目标}
    prev_choice = None

    for i, wolf_pid in enumerate(wolf_ids):
        prev_choice_hint = ""
        if prev_choice is not None:
            prev_choice_hint = f"前一个狼人（玩家{wolf_ids[i-1]}）选择了目标：{prev_choice}（0表示不杀人）"

        raw = chain.invoke({
            "round": current_round,
            "wolf_context": wolf_context,
            "prev_choice_hint": prev_choice_hint,
            "history_context": history_context,
            "pid": wolf_pid,
        }).strip()

        # 解析输出，容错处理
        # 有效目标：0（不杀）或任意存活玩家（含队友，万不得已可杀队友）
        valid_targets = [0] + alive
        try:
            choice = int(raw)
            if choice not in valid_targets:
                print(f"  玩家{wolf_pid} 输出了无效目标 {choice}，自动修正为跟随前狼或随机好人")
                choice = prev_choice if prev_choice is not None else (random.choice(good_ids) if good_ids else 0)
        except ValueError:
            choice = prev_choice if prev_choice is not None else (random.choice(good_ids) if good_ids else 0)

        vote_results[wolf_pid] = choice
        prev_choice = choice
        print(f"  狼人玩家{wolf_pid} 选择：{'不杀人' if choice == 0 else f'击杀玩家{choice}'}")

    # 少数服从多数：统计票数，排除0（弃票）
    kill_votes: Dict[int, int] = {}
    for target in vote_results.values():
        if target != 0:
            kill_votes[target] = kill_votes.get(target, 0) + 1

    if not kill_votes:
        final_target = 0  # 全体选择不杀人
        print("  所有狼人选择不杀人，今夜空刀")
    else:
        max_votes = max(kill_votes.values())
        top_targets = [t for t, v in kill_votes.items() if v == max_votes]
        final_target = random.choice(top_targets)  # 平票时随机
        print(f"  狼人最终决定击杀：玩家{final_target}（得票{max_votes}票）")

    # 写入 pending_deaths：空刀时不写入，有目标时写入待结算死亡
    new_live = dict(live)
    new_pending = dict(live["pending_deaths"])
    if final_target != 0:
        new_pending[final_target] = "wolf_kill"
    new_live["pending_deaths"] = new_pending
    new_history = dict(state["history"])

    # 将击杀目标写入每个狼人的 personal_notes，供白天发言时判断目标是否被救
    new_memories = dict(state["memories"])
    kill_note = f"第{current_round}夜：{'空刀，未击杀任何人' if final_target == 0 else f'击杀玩家{final_target}'}"
    for wolf_pid in wolf_ids:
        mem = dict(new_memories[wolf_pid])
        mem["personal_notes"] = list(mem.get("personal_notes", [])) + [kill_note]
        new_memories[wolf_pid] = mem

    return {
        "live": new_live,
        "memories": new_memories,
        "history": new_history,
    }

# 白天狼人伪装发言
def daytime_wolf_speech(state: WolfGameState, wolf_pid: int, realtime_context: str = "") -> dict:
    """
    白天狼人发言辅助函数：生成伪装发言内容，同时更新该狼人的 AgentMemory。
    返回 {"speech": str, "updated_memory": AgentMemory}
    由 speech_node 统一调用，不单独作为 LangGraph 节点。

    辅助函数没有能力直接修改全局状态，它只能返回"这个玩家的记忆更新后是什么"
    由 speech_node 统一把每个玩家的新记忆合并回去：
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]
    current_round = live["round"]
    mem = state["memories"][wolf_pid]

    wolf_ids = [pid for pid in alive if players[pid]["role"] == "werewolf"]
    good_ids = [pid for pid in alive if players[pid]["role"] != "werewolf"]

    # 从 personal_notes 找上一夜的行动记录，判断击杀目标是否被救
    # 只要匹配到上一夜的记录（无论击杀还是空刀）就停止，避免误用更早轮次的击杀
    last_kill_target = None
    for note in reversed(mem.get("personal_notes", [])):
        if f"第{current_round - 1}夜" in note:
            match = re.search(r"击杀玩家(\d+)", note)
            if match:
                last_kill_target = int(match.group(1))
            break

    kill_hint = ""
    if last_kill_target is not None:
        if last_kill_target in alive:
            kill_hint = (
                f"【重要】昨晚（第{current_round - 1}夜）狼人击杀了玩家{last_kill_target}，"
                f"但他今天仍然存活。推断：1.女巫使用解药救了他；2.他本人就是女巫（自救）。"
                f"该玩家是高度威胁目标，应列入重点怀疑名单，并在白天引导其他人怀疑他，但是不能公布自己狼人的身份，不能告知其他玩家昨天狼人的目标是该玩家。"
            )
            # 直接写入记忆：加入怀疑列表
            new_suspects = dict(mem.get("suspects", {}))
            new_suspects[last_kill_target] = f"第{current_round - 1}夜被击杀但存活，高度怀疑是女巫"
            mem = {**mem, "suspects": new_suspects}
        else:
            kill_hint = f"昨晚（第{current_round - 1}夜）击杀玩家{last_kill_target}成功，目标已死亡。"

    # 获取历史(不含本轮)
    history_context = format_history_context(state["history"])
    # 获取玩家记忆上下文
    memory_context = format_memory_context(mem, current_round)

    daytime_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的真实身份是狼人，但你必须隐瞒身份。
【你的信息】
- 你是玩家{pid}，真实身份：狼人
- 存活的狼人玩家：{wolf_ids}
- 存活的非狼人玩家：{good_ids}

【白天伪装规则】
1. 绝对不能主动透露自己是狼人
2. 可以伪装成村民或其他好人职业（预言家、女巫、猎人等）
3. 主动引导怀疑方向，把锅甩给好人玩家，制造混乱
4. 观察场上发言，找出可能是预言家/女巫/猎人的玩家，暗示其他人怀疑他们
5. 若自己身份被当众揭穿，可选择"反将一军"——指认某个好人玩家为狼人，搅乱局势保护队友

【紧急情况：身份被揭穿时】
- 可以声称对方才是狼人，并给出"理由"
- 转移话题，攻击揭穿你的人

【昨夜击杀情况】
{kill_hint}

【你的当前记忆与推测】
{memory_context}

【历史记录】
{history_context}

【本轮已发言内容】（⚠️只有下面列出的玩家在本轮已经发言，你只能针对这些已发言的内容和历史记录进行分析，绝对不能提及、评价或假设任何尚未发言玩家的观点）
{realtime_context}

【基于历史记录和本轮已发言内容的伪装策略】
- 发言分析：找出发言逻辑严密的好人（可能是预言家/女巫），在发言中暗示他们可疑，引导其他人怀疑
- 投票分析：找出上轮投了狼人队友的玩家，本轮重点反咬，转移矛头
- 伪装原则：顺着场上主流观点发言，不要太突出，避免成为焦点

请严格按以下JSON格式输出，不要输出任何额外内容：
{{
  "speech": "你的白天发言内容（50-150字）",
  "role_claim": "本次对外声称的身份（如村民/预言家/猎人等，若本次不声称身份则为null）",
  "suspects": {{玩家ID: "怀疑原因(30字以内)", ...}},
  "confirmed": {{玩家ID: "确信原因(30字以内)", ...}},
  "note": "本轮思考笔记（不超过100字）"
}}"""),
        ("user", "现在是第{round}轮 白天发言阶段，请发言：")
    ])

    chain = daytime_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "pid": wolf_pid,
            "wolf_ids": wolf_ids,
            "good_ids": good_ids,
            "kill_hint": kill_hint or "第一夜尚未行动，或昨夜是空刀。",
            "memory_context": memory_context,
            "history_context": history_context,
            "realtime_context": realtime_context,
            "round": current_round,
        })
        speech = result.get("speech", "").strip()
        suspects = result.get("suspects", mem.get("suspects", {}))
        confirmed = result.get("confirmed", mem.get("confirmed", {}))
        note = result.get("note", "")

        # role_claim 写进 personal_notes 而非单独字段
        role_claim_output = result.get("role_claim")
        if role_claim_output and role_claim_output != "null":
            note = f"第{current_round}轮 白天 声称身份为【{role_claim_output}】，这个身份最好不要变，如果变了容易引起其他玩家的怀疑。" + (f" {note}" if note else "")
    except Exception:
        speech = "我先不表达意见，我先观望一下场上的局势。"
        suspects = mem.get("suspects", {})
        confirmed = mem.get("confirmed", {})
        note = ""

    # 更新 memory
    updated_memory = {
        **mem,
        "suspects": suspects,
        "confirmed": confirmed,
        "personal_notes": list(mem.get("personal_notes", [])) + ([note] if note else []),
    }

    return {
        "speech": speech,
        "updated_memory": updated_memory,
    }

# 3.女巫行动节点
def night_witch_node(state: WolfGameState) -> dict:
    """
    夜间：女巫行动。
    - 被告知狼人击杀目标，决定是否使用解药（救人）
    - 若不用解药，可选择使用毒药（毒杀任意存活玩家）
    - 解药和毒药不能同一晚使用，且各只有一瓶
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]
    current_round = live["round"]
    pending_deaths = dict(live["pending_deaths"])

    print_stage("女巫请睁眼")

    # 找到女巫玩家
    witch_pid = next((pid for pid in alive if players[pid]["role"] == "witch"), None)
    if witch_pid is None:
        return {}  # 女巫已死，跳过

    witch_state = players[witch_pid]
    antidote_used = witch_state["witch_antidote_used"]          # 解药是否使用
    poison_used = witch_state["witch_poison_used"]              # 毒药是否使用

    # 若解药和毒药都已用尽，直接跳过
    if antidote_used and poison_used:
        print_stage(f"第{current_round}轮 黑夜 女巫行动阶段")
        print("  女巫的解药和毒药均已用尽，无法行动。")
        return {}

    # 从 pending_deaths 读取狼人击杀目标
    wolf_kill_target = next(
        (pid for pid, cause in pending_deaths.items() if cause == "wolf_kill"), 0
    )

    # 获取历史信息
    history_context = format_history_context(state["history"])

    print_stage(f"第{current_round}轮 黑夜 女巫行动阶段")
    # ===== 第一步：询问解药 =====
    use_antidote = False
    mem = state["memories"][witch_pid]
    if not antidote_used and wolf_kill_target != 0:
        print(f"  【法官】今晚狼人击杀了玩家{wolf_kill_target}，女巫是否使用解药？")
        antidote_prompt = ChatPromptTemplate.from_messages([
            ("system", """你正在参与狼人杀游戏，你的身份是女巫。现在是第{round}轮 夜晚。
【法官通知】今晚狼人击杀了玩家{kill_target}。

【你的记忆】
- 你确信是好人的玩家：{confirmed}
- 你怀疑是狼人的玩家：{suspects}

【历史记录（发言/投票/公告）】
{history_context}

【基于历史记录分析被击杀者的价值】
- 发言分析：被击杀玩家存活时发言是否逻辑严密、有主动引导局势（可能是预言家）
- 投票分析：被击杀玩家存活时的投票是否准确指向狼人（说明他判断力强，是关键好人）
- 救人原则：若被击杀者是发言或投票中表现突出的好人，应优先使用解药

【最高优先规则】如果今晚被击杀的目标（玩家{kill_target}）就是你自己（你是玩家{pid}），你必须使用解药自救，直接输出 true，不要犹豫。

是否使用解药，回答严格输出True或False。
"""),
            ("user", "你是玩家{pid}（女巫），是否使用解药救玩家{kill_target}？请严格输出 true 或 false：")
        ])
        try:
            r1 = (antidote_prompt | llm | StrOutputParser()).invoke({
                "round": current_round,
                "kill_target": wolf_kill_target,
                "confirmed": mem.get("confirmed", {}),
                "suspects": mem.get("suspects", {}),
                "history_context": history_context,
                "pid": witch_pid,
            })
            raw = r1.strip().lower()
            use_antidote = raw == "true"

            # 代码层强制兜底：狼人击杀目标是女巫自己时，必须自救
            if wolf_kill_target == witch_pid and not use_antidote:
                use_antidote = True
                print("  女巫是本夜击杀目标，强制使用解药自救")

            # 将决策理由记入女巫的思考笔记
            note = f"第{current_round}夜：狼人击杀玩家{wolf_kill_target}，{'使用解药救人' if use_antidote else '选择不使用解药'}\n"
            new_memories = dict(state["memories"])
            new_witch_mem = dict(mem)
            new_witch_mem["personal_notes"] = list(mem.get("personal_notes", [])) + [note]
            new_memories[witch_pid] = new_witch_mem
            print(f"  女巫决定：{'使用解药' if use_antidote else '不使用解药'}")
        except Exception:
            use_antidote = False
            new_memories = state["memories"]
            print("  解析失败，女巫选择不使用解药")
    elif wolf_kill_target == 0:
        print("  【法官】今晚是平安夜，狼人未击杀任何人，无法使用解药。")
        new_memories = dict(state["memories"])
        new_witch_mem = dict(mem)
        new_witch_mem["personal_notes"] = list(mem.get("personal_notes", [])) + [
            f"第{current_round}夜：平安夜，狼人未击杀任何人，可能是空刀保护队友或迷惑好人"
        ]
        new_memories[witch_pid] = new_witch_mem
    else:
        new_memories = state["memories"]
        print("  【法官】女巫解药已使用，跳过解药阶段。")

    # ===== 第二步：询问毒药（只有本晚未使用解药时才能使用）=====
    use_poison = False
    poison_target = 0
    if not poison_used and not use_antidote:
        print(f"  【法官】女巫是否使用毒药？存活玩家：{alive}")
        poison_prompt = ChatPromptTemplate.from_messages([
            ("system", """你正在参与狼人杀游戏，你的身份是女巫。现在是第{round}轮 夜晚。

【你的记忆】
- 你确信是好人的玩家：{confirmed}
- 你怀疑是狼人的玩家：{suspects}
- 存活玩家列表：{alive}

【历史记录（发言/投票/公告）】
{history_context}

【基于历史记录分析毒药目标】
- 发言分析：谁的发言前后矛盾、逻辑混乱（可能是狼人在伪装）；谁在为狼人辩护（可能是狼人队友）
- 投票分析：谁在投票时明显在保护狼人、投票方向和大多数好人相反（可能是狼人）
- 毒人原则：高度确信是狼人再使用，宁可不用也不要误杀好人

先回答是否使用毒药（true/false），若使用则再回答目标玩家编号。
请严格按以下JSON格式输出，不需要理由：
{{"use_poison": true或false, "poison_target": 目标玩家ID（整数，如果使用毒药，则需要选择仅一名存活玩家）}}"""),
            ("user", "你是玩家{pid}（女巫），是否使用毒药？请只输出JSON：")
        ])
        try:
            r2 = (poison_prompt | llm | JsonOutputParser()).invoke({
                "round": current_round,
                "confirmed": mem.get("confirmed", {}),
                "suspects": mem.get("suspects", {}),
                "alive": alive,
                "history_context": history_context,
                "pid": witch_pid,
            })
            use_poison = bool(r2.get("use_poison", False))
            poison_target = int(r2.get("poison_target", 0))

            # 使用毒药但是是无效目标
            if use_poison and poison_target not in alive:
                use_poison = False
                poison_target = 0
                print("  毒药目标无效，已取消")

            # 将决策记入女巫思考笔记
            note = f"第{current_round}夜：{'使用毒药毒杀玩家' + str(poison_target) if use_poison else '选择不使用毒药'}"
            new_witch_mem = dict(new_memories[witch_pid])
            new_witch_mem["personal_notes"] = list(new_witch_mem.get("personal_notes", [])) + [note]
            new_memories = dict(new_memories)
            new_memories[witch_pid] = new_witch_mem
            print(f"  女巫决定：{'使用毒药毒杀玩家' + str(poison_target) if use_poison else '不使用毒药'}")
        except Exception:
            use_poison = False
            poison_target = 0
            print("  解析失败，女巫选择不使用毒药")
    elif use_antidote:
        print("  【法官】本晚已使用解药，不能再使用毒药。")
    else:
        print("  【法官】毒药已使用，跳过毒药阶段。")

    # 更新 pending_deaths
    new_pending = dict(pending_deaths)
    if use_antidote and wolf_kill_target != 0:
        new_pending.pop(wolf_kill_target, None)
        print(f"  女巫使用解药救活了玩家{wolf_kill_target}")
    if use_poison and poison_target != 0:
        new_pending[poison_target] = "poison"
        print(f"  女巫使用毒药毒杀了玩家{poison_target}")

    new_live = dict(live)
    new_live["pending_deaths"] = new_pending

    # 更新女巫技能使用标记
    new_players = dict(players)
    new_players[witch_pid] = {
        **players[witch_pid],
        "witch_antidote_used": antidote_used or use_antidote,
        "witch_poison_used": poison_used or use_poison,
    }

    new_history = dict(state["history"])

    return {
        "live": new_live,
        "players": new_players,
        "memories": new_memories,
        "history": new_history,
    }

# 白天女巫发言辅助函数
def daytime_witch_speech(state: WolfGameState, witch_pid: int, realtime_context: str = "") -> dict:
    """
    白天女巫发言辅助函数：基于女巫掌握的私密信息（死亡情况、技能使用）生成发言。
    目标是隐藏女巫身份，同时利用信息优势引导好人找出狼人。
    由 speech_node 统一调用，不单独作为 LangGraph 节点。
    返回 {"speech": str, "updated_memory": AgentMemory}
    """
    current_round = state["live"]["round"]
    mem = state["memories"][witch_pid]

    history_context = format_history_context(state["history"])
    memory_context = format_memory_context(mem, current_round)

    daytime_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的真实身份是女巫，但白天不能主动暴露自己是女巫。

【女巫白天发言策略】
1. 隐藏身份：不主动透露自己是女巫，可以伪装成村民
2. 利用信息优势：你的思考笔记中记录了每晚的行动和推断，可以据此分析狼人
3. 引导好人：通过发言帮助好人识别狼人，但不能直接说"我用了解药/毒药"
4. 若身份被逼问：可声称自己是村民，必要时可以承认是女巫来拯救局势

【你的推理记忆】
{memory_context}

【历史发言记录】
{history_context}

【本轮已发言内容】（⚠️只有下面列出的玩家在本轮已经发言，你只能针对这些已发言的内容和历史记录进行分析，绝对不能提及、评价或假设任何尚未发言玩家的观点）
{realtime_context}

【基于历史记录和本轮已发言内容的分析要点】
- 发言分析：结合你掌握的死亡信息，判断谁的发言在刻意误导方向（可能是狼人）；谁的发言和死亡规律吻合
- 投票分析：谁在上轮投票中保护了后来证明是坏人的玩家（可能是狼人）；谁的投票一直准确（好人）
- 信息利用：你知道每晚的死亡情况，可以间接引导好人推断狼人，但不能直接暴露女巫身份

请严格按以下JSON格式输出：
{{
  "speech": "白天发言内容（50-150字）",
  "role_claim": "本次对外声称的身份（如村民/猎人等，若本次不声称则为null）",
  "suspects": {{玩家ID: "怀疑原因(30字以内)", ...}},
  "confirmed": {{玩家ID: "确信原因(30字以内)", ...}},
  "note": "本轮思考笔记（不超过100字）"
}}"""),
        ("user", "你是玩家{pid}（女巫），现在是第{round}轮 白天发言阶段，请发言：")
    ])

    chain = daytime_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "pid": witch_pid,
            "round": current_round,
            "memory_context": memory_context,
            "history_context": history_context,
            "realtime_context": realtime_context,
        })
        speech = result.get("speech", "").strip()
        suspects = result.get("suspects", mem.get("suspects", {}))
        confirmed = result.get("confirmed", mem.get("confirmed", {}))
        note = result.get("note", "")

        role_claim_output = result.get("role_claim")
        if role_claim_output and role_claim_output != "null":
            note = f"第{current_round}天声称身份为【{role_claim_output}】尽量别随便变更身份，避免引起其他玩家的怀疑。。" + (f" {note}" if note else "")
    except Exception:
        speech = "我觉得昨晚的死亡有些蹊跷，大家要仔细分析。"
        suspects = mem.get("suspects", {})
        confirmed = mem.get("confirmed", {})
        note = ""

    updated_memory = {
        **mem,
        "suspects": suspects,
        "confirmed": confirmed,
        "personal_notes": list(mem.get("personal_notes", [])) + ([note] if note else []),
    }

    return {
        "speech": speech,
        "updated_memory": updated_memory,
    }

# 4.夜间预言家行动节点
def night_seer_node(state: WolfGameState) -> dict:
    """
    夜间：预言家查验一名存活玩家的真实身份。
    查验结果只有预言家自己知道，写入其 confirmed 或 suspects 记忆。
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]
    current_round = live["round"]

    seer_pid = next((pid for pid in alive if players[pid]["role"] == "seer"), None)
    if seer_pid is None:
        return {}  # 预言家已死，跳过

    mem = state["memories"][seer_pid]

    # 已查验过的玩家（只看 checked，不受白天发言写入的主观 suspects/confirmed 影响）
    already_checked = set(mem.get("checked", {}).keys())
    candidates = [pid for pid in alive if pid != seer_pid and pid not in already_checked]

    print_stage("预言家请睁眼。请选择你要查验的目标玩家ID:")

    if not candidates:
        print_stage(f"第{current_round}轮 黑夜 预言家行动阶段")
        print("  所有存活玩家均已查验，预言家无需行动。")
        return {}

    print_stage(f"第{current_round}轮 黑夜 预言家行动阶段")
    print(f"  【法官】预言家请睁眼，你可以查验一名玩家的身份。")
    print(f"  可查验的玩家：{candidates}")

    memory_context = format_memory_context(mem, current_round)
    history_context = format_history_context(state["history"])

    seer_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的身份是预言家。现在是第{round}轮 夜晚，你可以查验一名玩家的真实阵营。

【查验规则】
- 每晚只能查验一名玩家
- 已查验过的玩家无需重复查验
- 查验结果只有你自己知道

【你的记忆】
{memory_context}

可查验的玩家列表：{candidates}

【历史记录（发言/投票/公告）】
{history_context}

【基于历史记录分析查验目标】
- 发言分析：谁的发言含糊、逻辑前后矛盾（可能是狼人）；谁在刻意回避关键问题（可能在隐藏身份）
- 投票分析：谁的投票方向和好人明显不同（可能是狼人）；谁在关键轮次投了关键好人（可能是狼人帮凶）
- 查验原则：优先查验最可疑的玩家，查验结果用于白天引导好人投票

请只输出一个整数：你要查验的目标玩家ID。不要输出任何解释。"""),
        ("user", "你是玩家{pid}（预言家），请选择今晚要查验的玩家：")
    ])

    chain = seer_prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "round": current_round,
            "memory_context": memory_context,
            "candidates": candidates,
            "history_context": history_context,
            "pid": seer_pid,
        }).strip()
        check_target = int(raw)
        if check_target not in candidates:
            check_target = random.choice(candidates)
            print(f"  目标无效，随机选择玩家{check_target}")
    except (ValueError, Exception):
        check_target = random.choice(candidates)
        print(f"  解析失败，随机选择玩家{check_target}")

    # 获取目标真实身份
    target_role = players[check_target]["role"]
    is_wolf = target_role == "werewolf"
    print(f"  预言家查验了玩家{check_target}，结果：{'狼人' if is_wolf else '好人'}")

    # 写入预言家记忆
    new_memories = dict(state["memories"])
    new_mem = dict(mem)
    note = f"第{current_round}夜：查验玩家{check_target}，结果为{'狼人' if is_wolf else '好人'}（{target_role}）"

    # checked：客观查验记录，作为夜间去重的唯一依据
    new_checked = dict(new_mem.get("checked", {}))
    new_checked[check_target] = "狼人" if is_wolf else "好人"
    new_mem["checked"] = new_checked

    # suspects/confirmed：供白天发言引导使用
    if is_wolf:
        new_suspects = dict(new_mem.get("suspects", {}))
        new_suspects[check_target] = f"预言家查验确认是狼人（第{current_round}轮 夜晚）"
        new_mem["suspects"] = new_suspects
    else:
        new_confirmed = dict(new_mem.get("confirmed", {}))
        new_confirmed[check_target] = f"预言家查验确认是好人（第{current_round}轮 夜晚）"
        new_mem["confirmed"] = new_confirmed

    new_mem["personal_notes"] = list(new_mem.get("personal_notes", [])) + [note]
    new_memories[seer_pid] = new_mem

    return {
        "memories": new_memories,
    }

# 白天预言家发言辅助函数
def daytime_seer_speech(state: WolfGameState, seer_pid: int, realtime_context: str = "") -> dict:
    """
    白天预言家发言辅助函数：基于查验结果决策是否公开身份，引导好人找出狼人。
    由 speech_node 统一调用，不单独作为 LangGraph 节点。
    返回 {"speech": str, "updated_memory": AgentMemory}
    """
    current_round = state["live"]["round"]
    mem = state["memories"][seer_pid]

    history_context = format_history_context(state["history"])
    memory_context = format_memory_context(mem, current_round)

    daytime_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的真实身份是预言家。

【预言家白天发言策略】
1. 默认隐藏身份：以平民口吻发言，用逻辑和分析说服其他玩家，不直接暴露自己是预言家
2. 利用查验信息引导：你通过查验知道部分玩家的真实身份，可以用"我感觉""我分析""从他的发言来看"等措辞，间接引导好人朝正确方向投票，而不说"我查验了他"
3. 建立说服力：给出有逻辑依据的分析，让其他玩家自愿跟从，而不是依赖身份权威
4. 公开身份的时机（非必要不公开，但必要时果断公开）：
   - 有你所怀疑的敌对玩家冒充预言家跳出来对跳时
   - 好人即将误投好人，必须出手纠正时
   - 场上局势对好人极为不利，公开能扭转局势时
   - 已查验到狼人且需要引导投票时，可公开身份直接报出查验结果揭穿狼人
   - 女巫/猎人等关键好人身份即将被揭穿或被误投时，站出来为其背书保护

【你的查验记忆与推断】
{memory_context}

【历史发言记录】
{history_context}

【本轮已发言内容】（⚠️只有下面列出的玩家在本轮已经发言，你只能针对这些已发言的内容和历史记录进行分析，绝对不能提及、评价或假设任何尚未发言玩家的观点）
{realtime_context}

【基于历史记录和本轮已发言内容的分析要点】
- 发言分析：结合查验结果，判断谁在为狼人辩护或转移视线；用"他的发言逻辑有问题"等措辞引导，不说"我查了他是狼人"
- 投票分析：谁的投票方向一直在保护狼人（可能是狼人）；谁的判断和你一致（可能是可信好人）
- 说服策略：给出具体理由，让其他玩家觉得你的分析有道理，而不是盲目跟从

请严格按以下JSON格式输出：
{{
  "speech": "白天发言内容（50-150字）",
  "role_claim": "本次对外声称的身份（如村民/预言家等，若本次不声称则为null）",
  "suspects": {{玩家ID: "怀疑原因(30字以内)", ...}},
  "confirmed": {{玩家ID: "确信原因(30字以内)", ...}},
  "note": "本轮思考笔记（不超过100字）"
}}"""),
        ("user", "你是玩家{pid}（预言家），现在是第{round}天发言阶段，请发言：")
    ])

    chain = daytime_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "pid": seer_pid,
            "round": current_round,
            "memory_context": memory_context,
            "history_context": history_context,
            "realtime_context": realtime_context,
        })
        speech = result.get("speech", "").strip()
        suspects = result.get("suspects", mem.get("suspects", {}))
        confirmed = result.get("confirmed", mem.get("confirmed", {}))
        note = result.get("note", "")

        role_claim_output = result.get("role_claim")
        if role_claim_output and role_claim_output != "null":
            note = f"第{current_round}轮 白天声称身份为【{role_claim_output}】尽量别随便变更身份，避免引起其他玩家的怀疑。" + (f" {note}" if note else "")
            if role_claim_output == "预言家":
                print(f"  预言家玩家{seer_pid} 选择公开身份")
    except Exception:
        speech = "我仔细观察了大家的发言，感觉有人在说谎，大家要注意。"
        suspects = mem.get("suspects", {})
        confirmed = mem.get("confirmed", {})
        note = ""

    updated_memory = {
        **mem,
        "suspects": suspects,
        "confirmed": confirmed,
        "personal_notes": list(mem.get("personal_notes", [])) + ([note] if note else []),
    }

    return {
        "speech": speech,
        "updated_memory": updated_memory,
    }

# 白天猎人发言辅助函数
def daytime_hunter_speech(state: WolfGameState, hunter_pid: int, realtime_context: str = "") -> dict:
    """
    白天猎人发言辅助函数。
    猎人身份特殊：通常隐藏身份，必要时可公开以震慑狼人或保护队友。
    由 speech_node 统一调用，不单独作为 LangGraph 节点。
    """
    current_round = state["live"]["round"]
    mem = state["memories"][hunter_pid]

    history_context = format_history_context(state["history"])
    memory_context = format_memory_context(mem, current_round)

    daytime_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的真实身份是猎人，但白天通常不主动暴露。

【猎人白天发言策略】
1. 默认隐藏身份：尽量以村民口吻发言，避免成为狼人夜间击杀的目标
2. 技能威慑：若局势需要，可声称自己是猎人——狼人若击杀你，你会开枪带走一人，形成威慑
3. 保护关键好人：若预言家/女巫身份即将暴露，可声称自己是猎人转移狼人注意力
4. 投票引导：基于发言和投票记录，引导大家投出最可疑的狼人

【你的记忆与推断】
{memory_context}

【历史记录（发言/投票/公告）】
{history_context}

【本轮已发言内容】（⚠️只有下面列出的玩家在本轮已经发言，你只能针对这些已发言的内容和历史记录进行分析，绝对不能提及、评价或假设任何尚未发言玩家的观点）
{realtime_context}

【基于历史记录和本轮已发言内容的分析要点】
- 发言分析：谁的发言前后矛盾、逻辑混乱（可能是狼人）
- 投票分析：谁的投票方向一直在保护狼人（可能是狼人同伙）
- 威慑时机：若自己即将被投票淘汰，可亮明猎人身份威慑狼人阵营

请严格按以下JSON格式输出：
{{
  "speech": "白天发言内容（50-150字）",
  "role_claim": "本次对外声称的身份（如村民/猎人等，若本次不声称则为null）",
  "suspects": {{玩家ID: "怀疑原因(30字以内)", ...}},
  "confirmed": {{玩家ID: "确信原因(30字以内)", ...}},
  "note": "本轮思考笔记（不超过100字）"
}}"""),
        ("user", "你是玩家{pid}（猎人），现在是第{round}轮 白天发言阶段，请发言：")
    ])

    chain = daytime_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "pid": hunter_pid,
            "round": current_round,
            "memory_context": memory_context,
            "history_context": history_context,
            "realtime_context": realtime_context, # 本轮已发言内容
        })
        speech = result.get("speech", "").strip()
        suspects = result.get("suspects", mem.get("suspects", {}))
        confirmed = result.get("confirmed", mem.get("confirmed", {}))
        note = result.get("note", "")

        role_claim_output = result.get("role_claim")
        if role_claim_output and role_claim_output != "null":
            note = f"第{current_round}天声称身份为【{role_claim_output}】。" + (f" {note}" if note else "")
    except Exception:
        speech = "我觉得场上有人在说谎，大家要仔细分析。"
        suspects = mem.get("suspects", {})
        confirmed = mem.get("confirmed", {})
        note = ""

    updated_memory = {
        **mem,
        "suspects": suspects,
        "confirmed": confirmed,
        "personal_notes": list(mem.get("personal_notes", [])) + ([note] if note else []),
    }

    return {
        "speech": speech,
        "updated_memory": updated_memory,
    }

# 白天村民发言辅助函数
def daytime_villager_speech(state: WolfGameState, villager_pid: int, realtime_context: str = "") -> dict:
    """
    白天村民发言辅助函数。
    村民没有特殊技能，完全依赖逻辑分析和观察来找出狼人。
    由 speech_node 统一调用，不单独作为 LangGraph 节点。
    """
    current_round = state["live"]["round"]
    mem = state["memories"][villager_pid]

    history_context = format_history_context(state["history"])
    memory_context = format_memory_context(mem, current_round)

    daytime_prompt = ChatPromptTemplate.from_messages([
        ("system", """你正在参与狼人杀游戏，你的身份是村民，没有任何特殊技能。

【村民白天发言策略】
1. 逻辑分析：仔细观察每个人的发言，找出前后矛盾、逻辑混乱的玩家
2. 投票走向：分析历史投票，找出投票方向异常的玩家（可能是狼人）
3. 引导投票：用有说服力的逻辑，引导其他好人把票投给最可疑的玩家
4. 可以声称任意身份：如果认为声称某个特殊身份有助于局势，可以这么做（风险自担）

【你的记忆与推断】
{memory_context}

【历史记录（发言/投票/公告）】
{history_context}

【本轮已发言内容】（⚠️只有下面列出的玩家在本轮已经发言，你只能针对这些已发言的内容和历史记录进行分析，绝对不能提及、评价或假设任何尚未发言玩家的观点）
{realtime_context}

【基于历史记录和本轮已发言内容的分析要点】
- 发言分析：谁的发言含糊、回避关键问题、或刻意引导怀疑无辜玩家
- 投票分析：谁在关键轮次投了好人、谁的投票和狼人方向一致
- 死亡规律：结合历史死亡情况，判断狼人倾向于击杀哪类玩家

请严格按以下JSON格式输出：
{{
  "speech": "白天发言内容（50-150字）",
  "role_claim": "本次对外声称的身份（如村民/预言家/猎人等，若本次不声称则为null）",
  "suspects": {{玩家ID: "怀疑原因(30字以内)", ...}},
  "confirmed": {{玩家ID: "确信原因(30字以内)", ...}},
  "note": "本轮思考笔记（不超过100字）"
}}"""),
        ("user", "你是玩家{pid}（村民），现在是第{round}轮 白天发言阶段，请发言：")
    ])

    chain = daytime_prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "pid": villager_pid,
            "round": current_round,
            "memory_context": memory_context,
            "history_context": history_context,
            "realtime_context": realtime_context,
        })
        speech = result.get("speech", "").strip()
        suspects = result.get("suspects", mem.get("suspects", {}))
        confirmed = result.get("confirmed", mem.get("confirmed", {}))
        note = result.get("note", "")

        role_claim_output = result.get("role_claim")
        if role_claim_output and role_claim_output != "null":
            note = f"第{current_round}轮 白天声称身份为【{role_claim_output}】。" + (f" {note}" if note else "")
    except Exception:
        speech = "我觉得大家要仔细分析每个人的发言，找出逻辑有问题的人。"
        suspects = mem.get("suspects", {})
        confirmed = mem.get("confirmed", {})
        note = ""

    updated_memory = {
        **mem,
        "suspects": suspects,
        "confirmed": confirmed,
        "personal_notes": list(mem.get("personal_notes", [])) + ([note] if note else []),
    }

    return {
        "speech": speech,
        "updated_memory": updated_memory,
    }

# 5.核心：阶段模块（所有死亡都经过这里）
def resolve_deaths_node(state: WolfGameState) -> dict:
    """
    结算节点：将 pending_deaths 中待处理的玩家正式死亡，处理猎人开枪连锁反应。
    - 毒药死亡的猎人不能开枪
    - 狼人击杀/投票淘汰的猎人可以选择开枪
    """
    players = state["players"]
    live = state["live"]
    alive = list(live["alive_players"])
    current_round = live["round"]
    phase = live["phase"]
    # 获得待死亡玩家列表
    pending_deaths = dict(live["pending_deaths"])

    new_players = dict(players)
    dead_this_round = []

    # 结算所有待处理死亡
    for pid, cause in pending_deaths.items():
        if pid in alive:
            alive.remove(pid)
            dead_this_round.append(pid)
            new_players[pid] = {**players[pid], "dead_reason": cause}
            print_stage(f"第{current_round}轮 黑夜，玩家{pid}({players[pid]['role']})死亡，死因：{cause}")

    # 写入系统公告
    new_history = dict(state["history"])
    if dead_this_round:
        content = f"{'昨晚' if phase == 'night' else '本轮投票后'}死亡玩家：{dead_this_round}"
    else:
        content = "昨晚是平安夜，无人死亡。" if phase == "night" else "本次投票结果：平票，无人被淘汰。"
    new_history["system_announcements"] = list(state["history"]["system_announcements"]) + [{
        "round": current_round,
        "phase": phase,
        "content": content,
    }]

    # 猎人开枪：被狼杀或被投票淘汰时可开枪，被毒药毒死时不能开枪
    shoot_target = None
    for pid in dead_this_round:
        if players[pid]["role"] == "hunter" and players[pid]["hunter_can_shoot"]:
            cause = pending_deaths[pid]
            if cause == "poison":
                print(f"  猎人玩家{pid} 被毒药毒死，无法开枪。")
                continue

            print(f"  猎人玩家{pid} 死亡（{cause}），是否选择开枪？")
            mem = state["memories"][pid]
            hunter_prompt = ChatPromptTemplate.from_messages([
                ("system", """你正在参与狼人杀游戏，你的身份是猎人，你刚刚死亡。

【猎人开枪规则】
- 被狼人击杀或被投票淘汰时，可以选择开枪带走一名存活玩家（一般选择你高度怀疑的狼人玩家）
- 被女巫毒药毒死时，不能开枪
- 可以选择不开枪（输出0），因为随便开枪可能会伤及队友

【你的记忆与推断】
{memory_context}

【历史记录】
{history_context}

当前存活玩家：{alive}

【开枪策略】
- 优先射杀你最确信是狼人的玩家
- 若不确定，可选择不开枪（输出0），避免误杀好人

请只输出一个整数：目标玩家ID（存活玩家之一），或 0 表示不开枪。不要输出任何解释。"""),
                ("user", "你是玩家{pid}（猎人），是否开枪？请只输出数字：")
            ])

            chain = hunter_prompt | llm | StrOutputParser()
            history_context = format_history_context(new_history)
            memory_context = format_memory_context(mem, current_round)

            # 可开枪目标：存活玩家中排除猎人自己（防止误伤自己）
            shoot_candidates = [p for p in alive if p != pid]

            try:
                raw = chain.invoke({
                    "pid": pid,
                    "memory_context": memory_context,
                    "history_context": history_context,
                    "alive": shoot_candidates,
                }).strip()

                shoot_target = int(raw)
                # 有效目标：0（不开枪）或候选玩家；不能打自己，无效则不开枪
                if shoot_target != 0 and shoot_target not in shoot_candidates:
                    shoot_target = 0
                    print(f"  猎人开枪目标无效，选择不开枪")
            except (ValueError, Exception):
                shoot_target = 0
                print(f"  解析失败，猎人选择不开枪")

            # 标记猎人开枪已使用
            new_players[pid] = {**new_players[pid], "hunter_can_shoot": False}

            if shoot_target and shoot_target != 0:
                print(f"  猎人玩家{pid} 开枪射杀玩家{shoot_target}")
                alive.remove(shoot_target)
                dead_this_round.append(shoot_target)
                new_players[shoot_target] = {**new_players[shoot_target], "dead_reason": "hunter_shot"}
                new_history["system_announcements"] = list(new_history["system_announcements"]) + [{
                    "round": current_round,
                    "phase": phase,
                    "content": f"猎人玩家{pid}开枪射杀了玩家{shoot_target}。",
                }]
            else:
                print(f"  猎人玩家{pid} 选择不开枪")

    new_live = dict(live)
    new_live["alive_players"] = alive
    new_live["dead_players"] = list(live["dead_players"]) + dead_this_round
    new_live["pending_deaths"] = {}  # 清空待结算列表

    return {
        "live": new_live,
        "players": new_players,
        "history": new_history,
    }

# 6.胜负判定节点
def check_win_node(state: WolfGameState) -> dict:
    """
    胜负判定节点：根据存活玩家的阵营构成判断游戏是否结束。
    - 狼人全部死亡 → 好人胜
    - 神职全部死亡（屠边）→ 狼人胜
    - 平民全部死亡（屠边）→ 狼人胜
    - 否则游戏继续
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]

    # 统计存活玩家的阵营/角色构成
    wolves = [p for p in alive if players[p]["team"] == "wolf"]
    gods = [p for p in alive if players[p]["role"] in ("seer", "witch", "hunter")]
    villagers = [p for p in alive if players[p]["role"] == "villager"]

    game_result = None
    if not wolves:
        game_result = "good_win"
        result_desc = "所有狼人已出局，好人阵营胜利！"
    elif not gods:
        game_result = "wolf_win"
        result_desc = "所有神职已出局（屠边），狼人阵营胜利！"
    elif not villagers:
        game_result = "wolf_win"
        result_desc = "所有平民已出局（屠边），狼人阵营胜利！"

    new_live = dict(live)
    new_history = dict(state["history"])

    if game_result:
        new_live["game_result"] = game_result
        new_history["system_announcements"] = list(state["history"]["system_announcements"]) + [{
            "round": live["round"],
            "phase": live["phase"],
            "content": f"游戏结束：{result_desc}",
        }]
        print_stage("游戏结束")
        print(f"  {result_desc}")
        print(f"  存活狼人：{wolves}，存活神职：{gods}，存活平民：{villagers}")
    else:
        print_stage(f"游戏继续 —— 目前存活狼人：{len(wolves)}，神职：{len(gods)}，平民：{len(villagers)}")

    return {
        "live": new_live,
        "history": new_history,
    }

# 7.白天发言节点
def speech_node(state: WolfGameState) -> dict:
    """
    白天发言节点：所有存活玩家依次发言。
    每个玩家发言时能看到本轮之前所有玩家的发言（实时累积），并按角色调用对应的发言函数。
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]
    current_round = live["round"]

    # 本轮发言内容
    current_round_speeches = []          # 本轮实时发言，随循环累积
    new_memories = dict(state["memories"])

    print_stage(f"第{current_round}轮 白天发言阶段")

    for pid in alive:
        role = players[pid]["role"]

        # 格式化本轮已发言（在我之前发言的玩家）
        if current_round_speeches:
            realtime_context = "".join(
                f"  玩家{s['speaker']}号：{s['content']}\n" for s in current_round_speeches
            )
        else:
            realtime_context = "（你是本轮第一个发言的玩家，之前还没有任何人发言）"

        # 构建临时 state，让发言函数读到最新记忆
        temp_state = {**state, "memories": new_memories}

        # 按角色分流调用发言函数
        if role == "werewolf": # 狼人发言
            result = daytime_wolf_speech(temp_state, pid, realtime_context)
        elif role == "witch": # 女巫发言
            result = daytime_witch_speech(temp_state, pid, realtime_context)
        elif role == "seer": # 预言家发言
            result = daytime_seer_speech(temp_state, pid, realtime_context)
        elif role == "hunter": # 猎人发言
            result = daytime_hunter_speech(temp_state, pid, realtime_context)
        else: # 村民发言
            result = daytime_villager_speech(temp_state, pid, realtime_context)

        speech = result["speech"]
        new_memories[pid] = result["updated_memory"]

        # 追加到本轮发言，供后续玩家看到
        current_round_speeches.append({
            "speaker": pid,
            "round": current_round,
            "content": speech,
        })
        print(f"  玩家{pid}（{role}）：{speech}")

    # 本轮发言全部写入历史
    new_history = dict(state["history"])
    new_history["speech_log"] = list(state["history"]["speech_log"]) + current_round_speeches

    return {
        "memories": new_memories,
        "history": new_history,
    }

# 8.投票节点
def vote_node(state: WolfGameState) -> dict:
    """
    白天投票节点：所有存活玩家依次投票，得票最高者被淘汰。
    - 每个玩家能看到本轮之前谁投了谁（可跟票/反水）
    - 根据角色给出阵营引导（狼人保票型带好人，好人投可疑狼人）
    - 得票最高者写入 pending_deaths（死因 vote），交给结算节点处理连锁反应
    - 平票则无人淘汰
    """
    players = state["players"]
    live = state["live"]
    alive = live["alive_players"]
    current_round = live["round"]

    history_context = format_history_context(state["history"])
    votes = {}                       # {voter_id: target_id}
    current_round_votes = []         # 本轮实时投票，随循环累积

    print_stage(f"第{current_round}轮 投票阶段")

    for pid in alive:
        role = players[pid]["role"]
        mem = state["memories"][pid]
        memory_context = format_memory_context(mem, current_round)

        # 阵营投票引导
        if role == "werewolf":
            faction_hint = "你是狼人：保护狼人队友，集中票型把好人（尤其是预言家/女巫）投出去，不要投自己的队友。"
        else:
            faction_hint = "你是好人：把票投给你最怀疑是狼人的玩家，帮助好人阵营找出并淘汰狼人。"

        # 本轮已投票情况
        realtime_votes = ""
        if current_round_votes:
            realtime_votes = "".join(
                f"  玩家{v['voter']}号 → 玩家{v['target']}号\n" for v in current_round_votes
            )

        # 可投票目标（存活玩家，不含自己）
        candidates = [p for p in alive if p != pid]

        vote_prompt = ChatPromptTemplate.from_messages([
            ("system", """你正在参与狼人杀游戏。你是玩家{pid}，你的真实身份是【{role}】。现在是第{round}轮投票阶段，你需要投票淘汰一名你认为是狼人的玩家。
【你的阵营策略】
{faction_hint}

【你的记忆与推断】
{memory_context}

【历史记录（发言/投票/公告）】
{history_context}

【本轮已投票情况】
{realtime_votes}

可投票的玩家：{candidates}

【投票分析要点】
- 结合白天发言，找出逻辑矛盾、行为可疑的玩家
- 参考本轮已投票情况，可选择跟票集中票型，也可独立判断
- 若你无法确定谁是狼人，或认为弃票对你的阵营更有利，可以选择弃票（输出0）

请只输出一个整数：你要投票淘汰的玩家ID（必须是可投票玩家之一），或输出 0 表示弃票。不要输出任何解释。"""),
            ("user", "你是玩家{pid}，请投票：")
        ])

        chain = vote_prompt | llm | StrOutputParser()

        try:
            raw = chain.invoke({
                "round": current_round,
                "role": role,
                "faction_hint": faction_hint,
                "memory_context": memory_context,
                "history_context": history_context,
                "realtime_votes": realtime_votes or "（你是第一个投票的玩家）",
                "candidates": candidates,
                "pid": pid,
            }).strip()

            target = int(raw)
            # 有效目标：0（弃票）或存活玩家（不含自己）；无效则视为弃票
            if target != 0 and target not in candidates:
                print(f"  玩家{pid} 投票目标无效，视为弃票")
                target = 0
        except (ValueError, Exception):
            print(f"  玩家{pid} 解析失败，视为弃票")
            target = 0

        votes[pid] = target
        current_round_votes.append({"voter": pid, "target": target})
        print(f"  玩家{pid}（{role}）→ {'弃票' if target == 0 else f'玩家{target}'}")

    # 统计票数（排除弃票0）
    vote_count = {}
    for target in votes.values():
        if target != 0:
            vote_count[target] = vote_count.get(target, 0) + 1

    new_live = dict(live)
    new_history = dict(state["history"])

    if not vote_count:
        # 全员弃票，无人被淘汰
        eliminated = None
        result_desc = "全员弃票，本轮无人被淘汰"
    else:
        max_votes = max(vote_count.values())
        top_targets = [t for t, c in vote_count.items() if c == max_votes]
        if len(top_targets) == 1:  # 唯一最高票者被淘汰
            eliminated = top_targets[0]
            result_desc = f"玩家{eliminated}号以{max_votes}票被投票淘汰"
            # 写入 pending_deaths，交给结算节点处理（可能触发猎人开枪）
            new_pending = dict(live["pending_deaths"])
            new_pending[eliminated] = "vote"
            new_live["pending_deaths"] = new_pending
        else:
            eliminated = None
            result_desc = f"平票（玩家{top_targets}各{max_votes}票），本轮无人被淘汰"

    print(f"  投票结果：{result_desc}")

    # 记录投票和票数
    new_live["votes"] = votes
    new_live["vote_count"] = vote_count

    # 写入投票日志
    new_history["vote_results_log"] = list(state["history"]["vote_results_log"]) + [{
        "round": current_round,
        "votes": votes,
        "eliminated": eliminated,
        "result": result_desc,
    }]

    return {
        "live": new_live,
        "history": new_history,
    }

# 9.阶段推进节点
def advance_phase_node(state: WolfGameState) -> dict:
    """
    阶段推进节点：切换游戏阶段（夜晚 <-> 白天），并在进入新的一夜时推进轮次。
    - night -> daytime：同一轮，仅切换到白天
    - daytime -> night：进入下一轮，round + 1，切换到夜晚
    """
    live = state["live"]
    new_live = dict(live)

    if live["phase"] == "night":
        new_live["phase"] = "daytime"
        print_stage(f"天亮了，进入第{live['round']}轮 白天")
    else:
        new_live["phase"] = "night"
        new_live["round"] = live["round"] + 1
        print_stage(f"天黑了，进入第{new_live['round']}轮 夜晚")

    return {
        "live": new_live,
    }


# ========== 构建图 ==========
def build_werewolf_graph():
    """构建并编译狼人杀游戏的 LangGraph 状态图"""

    # 路由：check_win 之后判断游戏是否结束
    def route_after_check_win(state: WolfGameState):
        if state["live"]["game_result"] is not None:
            return END
        return "advance_phase"

    # 路由：advance_phase 之后判断进入夜晚还是白天
    def route_after_advance(state: WolfGameState):
        if state["live"]["phase"] == "night":
            return "night_wolf"
        return "speech"

    graph = StateGraph(WolfGameState)

    # 注册节点
    graph.add_node("assign_roles", assign_roles_node)
    graph.add_node("night_wolf", night_wolf_node)
    graph.add_node("night_witch", night_witch_node)
    graph.add_node("night_seer", night_seer_node)
    graph.add_node("resolve_deaths", resolve_deaths_node)
    graph.add_node("check_win", check_win_node)
    graph.add_node("advance_phase", advance_phase_node)
    graph.add_node("speech", speech_node)
    graph.add_node("vote", vote_node)

    # 入口：分配角色 -> 第一夜
    graph.add_edge(START, "assign_roles")
    graph.add_edge("assign_roles", "night_wolf")

    # 夜晚流程：狼人 -> 女巫 -> 预言家 -> 结算 -> 判定
    graph.add_edge("night_wolf", "night_witch")
    graph.add_edge("night_witch", "night_seer")
    graph.add_edge("night_seer", "resolve_deaths")

    # 结算后统一进入胜负判定
    graph.add_edge("resolve_deaths", "check_win")
    # 判定后条件路由：结束 or 推进阶段
    graph.add_conditional_edges("check_win", route_after_check_win)
    # 阶段推进后条件路由：进入夜晚 or 白天
    graph.add_conditional_edges("advance_phase", route_after_advance)

    # 白天流程：发言 -> 投票 -> 结算 -> 判定
    graph.add_edge("speech", "vote")
    graph.add_edge("vote", "resolve_deaths")

    return graph.compile()


if __name__ == "__main__":
    app = build_werewolf_graph()

    # 8人局配置：2狼 + 3民 + 1预言家 + 1女巫 + 1猎人（此处可以实现为 标准自动配置）
    init_state = create_initial_state(
        total_players=8,
        role_distribution={
            "werewolf": 2,
            "villager": 3,
            "seer": 1,
            "witch": 1,
            "hunter": 1,
        },
    )
    # 游戏从第1轮夜晚开始
    init_state["live"]["round"] = 1

    # recursion_limit 防止意外死循环（每轮约 7 个节点，给足够额度）
    result = app.invoke(init_state, config={"recursion_limit": 200})

    print_stage("游戏最终结果")
    print(f"  获胜阵营：{'好人' if result['live']['game_result'] == 'good_win' else '狼人'}")
    print(f"  最终存活玩家：{result['live']['alive_players']}")
    print("  玩家身份揭晓：")
    for pid, p in result["players"].items():
        status = "存活" if pid in result["live"]["alive_players"] else f"死亡({p['dead_reason']})"
        print(f"    玩家{pid}：{p['role']}（{status}）")