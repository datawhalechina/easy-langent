import os
import uuid
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional, TypedDict, Annotated
import operator
from typing_extensions import NotRequired
from dotenv import load_dotenv

# 核心算法与数据库
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ===================== 1. 基础环境配置 =====================
load_dotenv()
st.set_page_config(page_title="哈尔滨冰雪大世界：舆情决策系统", page_icon="❄️", layout="wide")

# 获取 API Key (优先从侧边栏或环境变量获取
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = st.sidebar.text_input("🔑 请输入 DashScope API Key", type="password")
    if API_KEY:
        os.environ["API_KEY"] = API_KEY

# 本地数据路径
DATA_FOLDER = "data"


# ===================== 2. 向量语义引擎 (FAISS + 自动分批构建 + 本地缓存) =====================
class VectorDataCenter:
    def __init__(self, folder_path, api_key):
        self.index_path = "faiss_local_index"  # 本地索引缓存文件夹
        self.df = self._load_data(folder_path)
        self.vector_db = None

        if api_key and not self.df.empty:
            self.embeddings = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key
            )
            # 自动检查缓存：如果有缓存，0.1秒加载；没有则启动分批构建
            if os.path.exists(self.index_path):
                try:
                    self.vector_db = FAISS.load_local(
                        self.index_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.sidebar.success("⚡ 已从本地极速加载语义库缓存")
                except Exception as e:
                    st.sidebar.error(f"加载缓存失败，重新构建: {e}")
                    self._build_vector_db()
            else:
                self._build_vector_db()

    def _load_data(self, path):
        all_data = []
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(('.xlsx', '.csv')) and not f.startswith('~$')]
            for f in files:
                try:
                    p = os.path.join(path, f)
                    df = pd.read_excel(p) if f.endswith('.xlsx') else pd.read_csv(p)
                    all_data.append(df)
                except:
                    continue
        if not all_data:
            # 兜底测试数据
            return pd.DataFrame({
                "清洗后正文": ["排队太久了，零下三十度受罪", "保安送姜茶很暖心", "滑梯预约黄牛横行", "尔滨服务升级"],
                "情感分数": [0.15, 0.9, 0.2, 0.85]
            })
        return pd.concat(all_data, ignore_index=True)

    def _build_vector_db(self):
        """分批次构建，防止网络中断，完成后保存至硬盘"""
        texts = self.df['清洗后正文'].fillna("无内容").tolist()
        metadatas = [{"score": row["情感分数"]} for _, row in self.df.iterrows()]

        batch_size = 100  # 核心：每批处理100条，降低网络压力
        total_batches = (len(texts) + batch_size - 1) // batch_size

        with st.spinner(f"🚀 正在分批训练语义大脑 (共 {total_batches} 批)..."):
            # 1. 初始化数据库（第一批）
            self.vector_db = FAISS.from_texts(
                texts[:batch_size],
                self.embeddings,
                metadatas=metadatas[:batch_size]
            )

            # 2. 循环添加后续批次
            progress_bar = st.progress(0)
            for i in range(1, total_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(texts))
                self.vector_db.add_texts(texts[start:end], metadatas=metadatas[start:end])
                progress_bar.progress((i + 1) / total_batches)

            # 3. 核心：存入本地磁盘，以后再也不用等待
            self.vector_db.save_local(self.index_path)

        st.sidebar.success("✅ 语义库已完成持久化缓存")

    def semantic_search(self, query: str, k: int = 3):
        if not self.vector_db: return "未找到对标案例", 0.5
        docs = self.vector_db.similarity_search(query, k=k)
        context = "\n".join([f"- {d.page_content} (极性:{d.metadata['score']})" for d in docs])
        avg_pol = sum([d.metadata['score'] for d in docs]) / len(docs)
        return context, float(avg_pol)


# 🌟 利用 Streamlit 资源缓存，确保 VectorCenter 全局唯一，不重复加载
@st.cache_resource
def get_vector_center(folder_path, api_key):
    if not api_key: return None
    return VectorDataCenter(folder_path, api_key)


# ===================== 3. 多智能体架构设计 =====================
class SandboxState(TypedDict):
    crisis_event: str
    rag_context: str
    historical_polarity: float
    op_report: NotRequired[str]
    fin_report: NotRequired[str]
    pr_draft: NotRequired[str]
    final_score: NotRequired[float]
    debate_log: Annotated[List[str], operator.add]
    timeline: Annotated[List[str], operator.add]
    rework_command: NotRequired[str]


# 节点1：数据语义分析 (RAG)
def rag_node(state: SandboxState):
    vdc = get_vector_center(DATA_FOLDER, API_KEY)
    if not vdc: return {"timeline": ["❌ 错误：未输入 API KEY"]}
    context, polarity = vdc.semantic_search(state["crisis_event"])
    return {"rag_context": context, "historical_polarity": polarity, "timeline": ["🧠 语义引擎：历史语义对标完成"]}


# 节点2：运营总监 (现场处置)
def op_agent(state: SandboxState):
    llm = ChatOpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus")
    prompt = f"针对危机：{state['crisis_event']}，请作为运营总监给出40字内现场管理和分流建议。"
    res = llm.invoke(prompt).content
    return {"op_report": res, "timeline": ["🛠️ 运营专家：现场策略就绪"]}


# 节点3：财务总监 (成本审批)
def fin_agent(state: SandboxState):
    llm = ChatOpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus")
    prompt = f"历史情绪极性：{state['historical_polarity']}，请作为财务总监给出30字内退赔或预算审批意见。"
    res = llm.invoke(prompt).content
    return {"fin_report": res, "timeline": ["💰 财务总监：资金链路已核算"]}


# 节点4：公关总监 (全量输出)
def pr_agent(state: SandboxState):
    llm = ChatOpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus")
    rework = state.get("rework_command", "无")
    prompt = f"融合运营建议：{state['op_report']}和财务意见：{state['fin_report']}，针对重修指令：{rework}，撰写正式公关通稿。"
    res = llm.invoke(prompt).content
    return {"pr_draft": res, "timeline": ["📢 公关总监：危机通稿拟定完成"]}


# 节点5：主管审核 (网民仿真)
def supervisor_node(state: SandboxState):
    llm = ChatOpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus")
    prompt = f"作为一名刻薄网民，评价此方案并打分。末行写'指数：数字'：\n{state['pr_draft']}"
    res = llm.invoke(prompt).content
    score = 50.0
    match = re.search(r'指数[：:]\s*(\d+)', res)
    if match: score = float(match.group(1))
    return {"final_score": score, "debate_log": [res], "timeline": ["🌐 仿真中心：网民矩阵测试结束"]}


# ===================== 4. 构建与绘图 =====================
def build_pro_app():
    builder = StateGraph(SandboxState)
    builder.add_node("rag", rag_node)
    builder.add_node("op", op_agent)
    builder.add_node("fin", fin_agent)
    builder.add_node("pr", pr_agent)
    builder.add_node("sup", supervisor_node)

    builder.set_entry_point("rag")
    builder.add_edge("rag", "op")
    builder.add_edge("op", "fin")
    builder.add_edge("fin", "pr")
    builder.add_edge("pr", "sup")
    builder.add_edge("sup", END)

    return builder.compile(checkpointer=MemorySaver())


def draw_chart(score):
    """基于当前分数的时序演化仿真图表"""
    hours = np.arange(0, 78, 6)
    # 模拟情绪平复模型
    trend = score * np.exp(-0.04 * hours) + np.random.normal(0, 2, len(hours))
    trend = np.clip(trend, 0, 100)
    fig = go.Figure(go.Scatter(x=hours, y=trend, mode='lines+markers',
                               line=dict(color='#FF4B4B', width=3),
                               name='预测轨迹'))
    fig.update_layout(title="📉 未来 72 小时压力发酵预测曲线", template="plotly_white", xaxis_title="小时",
                      yaxis_title="危险指数")
    return fig


# ===================== 5. Streamlit 主界面 =====================
st.title("❄️ 哈尔滨冰雪大世界：舆情决策系统")
st.caption("全栈 AI 架构：LangGraph 状态机 / FAISS 语义库 / Plotly 动态仿真")

if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "app" not in st.session_state: st.session_state.app = build_pro_app()
if "current_state" not in st.session_state: st.session_state.current_state = None

# 侧边栏
with st.sidebar:
    st.header("🧠 决策流状态")
    st.write(f"Session ID: {st.session_state.thread_id[:8]}")
    if st.button("🔄 重置沙盘"):
        st.session_state.current_state = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 聊天式输入
user_input = st.chat_input("🚨 请输入危机事件描述...")

if user_input:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    # 判断是启动还是修正
    if st.session_state.current_state is None:
        init_state = {"crisis_event": user_input, "debate_log": [], "timeline": []}
    else:
        init_state = None
        st.session_state.app.update_state(config, {"rework_command": user_input})

    with st.spinner("🤖 正在启动多智能体分层决策系统..."):
        result = st.session_state.app.invoke(init_state, config=config)
        st.session_state.current_state = result

# 渲染仪表盘
if st.session_state.current_state:
    s = st.session_state.current_state

    # 指标行
    m1, m2, m3 = st.columns(3)
    m1.metric("📌 语义库匹配极性", f"{s.get('historical_polarity', 0):.2f}")
    m2.metric("📉 预测负面峰值", f"{s.get('final_score', 0):.1f}/100")
    m3.metric("🤝 智能体协同", "4个部门已参与")

    st.divider()

    # 核心展示区
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.plotly_chart(draw_chart(s.get("final_score", 50)), use_container_width=True)
    with col_r:
        st.subheader("📢 决策通稿草案")
        st.info(s.get("pr_draft"))

    # 黑盒解密
    with st.expander("🕵️ 查看各部门决策内部逻辑"):
        tc1, tc2, tc3 = st.columns(3)
        tc1.error(f"**[语义 RAG 历史对比]**\n{s.get('rag_context')}")
        tc2.warning(f"**[运营部门建议]**\n{s.get('op_report')}")
        tc3.success(f"**[财务部门预算]**\n{s.get('fin_report')}")

    # 报告导出
    if st.button("✅ 批准并导出 HTML 报告"):
        st.balloons()
        report_html = f"""
        <!DOCTYPE html><html><head><meta charset="utf-8"><title>报告</title></head>
        <body style="font-family:sans-serif;background:#f4f7f6;padding:40px;">
            <div style="background:white;padding:30px;border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color:#2980b9;">❄️ 尔滨舆情决策报告</h2>
                <p>危机：{s['crisis_event']}</p>
                <p>负面分：{s['final_score']}</p>
                <hr><div style="line-height:1.6;">{s['pr_draft'].replace('\n', '<br>')}</div>
            </div>
        </body></html>
        """
        st.download_button("📥 点击下载报告", data=report_html, file_name="final_report.html", mime="text/html")