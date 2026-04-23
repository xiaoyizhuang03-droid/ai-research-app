import streamlit as st
import openai
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# ==================== 页面配置 ====================
st.set_page_config(page_title="AI 专家精研室", page_icon="🏢", layout="wide")

# ==================== Supabase 配置 ====================
# 从 Streamlit Secrets 读取凭据
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("❌ 未配置 Supabase，请先在 Secrets 中设置 SUPABASE_URL 和 SUPABASE_KEY")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# ==================== 嵌入模型（用于语义搜索） ====================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ==================== 数据库表初始化（自动创建） ====================
def ensure_table_exists():
    """在 Supabase 中创建 research 表（如果不存在）"""
    try:
        # 尝试查询一条记录以测试表是否存在
        supabase.table("research").select("id").limit(1).execute()
    except Exception as e:
        # 表不存在则手动创建（通过 SQL 执行，但 Supabase 客户端不直接支持 DDL）
        # 替代方案：在 Supabase 控制台执行以下 SQL，或程序自动执行（需开启 pg_cron 等权限）
        # 这里我们提示用户去控制台建表，以免复杂化
        st.warning("⚠️ 请先在 Supabase 控制台执行以下 SQL 创建表，详见下方说明。")
        st.code("""
CREATE TABLE research (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    topic TEXT NOT NULL,
    history_json JSONB NOT NULL,
    final_verdict TEXT NOT NULL,
    embedding VECTOR(384)  -- 需要安装 pgvector 扩展
);
        """)
        st.stop()

ensure_table_exists()

# ==================== 专家角色设定 ====================
MODEL_MAP = {
    "gpt-4o": "openai/gpt-4o",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral-large": "mistral-ai/Mistral-large"
}

SYSTEM_PROMPTS = {
    "researcher": "你是一位严谨的研究员。请针对主题提供详尽、多维度的初始报告，要求逻辑清晰，尽可能挖掘深层原因。",
    "reviewer": "你是一位专业的评审专家。请审阅初始报告，指出其中的逻辑漏洞、遗漏的风险点或未考虑到的视角，并提供优化建议。",
    "synthesizer": "你是一位首席分析师。请结合初步报告和评审意见，剔除冗余，整合精髓，输出一份极具洞察力且精炼的最终结论。"
}

# ==================== Session State ====================
if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "final_verdict" not in st.session_state:
    st.session_state.final_verdict = ""

# ==================== 工具函数 ====================
def stream_chat(client, model, messages):
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.5, stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n\n❌ 运行出错: {str(e)}"

def save_to_supabase(topic, history, verdict):
    """将研讨记录保存到 Supabase，同时生成向量嵌入（如果扩展已启用）"""
    timestamp = datetime.now().isoformat()
    try:
        # 生成主题向量（仅当 pgvector 扩展已安装时有效）
        embedding = embedding_model.encode(topic).tolist()
        data = {
            "timestamp": timestamp,
            "topic": topic,
            "history_json": json.dumps(history, ensure_ascii=False),
            "final_verdict": verdict,
            "embedding": embedding
        }
    except Exception:
        # 如果字段不存在或未安装向量扩展，则不写入向量
        data = {
            "timestamp": timestamp,
            "topic": topic,
            "history_json": json.dumps(history, ensure_ascii=False),
            "final_verdict": verdict
        }
    supabase.table("research").insert(data).execute()

def search_by_keyword(keyword=None, limit=10):
    """关键词搜索（使用 Supabase 的 ilike）"""
    query = supabase.table("research").select("id, timestamp, topic").order("id", desc=True).limit(limit)
    if keyword:
        query = query.ilike("topic", f"%{keyword}%")
    res = query.execute()
    return res.data if res.data else []

def search_semantic(query_text, limit=10):
    """语义搜索（需要 pgvector 扩展和 embedding 字段）"""
    if not query_text.strip():
        return []
    try:
        embedding = embedding_model.encode(query_text).tolist()
        # 调用 Supabase 的 RPC 函数进行向量相似度搜索（需提前创建函数）
        res = supabase.rpc(
            "match_research",
            {"query_embedding": embedding, "match_threshold": 0.1, "match_count": limit}
        ).execute()
        return res.data if res.data else []
    except Exception as e:
        st.warning(f"语义搜索暂不可用（请确保已安装 pgvector 并创建 match_research 函数）: {e}")
        return []

def load_research_by_id(research_id):
    """根据 ID 加载完整记录"""
    res = supabase.table("research").select("history_json, final_verdict").eq("id", research_id).execute()
    if res.data and len(res.data) > 0:
        data = res.data[0]
        return json.loads(data["history_json"]), data["final_verdict"]
    return None, None

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 研讨配置")
    token = st.text_input(
        "GitHub Token",
        type="password",
        help="在 https://github.com/settings/tokens 生成，无需任何权限"
    )

    st.divider()
    st.subheader("🧠 选择专家模型")
    m_res = st.selectbox("研究员 (Drafting)", list(MODEL_MAP.keys()), index=0)
    m_rev = st.selectbox("评审员 (Reviewing)", list(MODEL_MAP.keys()), index=1)
    m_syn = st.selectbox("总结官 (Finalizing)", list(MODEL_MAP.keys()), index=0)

    st.divider()
    if st.button("🔄 重置研讨", use_container_width=True):
        st.session_state.research_history = []
        st.session_state.is_running = False
        st.session_state.final_verdict = ""
        st.rerun()

    # ========== 历史记录搜索区域 ==========
    st.divider()
    st.subheader("📚 往期研讨记录")

    search_mode = st.radio("搜索模式", ["🔤 关键词", "🧠 语义搜索"], horizontal=True)

    if search_mode == "🔤 关键词":
        keyword = st.text_input("🔍 搜索主题关键词", placeholder="例如：竞争力、AI...")
        records = search_by_keyword(keyword=keyword if keyword else None, limit=10)
    else:
        query = st.text_input("🤖 语义搜索", placeholder="描述你想找的内容...")
        if query:
            records = search_semantic(query, limit=10)
        else:
            records = []

    if not records:
        st.caption("暂无记录，开始一场研讨吧！")
    else:
        for item in records:
            if search_mode == "🔤 关键词":
                r_id, r_time, r_topic = item["id"], item["timestamp"], item["topic"]
                label = f"📜 {r_time[:10]} | {r_topic[:20]}{'…' if len(r_topic)>20 else ''}"
            else:
                # 语义搜索结果包含相似度
                r_id = item["id"]
                r_time = item["timestamp"]
                r_topic = item["topic"]
                similarity = item.get("similarity", 0.0)
                label = f"📜 {r_time[:10]} | {r_topic[:20]}… (相似度 {similarity:.2f})"

            if st.button(label, key=f"hist_{r_id}", use_container_width=True):
                hist, verdict = load_research_by_id(r_id)
                if hist is not None:
                    st.session_state.research_history = hist
                    st.session_state.final_verdict = verdict
                    st.session_state.is_running = False
                    st.rerun()

# ==================== 主界面 ====================
st.title("🏛️ AI 专家精研室")
st.markdown("通过“初稿-评审-定稿”三阶段流程，为您打磨最深刻的课题结论。")

topic = st.text_area("📝 研讨课题", value="后人工智能时代，人类核心竞争力的重构逻辑", height=100)

if st.button("🚀 开始研讨", type="primary", disabled=not token or st.session_state.is_running):
    st.session_state.is_running = True
    st.session_state.research_history = []
    st.session_state.final_verdict = ""
    st.rerun()

# --- 研讨执行逻辑 ---
if st.session_state.is_running:
    client = openai.OpenAI(base_url="https://models.inference.ai.azure.com", api_key=token)

    # 1. 初始分析阶段
    with st.chat_message("assistant", avatar="📝"):
        st.subheader("第一阶段：深度课题分析")
        ph = st.empty()
        full_text = ""
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPTS["researcher"]},
            {"role": "user", "content": f"课题：{topic}。请开始你的深度分析报告。"}
        ]
        for chunk in stream_chat(client, MODEL_MAP[m_res], msgs):
            full_text += chunk
            ph.markdown(full_text + "▌")
        ph.markdown(full_text)
        st.session_state.research_history.append({"role": "Researcher", "content": full_text})

    # 2. 评审优化阶段
    with st.chat_message("assistant", avatar="🔍"):
        st.subheader("第二阶段：逻辑审视与优化建议")
        ph = st.empty()
        full_text_rev = ""
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPTS["reviewer"]},
            {"role": "user", "content": f"以下是初始报告：\n{full_text}\n\n请指出盲点并补充。"}
        ]
        for chunk in stream_chat(client, MODEL_MAP[m_rev], msgs):
            full_text_rev += chunk
            ph.markdown(full_text_rev + "▌")
        ph.markdown(full_text_rev)
        st.session_state.research_history.append({"role": "Reviewer", "content": full_text_rev})

    # 3. 最终定稿阶段
    with st.chat_message("assistant", avatar="🏆"):
        st.subheader("第三阶段：最终精炼结论")
        ph = st.empty()
        full_text_syn = ""
        context = f"初始分析：{full_text}\n\n评审建议：{full_text_rev}"
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPTS["synthesizer"]},
            {"role": "user", "content": f"结合上述内容，请给出一个极简、深刻的最终结论：\n{context}"}
        ]
        for chunk in stream_chat(client, MODEL_MAP[m_syn], msgs):
            full_text_syn += chunk
            ph.markdown(full_text_syn + "▌")
        ph.markdown(full_text_syn)
        st.session_state.final_verdict = full_text_syn

    # 保存结果到 Supabase
    save_to_supabase(topic, st.session_state.research_history, full_text_syn)
    st.session_state.is_running = False
    st.success("🎉 精研完成！结论已永久保存至云端。")

# 显示历史信息（若非运行状态且有结论）
elif st.session_state.final_verdict:
    st.divider()
    st.subheader("📌 本次研讨最终结论")
    st.info(st.session_state.final_verdict)