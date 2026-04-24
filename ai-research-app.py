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
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("❌ 未配置 Supabase，请先在 Secrets 中设置 SUPABASE_URL 和 SUPABASE_KEY")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# ==================== 嵌入模型 ====================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ==================== 专家角色设定 ====================
MODEL_MAP = {
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "Phi-4": "Phi-4-multimodal-instruct",
    "DeepSeek-V3": "DeepSeek-V3-0324",
    "Llama-3.3-70B": "Llama-3.3-70B-Instruct",
    "Mistral-Large": "Mistral-Large-2411",
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
    """保存到 Supabase"""
    timestamp = datetime.now().isoformat()
def save_to_supabase(topic, history, verdict):
    """保存到 Supabase，修复字段类型问题"""
    timestamp = datetime.now().isoformat()
    data = {
        "created_at": timestamp,
        "topic": topic,
        "history_json": history,          # 直接传字典，不要 json.dumps
        "final_verdict": verdict,
    }
    
    # 只有在向量扩展可用时才嵌入 embedding 字段
    try:
        embedding = embedding_model.encode(topic).tolist()
        data["embedding"] = embedding
    except Exception:
        pass   # 如果生成失败，不添加 embedding 列

    supabase.table("research").insert(data).execute()
def search_by_keyword(keyword=None, limit=10):
    query = supabase.table("research").select("id, created_at, topic").order("id", desc=True).limit(limit)
    if keyword:
        query = query.ilike("topic", f"%{keyword}%")
    res = query.execute()
    return res.data if res.data else []

def search_semantic(query_text, limit=10):
    if not query_text.strip():
        return []
    try:
        embedding = embedding_model.encode(query_text).tolist()
        res = supabase.rpc(
            "match_research",
            {"query_embedding": embedding, "match_threshold": 0.1, "match_count": limit}
        ).execute()
        return res.data if res.data else []
    except Exception as e:
        st.warning(f"语义搜索暂不可用: {e}")
        return []

def load_research_by_id(research_id):
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

    # ========== 历史记录搜索 ==========
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
                r_id, r_time, r_topic = item["id"], item["created_at"], item["topic"]
                label = f"📜 {r_time[:10]} | {r_topic[:20]}{'…' if len(r_topic)>20 else ''}"
            else:
                r_id = item["id"]
                r_time = item["created_at"]
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

# 显示历史信息
elif st.session_state.final_verdict:
    st.divider()
    st.subheader("📌 本次研讨最终结论")
    st.info(st.session_state.final_verdict)
