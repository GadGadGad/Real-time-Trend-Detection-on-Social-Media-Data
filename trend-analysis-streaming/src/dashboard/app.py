import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import re
import html as ihtml
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy import create_engine, text

# --- Cấu hình kết nối Database ---
DB_URL = "postgresql://user:password@localhost:5432/trend_db"

st.set_page_config(page_title="Evolutionary Social Intel", layout="wide", page_icon="🌐")

# --- 1. CSS & TEMPLATE (Giữ nguyên đầy đủ) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    .main { 
        background-color: #0f172a; 
        color: #f1f5f9; 
    }

    .stMetric {
        background-color: #1e293b; 
        padding: 20px; 
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .live-feed-item {
        border-left: 6px solid #10b981; 
        background-color: #1e293b; 
        padding: 16px;
        margin-bottom: 12px; 
        border-radius: 8px; 
        border: 1px solid #334155;
        transition: transform 0.2s ease;
    }
    
    .live-feed-item:hover {
        transform: scale(1.01);
        border-color: #475569;
    }

    .source-fb { border-left-color: #3b82f6 !important; }
    .source-news { border-left-color: #f97316 !important; }
    .source-nld { border-left-color: #ef4444 !important; }
    .source-tn { border-left-color: #0ea5e9 !important; }
    
    .item-highlight { background-color: #1e293b; box-shadow: 0 0 15px rgba(16, 185, 129, 0.1); }
    .item-noise { opacity: 0.6; filter: grayscale(30%); }
    
    .topic-tag {
        background: #059669; 
        color: #ffffff; 
        padding: 4px 12px; 
        border-radius: 6px;
        font-size: 0.75rem; 
        font-weight: 700; 
        text-transform: uppercase; 
        display: inline-block; 
        margin-bottom: 8px;
    }
    .noise-tag { background: #64748b; }
    
    .sentiment-tag {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 10px;
        display: inline-block;
    }

    .source-tag {
        padding: 4px 12px; 
        border-radius: 6px; 
        font-size: 0.75rem; 
        font-weight: 700;
        display: inline-block; 
        color: #ffffff;
    }
    
    .entity-tag {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.5);
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-right: 4px;
        display: inline-block;
    }

    .st-fb { background: #3b82f6; }
    .st-news { background: #f97316; }
    .st-nld { background: #ef4444; }
    .st-tn { background: #0ea5e9; }
    .st-gen { background: #6366f1; }
    
    .time-stamp { font-size: 0.85rem; color: #94a3b8; font-weight: 600; }
    .post-content { color: #cbd5e1; line-height: 1.6; margin-top: 10px; }
    .ai-summary { font-style: italic; color: #94a3b8; margin-top: 8px; font-size: 0.9rem; border-top: 1px dashed #334155; padding-top: 8px;}
</style>
""", unsafe_allow_html=True)

# --- 2. HELPERS (gốc + thêm clean text để không lộ HTML/tag) ---
TAG_RE = re.compile(r"<[^>]+>")

def clean_text(x) -> str:
    """Unescape -> strip HTML tags -> normalize whitespace."""
    if x is None:
        return ""
    x = str(x)
    x = ihtml.unescape(x)
    x = TAG_RE.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def esc(x) -> str:
    """Escape để nhúng an toàn vào HTML string."""
    if x is None:
        return ""
    return ihtml.escape(str(x))

def normalize_source(source):
    s = str(source).strip()
    if 'Face' in s or 'beatvn' in s.lower(): return 'FACEBOOK'
    if 'VNEXPRESS' in s: return 'VNEXPRESS'
    return s.upper()

def get_source_class(source):
    s = normalize_source(source).lower()
    if 'facebook' in s: return 'source-fb', 'st-fb'
    if 'nld' in s: return 'source-nld', 'st-nld'
    if 'thanhnien' in s or 'tn' in s: return 'source-tn', 'st-tn'
    if any(x in s for x in ['news', 'vietnamnet', 'vnexpress', 'tuoitre']): return 'source-news', 'st-news'
    return '', 'st-gen'

def get_sentiment_info(label):
    s = str(label).strip().lower()
    if s == 'positive': return "TÍCH CỰC", "#10b981"
    if s == 'negative': return "TIÊU CỰC", "#ef4444"
    return "TRUNG LẬP", "#64748b"

@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

def load_realtime_data():
    engine = get_db_engine()
    query = text("SELECT * FROM detected_trends ORDER BY created_at DESC LIMIT 100")
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Lỗi kết nối Database: {e}")
        return pd.DataFrame()

def safe_parse_json(val):
    """Parse JSONB/list/string an toàn."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [val]
    if isinstance(val, str):
        try:
            x = json.loads(val)
            if isinstance(x, list):
                return x
            if isinstance(x, dict):
                return [x]
            return []
        except:
            return []
    return []

def format_ts(x):
    """Format created_at nếu có."""
    if x is None:
        return ""
    try:
        # pandas Timestamp
        return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(x)

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🌐 Intelligence Control")
score_threshold = st.sidebar.slider("Ngưỡng điểm nóng", 0.0, 100.0, 30.0)
auto_refresh = st.sidebar.toggle("Tự động cập nhật", value=True)
refresh_rate = st.sidebar.select_slider("Tần suất (giây)", options=[2, 5, 10, 30], value=5)

# ✅ thêm toggle xem chi tiết
show_details = st.sidebar.toggle("🔍 Bật xem chi tiết", value=True)
preview_chars = st.sidebar.slider("Độ dài preview", 80, 400, 180, step=20)

if st.sidebar.button("🗑️ Xóa dữ liệu", use_container_width=True):
    with get_db_engine().begin() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends"))
    st.rerun()

# --- 4. XỬ LÝ DỮ LIỆU ---
df_full = load_realtime_data()

if df_full.empty:
    st.info("📡 Đang chờ dữ liệu từ Streaming Engine...")
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

def process_evolution(row):
    score = row['trend_score']
    if score < score_threshold:
        return "DIỄN BIẾN MỚI", True
    return row['trend_name'], False

df_full[['display_topic', 'is_noise']] = df_full.apply(lambda r: pd.Series(process_evolution(r)), axis=1)
identified_df = df_full[df_full['is_noise'] == False]

# --- 5. GIAO DIỆN CHÍNH ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cụm tin", f"{len(df_full):,}")
m2.metric("Sự kiện", f"{identified_df['trend_name'].nunique():,}")
m3.metric("Max Score", f"{df_full['trend_score'].max():.1f}")
m4.metric("Live", datetime.now().strftime("%H:%M:%S"))

st.markdown("---")
col_f, col_m, col_s = st.columns([1.5, 1.5, 1.0])

# CỘT 1: FEDERATED STREAM (Tích hợp AI Analysis + CHI TIẾT)
with col_f:
    st.subheader("📡 Federated Stream")
    latest_trends = df_full.head(15)

    for i, row in latest_trends.iterrows():
        topic = row['display_topic']
        is_noise = row['is_noise']
        score = row['trend_score']

        # Sentiment & Summary
        sentiment_label = row.get('sentiment', 'Neutral')
        sent_text, sent_color = get_sentiment_info(sentiment_label)
        summary = row.get('summary', '')

        # JSON fields
        entities = safe_parse_json(row.get('top_entities'))
        posts = safe_parse_json(row.get('representative_posts'))
        main_post = posts[0] if posts else {'source': 'System', 'content': 'No content'}

        s_name = normalize_source(main_post.get('source', 'Unknown'))
        s_cls, s_tag_cls = get_source_class(s_name)
        type_cls = " item-highlight" if not is_noise else " item-noise"

        # --- Clean entities để tránh “entity dài bất thường” làm vỡ UI ---
        cleaned_entities = []
        for e in entities:
            e2 = clean_text(e)
            if not e2:
                continue
            if len(e2) > 40:
                continue
            cleaned_entities.append(e2)
            if len(cleaned_entities) >= 4:
                break

        entity_html = "".join([f'<span class="entity-tag">{esc(e)}</span>' for e in cleaned_entities])

        # --- Clean content để không lộ HTML tags ---
        content_full = clean_text(main_post.get('content', ''))
        content_preview = content_full[:preview_chars] + ("..." if len(content_full) > preview_chars else "")

        # meta (nếu có)
        created_at = format_ts(row.get('created_at'))
        category = row.get('category', '')
        post_count = row.get('post_count', '')
        score_n = row.get('score_n', '')
        score_f = row.get('score_f', '')

        # ✅ Render card (HTML)
        st.markdown(f"""
        <div class="live-feed-item {s_cls}{type_cls}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span class="topic-tag {'noise-tag' if is_noise else ''}">{esc(topic)}</span>
                    <span class="sentiment-tag" style="background: {sent_color}; color: white;">{esc(sent_text)}</span>
                </div>
                <span class="time-stamp">SCORE: {score:.1f}</span>
            </div>
            <div style="margin-bottom: 8px;">
                <span class="source-tag {s_tag_cls}">{esc(s_name)}</span>
                <span style="margin-left: 10px;">{entity_html}</span>
            </div>
            <div class="post-content">{esc(content_preview)}</div>
            {f'<div class="ai-summary"><b>AI Summary:</b> {esc(clean_text(summary))}</div>' if summary else ''}
        </div>
        """, unsafe_allow_html=True)

        # ✅ XEM CHI TIẾT (Streamlit widget, đặt ngoài HTML)
        if show_details:
            # key ổn định: ưu tiên id nếu có
            rid = row.get('id', None)
            key = f"detail_{rid if rid is not None else i}"

            with st.expander("🔎 Xem chi tiết", expanded=False):
                st.markdown(f"**Chủ đề:** `{row.get('trend_name', '')}`")
                st.markdown(f"**Display:** `{topic}`  |  **Noise:** `{is_noise}`")
                if created_at:
                    st.markdown(f"**Thời gian:** `{created_at}`")
                st.markdown(f"**Nguồn:** `{s_name}`")
                if category != "":
                    st.markdown(f"**Category:** `{category}`")
                if post_count != "":
                    st.markdown(f"**Post count:** `{post_count}`")
                if score_n != "" or score_f != "":
                    st.markdown(f"**Score N:** `{score_n}`  |  **Score F:** `{score_f}`")

                st.markdown("---")
                st.markdown("### 🧾 Nội dung đầy đủ")
                st.write(content_full if content_full else "(trống)")

                if summary:
                    st.markdown("### 🤖 AI Summary")
                    st.write(clean_text(summary))

                if cleaned_entities:
                    st.markdown("### 🏷️ Entities")
                    st.write(cleaned_entities)

                if posts:
                    st.markdown("### 📌 Representative Posts")
                    # hiển thị tối đa 5 posts để tránh dài
                    for j, p in enumerate(posts[:5], start=1):
                        p_source = normalize_source(p.get("source", "Unknown"))
                        p_content = clean_text(p.get("content", ""))
                        st.markdown(f"**{j}. Source:** `{p_source}`")
                        st.write(p_content[:800] + ("..." if len(p_content) > 800 else ""))
                        st.markdown("---")

# CỘT 2: GRAVITY MAP GỐC
with col_m:
    st.subheader("🧩 Gravity Map")
    plot_df = df_full.copy()
    plot_df['Size'] = plot_df['post_count'].apply(lambda x: min(x * 5, 50))
    plot_df['Legend'] = plot_df['display_topic'].apply(lambda x: "Emerging" if x == "DIỄN BIẾN MỚI" else x)

    if not plot_df.empty:
        fig = px.scatter(
            plot_df, x='score_n', y='score_f', size='Size', color='Legend',
            hover_name='trend_name', template="plotly_dark",
            labels={'score_n': 'Nhiệt lượng Báo chí', 'score_f': 'Tương tác Social'}
        )
        fig.update_layout(showlegend=False, height=550, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# CỘT 3: ANALYTICS (Sentiment + Category)
with col_s:
    st.subheader("📊 Analytics")

    if 'sentiment' in df_full.columns:
        sent_counts = df_full['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['sentiment', 'count']

        fig_sent = px.pie(
            sent_counts, values='count', names='sentiment', hole=0.5,
            title="Thái độ dư luận (Sentiment)",
            color='sentiment',
            color_discrete_map={'Positive':'#10b981', 'Negative':'#ef4444', 'Neutral':'#64748b'},
            template="plotly_dark"
        )
        fig_sent.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)

    if 'category' in df_full.columns:
        type_counts = df_full['category'].value_counts().reset_index()
        type_counts.columns = ['category', 'count']

        fig_cat = px.bar(
            type_counts, x='count', y='category', orientation='h',
            title="Phân loại Taxonomy (T1-T7)",
            template="plotly_dark", color='category'
        )
        fig_cat.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cat, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
