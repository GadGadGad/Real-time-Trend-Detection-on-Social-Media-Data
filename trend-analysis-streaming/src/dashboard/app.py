import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import plotly.express as px
from datetime import datetime
from sqlalchemy import create_engine, text

DB_URL = "postgresql://user:password@localhost:5432/trend_db"

st.set_page_config(page_title="Evolutionary Social Intel", layout="wide", page_icon="🌐")

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
    
    .source-tag {
        padding: 4px 12px; 
        border-radius: 6px; 
        font-size: 0.75rem; 
        font-weight: 700;
        display: inline-block; 
        color: #ffffff;
    }
    .st-fb { background: #3b82f6; }
    .st-news { background: #f97316; }
    .st-nld { background: #ef4444; }
    .st-tn { background: #0ea5e9; }
    .st-gen { background: #6366f1; }
    
    .time-stamp { font-size: 0.85rem; color: #94a3b8; font-weight: 600; }
    .post-content { color: #cbd5e1; line-height: 1.6; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

def load_realtime_data():
    engine = get_db_engine()
    query = text("SELECT * FROM detected_trends ORDER BY created_at DESC LIMIT 100")
    return pd.read_sql(query, engine)

st.sidebar.title("🌐 Intelligence Control")
score_threshold = st.sidebar.slider("Ngưỡng điểm nóng", 0.0, 100.0, 30.0)
auto_refresh = st.sidebar.toggle("Tự động cập nhật", value=True)
refresh_rate = st.sidebar.select_slider("Tần suất (giây)", options=[2, 5, 10, 30], value=5)

if st.sidebar.button("🗑️ Xóa dữ liệu", use_container_width=True):
    with get_db_engine().begin() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends"))
    st.rerun()

df_full = load_realtime_data()

if df_full.empty:
    st.info("📡 Đang chờ dữ liệu...")
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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cụm tin", f"{len(df_full):,}")
m2.metric("Sự kiện", f"{identified_df['trend_name'].nunique():,}")
m3.metric("Max Score", f"{df_full['trend_score'].max():.1f}")
m4.metric("Live", datetime.now().strftime("%H:%M:%S"))

st.markdown("---")
col_f, col_m, col_s = st.columns([1.3, 1.5, 0.8])

with col_f:
    st.subheader("📡 Federated Stream")
    latest_trends = df_full.head(15)
    
    for _, row in latest_trends.iterrows():
        topic = row['display_topic']
        is_noise = row['is_noise']
        score = row['trend_score']
        
        raw_posts = row['representative_posts']
        posts = json.loads(raw_posts) if isinstance(raw_posts, str) else raw_posts
        main_post = posts[0] if isinstance(posts, list) and posts else {'source': 'System', 'content': 'No content'}
            
        s_name = normalize_source(main_post.get('source', 'Unknown'))
        s_cls, s_tag_cls = get_source_class(s_name)
        type_cls = " item-highlight" if not is_noise else " item-noise"
        
        st.markdown(f"""
        <div class="live-feed-item {s_cls}{type_cls}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <span class="topic-tag {'noise-tag' if is_noise else ''}">{topic}</span>
                <span class="time-stamp">SCORE: {score:.1f}</span>
            </div>
            <div><span class="source-tag {s_tag_cls}">{s_name}</span></div>
            <div class="post-content">{main_post['content'][:140]}...</div>
        </div>
        """, unsafe_allow_html=True)

with col_m:
    st.subheader("🧩 Gravity Map")
    plot_df = df_full.copy()
    plot_df['Size'] = plot_df['post_count'].apply(lambda x: min(x * 5, 50))
    plot_df['Legend'] = plot_df['display_topic'].apply(lambda x: "Emerging" if x == "DIỄN BIẾN MỚI" else x)
    
    fig = px.scatter(
        plot_df, x='score_n', y='score_f', size='Size', color='Legend',
        hover_name='trend_name', template="plotly_dark",
        labels={'score_n': 'Báo chí', 'score_f': 'Social'}
    )
    fig.update_layout(showlegend=False, height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with col_s:
    st.subheader("📊 Analytics")
    if not identified_df.empty:
        top_t = identified_df['trend_name'].value_counts().head(5).reset_index()
        top_t.columns = ['Topic', 'Count']
        fig_t = px.pie(top_t, values='Count', names='Topic', hole=0.5, template="plotly_dark")
        fig_t.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)

    type_counts = df_full['topic_type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    fig_s = px.bar(type_counts, x='Count', y='Type', orientation='h', template="plotly_dark", color='Type')
    fig_s.update_layout(showlegend=False, height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_s, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()