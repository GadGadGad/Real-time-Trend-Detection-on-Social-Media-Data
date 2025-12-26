import streamlit as st
import pandas as pd
import json
from sqlalchemy import create_engine
import time

# Cấu hình
st.set_page_config(page_title="Trend Radar Streaming", layout="wide", page_icon="📡")
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

@st.cache_resource
def get_db_connection():
    return create_engine(POSTGRES_URL)

engine = get_db_connection()

# --- UI ---
st.title("📡 Real-time Trend Detection System")
st.caption("Powered by Spark Structured Streaming & Kafka")

# Auto-refresh logic
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = time.time()

# Sidebar controls
st.sidebar.header("Control Panel")
auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
if st.sidebar.button("Clear Database (Reset Demo)"):
    with engine.connect() as conn:
        conn.execute("TRUNCATE TABLE detected_trends")
        st.success("Database cleared!")

# --- MAIN CONTENT ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Clusters", value=pd.read_sql("SELECT COUNT(*) FROM detected_trends", engine).iloc[0,0])
with col2:
    st.metric("Trending Topics", value=pd.read_sql("SELECT COUNT(*) FROM detected_trends WHERE topic_type='Trending'", engine).iloc[0,0])
with col3:
    st.metric("Last Update", value=time.strftime("%H:%M:%S"))

# Tab Views
tab_trends, tab_raw = st.tabs(["🔥 Hot Trends", "📝 Raw Clusters"])

with tab_trends:
    # Lấy Top Trends mới nhất
    query = """
    SELECT trend_name, trend_score, post_count, topic_type, created_at, representative_posts
    FROM detected_trends 
    ORDER BY created_at DESC, trend_score DESC 
    LIMIT 20
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        st.info("Chưa có dữ liệu. Hãy chạy 'file_replay_demo.py' để đẩy dữ liệu vào.")
    else:
        for idx, row in df.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1:
                    st.metric("Score", f"{row['trend_score']:.1f}")
                with c2:
                    st.subheader(row['trend_name'])
                    try:
                        posts = json.loads(row['representative_posts'])
                        for p in posts:
                            st.text(f"• [{p['source']}] {p['content']}...")
                    except: pass
                with c3:
                    st.caption(f"{row['created_at']}")
                    if row['topic_type'] == 'Trending':
                        st.error("TRENDING")
                    else:
                        st.info("Discovery")
                st.divider()

with tab_raw:
    st.dataframe(pd.read_sql("SELECT * FROM detected_trends ORDER BY created_at DESC LIMIT 50", engine))

# Auto-rerun loop
if auto_refresh:
    time.sleep(5)
    st.rerun()