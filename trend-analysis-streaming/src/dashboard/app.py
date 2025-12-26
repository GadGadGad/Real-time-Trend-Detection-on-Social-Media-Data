import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import plotly.express as px
from datetime import datetime
from sqlalchemy import create_engine

# --- CẤU HÌNH KẾT NỐI DB (Khớp với docker-compose) ---
DB_URL = "postgresql://user:password@localhost:5432/trend_db"

# Setup Layout
st.set_page_config(page_title="Evolutionary Social Intel", layout="wide", page_icon="🌐")

# --- PREMIUM STYLING (GIỮ NGUYÊN TỪ FILE MẪU) ---
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric {
        background-color: #161b22; padding: 15px; border-radius: 12px;
        border: 1px solid #30363d; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .live-feed-item {
        border-left: 5px solid #238636; background-color: #1c2128; padding: 12px;
        margin-bottom: 10px; border-radius: 0 10px 10px 0; border: 1px solid #30363d;
        transition: all 0.3s ease; animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
    
    .source-fb { border-left-color: #1877F2 !important; }
    .source-news { border-left-color: #FF4500 !important; }
    .source-nld { border-left-color: #E60000 !important; }
    .source-tn { border-left-color: #00AEEF !important; }
    
    .item-highlight { background-color: #262c36; border-left-width: 8px !important; }
    .item-noise { opacity: 0.7; border-left-color: #484f58 !important; }
    
    .topic-tag {
        background: #238636; color: white; padding: 2px 10px; border-radius: 15px;
        font-size: 0.7em; font-weight: 800; text-transform: uppercase; display: inline-block; margin-right: 5px;
    }
    .noise-tag { background: #484f58; }
    
    .source-tag {
        padding: 2px 10px; border-radius: 15px; font-size: 0.7em; font-weight: 800;
        display: inline-block; color: white;
    }
    .st-fb { background: #1877F2; }
    .st-news { background: #FF4500; }
    .st-nld { background: #E60000; }
    .st-tn { background: #00AEEF; }
    .st-gen { background: #1f6feb; }
    
    .time-stamp { font-size: 0.8em; color: #8b949e; float: right; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
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
    """Đọc dữ liệu từ PostgreSQL do Spark đẩy vào"""
    engine = get_db_engine()
    try:
        # Lấy 100 trend/cluster mới nhất
        query = """
        SELECT * FROM detected_trends 
        ORDER BY created_at DESC 
        LIMIT 100
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối Database: {e}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.title("🌐 Live Intelligence")
st.sidebar.caption("Connected to Spark Stream")

# Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Event Evolution")
# Thay vì Discovery % (cần tổng số), ta dùng Trend Score làm ngưỡng
score_threshold = st.sidebar.slider("Ngưỡng điểm nóng (Score Threshold)", 0.0, 100.0, 30.0)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle("Auto Refresh (Real-time)", value=True)
refresh_rate = st.sidebar.select_slider("Nhịp cập nhật (giây)", options=[2, 5, 10, 30, 60], value=5)

if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    with get_db_engine().connect() as conn:
        conn.execute("TRUNCATE TABLE detected_trends") # Xóa bảng để demo lại từ đầu
        st.success("Đã xóa dữ liệu cũ!")
        time.sleep(1)
        st.rerun()

# --- MAIN APP LOGIC ---

# 1. Load Data
df_full = load_realtime_data()

st.title("🛰️ Federated Intelligence Platform")

if df_full.empty:
    st.info("📡 Đang chờ dữ liệu từ Spark... (Hãy chạy 'python src/producers/file_replay_demo.py')")
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

# 2. Process Data & Logic "Evolution"
# Logic: Nếu điểm trend < threshold -> Discovery (Noise), ngược lại -> Topic thật
def process_evolution(row):
    score = row['trend_score']
    # Nếu điểm thấp -> coi là tin mới phát hiện (Discovery)
    if score < score_threshold:
        return "Discovery", True # display_topic, is_noise
    return row['trend_name'], False

df_full[['display_topic', 'is_noise']] = df_full.apply(
    lambda r: pd.Series(process_evolution(r)), axis=1
)

# Tách dataframe
identified_df = df_full[df_full['is_noise'] == False]

# 3. Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cụm tin nhận diện", f"{len(df_full):,}")
m2.metric("Sự kiện Nóng", f"{identified_df['trend_name'].nunique():,}")
m3.metric("Điểm Trend cao nhất", f"{df_full['trend_score'].max():.1f}")
m4.metric("Live Time", datetime.now().strftime("%H:%M:%S"))

st.markdown("---")

# 4. Main Layout
# Chia cột: Feed (Trái) | Map (Giữa) | Stats (Phải)
col_f, col_m, col_s = st.columns([1.2, 1.4, 0.8])

# --- COLUMN 1: LIVE STREAM FEED ---
with col_f:
    st.subheader("📡 Live Federated Stream")
    
    # Hiển thị các cụm tin mới nhất
    # Vì Spark gom nhóm, mỗi row là 1 cụm. Ta sẽ hiển thị các bài tiêu biểu trong cụm đó.
    latest_trends = df_full.head(15) # Đã sort DESC ở SQL
    
    for _, row in latest_trends.iterrows():
        topic = row['display_topic']
        is_noise = row['is_noise']
        score = row['trend_score']
        
        # Parse bài viết đại diện từ JSON string
        try:
            posts = json.loads(row['representative_posts'])
            # Lấy bài đầu tiên làm đại diện hiển thị
            main_post = posts[0] if posts else {'source': 'System', 'content': 'No content'}
        except:
            main_post = {'source': 'System', 'content': 'Error parsing content'}
            
        s_name = normalize_source(main_post.get('source', 'Unknown'))
        s_cls, s_tag_cls = get_source_class(s_name)
        type_cls = " item-highlight" if not is_noise else " item-noise"
        
        # HTML hiển thị
        st.markdown(f"""
        <div class="live-feed-item {s_cls}{type_cls}">
            <span class="time-stamp">Score: {score:.1f}</span>
            <span class="topic-tag {'noise-tag' if is_noise else ''}">{topic if not is_noise else 'DIỄN BIẾN MỚI'}</span>
            <span class="source-tag {s_tag_cls}">{s_name}</span>
            <div style="margin-top:8px; font-weight:600; font-size: 0.9em;">
                {main_post['content'][:120]}...
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- COLUMN 2: SPATIAL MAP (Thay thế UMAP bằng Trend Gravity Chart) ---
with col_m:
    st.subheader("🧩 Trend Gravity Map")
    
    if not df_full.empty:
        # Tạo biểu đồ phân tán: X=News Score, Y=Social Score (hoặc G-Score)
        # Mục đích: Thấy được sự kiện đang nằm ở đâu (Báo chí hay Mạng xã hội hay cả 2)
        plot_df = df_full.copy()
        plot_df['Size'] = plot_df['post_count'].apply(lambda x: min(x * 5, 50)) # Scale size
        
        # Clean legend
        plot_df['Map_Legend'] = plot_df['display_topic'].apply(lambda x: "Emerging..." if x == "Discovery" else x)
        
        fig_map = px.scatter(
            plot_df, 
            x='score_n', # Điểm báo chí
            y='score_f', # Điểm Facebook
            size='Size',
            color='Map_Legend',
            hover_name='trend_name',
            hover_data={'trend_score': True, 'score_g': True},
            template="plotly_dark",
            labels={'score_n': 'Độ phủ Báo chí', 'score_f': 'Độ phủ Social'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig_map.update_layout(
            showlegend=False, 
            height=500, 
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, width='stretch', use_container_width=True)
    else:
        st.info("Chưa có dữ liệu để vẽ bản đồ.")

# --- COLUMN 3: ANALYTICS ---
with col_s:
    st.subheader("📊 Analytics")
    
    # Biểu đồ 1: Top Chủ đề (không tính Discovery)
    if not identified_df.empty:
        top_t = identified_df['trend_name'].value_counts().head(5).reset_index()
        top_t.columns = ['Topic', 'Count']
        
        fig_t = px.pie(top_t, values='Count', names='Topic', hole=0.5, template="plotly_dark")
        fig_t.update_layout(
            height=350, 
            margin=dict(l=0,r=0,t=0,b=0),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_t, width='stretch', use_container_width=True)
    else:
        st.caption("Đang phân tích cụm chủ đề...")

    # Biểu đồ 2: Phân bố loại tin (Trending vs Discovery)
    type_counts = df_full['topic_type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    fig_s = px.bar(
        type_counts, 
        x='Count', y='Type', 
        orientation='h', 
        template="plotly_dark", 
        color='Type',
        color_discrete_map={'Trending': '#238636', 'Discovery': '#484f58'}
    )
    fig_s.update_layout(
        showlegend=False, 
        height=200, 
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_s, width='stretch', use_container_width=True)

# --- AUTO REFRESH LOOP ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()