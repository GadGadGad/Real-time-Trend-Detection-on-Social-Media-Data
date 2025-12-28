import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import plotly.express as px
from datetime import datetime
from sqlalchemy import create_engine

# --- CONFIGURATION ---
# Kết nối DB khớp với docker-compose
DB_URL = "postgresql://user:password@localhost:5432/trend_db"

# Setup Layout
st.set_page_config(page_title="Evolutionary Social Intel", layout="wide", page_icon="🌐")

# --- PREMIUM STYLING (GIỮ NGUYÊN TỪ CODE CỦA BẠN) ---
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

def safe_parse_posts(json_str):
    """Hàm parse JSON an toàn, tránh crash UI"""
    try:
        if isinstance(json_str, list): return json_str
        if pd.isna(json_str) or not json_str: return []
        return json.loads(json_str)
    except:
        return []

def load_live_data():
    """Đọc dữ liệu Real-time từ DB"""
    engine = get_db_engine()
    try:
        # Lấy 150 xu hướng mới nhất
        query = """
        SELECT * FROM detected_trends 
        ORDER BY created_at DESC 
        LIMIT 150
        """
        df = pd.read_sql(query, engine)
        if not df.empty and 'created_at' in df.columns:
            df['time'] = pd.to_datetime(df['created_at'])
        return df
    except Exception as e:
        return pd.DataFrame()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🌐 Live Intelligence")
st.sidebar.caption("Connected to Spark Cluster")

# Evolution Setting
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Event Evolution")
# Thay vì % dữ liệu, ta dùng Trend Score làm ngưỡng lọc tin rác
score_threshold = st.sidebar.slider("Ngưỡng điểm nóng (Score Filter)", 0.0, 50.0, 15.0)

# Sidebar Controls
st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle("Auto Refresh (Live)", value=True)
refresh_rate = st.sidebar.select_slider("Nhịp cập nhật (giây)", options=[2, 5, 10, 30], value=5)

if st.sidebar.button("🗑️ Reset Database (Demo)", use_container_width=True):
    try:
        with get_db_engine().connect() as conn:
            conn.execute("TRUNCATE TABLE detected_trends")
        st.success("Database cleared!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# --- MAIN LOGIC ---
st.title("🛰️ Federated Intelligence Platform")

# 1. Load Data
df_full = load_live_data()

if df_full.empty:
    st.warning("⚠️ Chưa có dữ liệu từ Spark. Đang lắng nghe Kafka...")
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

# 2. Process Evolution Logic
def process_evolving_topic(row):
    score = row.get('trend_score', 0)
    topic = row.get('trend_name', 'Unknown')
    
    # Logic: Nếu điểm thấp -> Coi là đang khám phá (Discovery/Noise)
    is_noise = score < score_threshold
    
    display_topic = "Discovery" if is_noise else topic
    return display_topic, is_noise

df_full[['display_topic', 'is_noise']] = df_full.apply(
    lambda r: pd.Series(process_evolving_topic(r)), axis=1
)

# Tách tập dữ liệu đã nhận diện (Confirmed Trends)
identified_clusters_df = df_full[df_full['is_noise'] == False]

# 3. Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cụm tin phát hiện", f"{len(df_full):,}")
m2.metric("Sự kiện xác thực", f"{identified_clusters_df['display_topic'].nunique():,}")
m3.metric("Điểm nóng nhất", f"{df_full['trend_score'].max():.1f}")
m4.metric("Live Time", datetime.now().strftime("%H:%M:%S"))

st.markdown("---")

# 4. Layout Columns
col_f, col_m, col_s = st.columns([1.2, 1.4, 0.8])

# --- COLUMN 1: LIVE FEED ---
with col_f:
    st.subheader("📡 Live Federated Stream")
    latest = df_full.head(20) # Lấy 20 tin mới nhất
    
    if latest.empty: 
        st.info("Đang chờ dữ liệu...")
    else:
        for _, row in latest.iterrows():
            topic = row['display_topic']
            is_noise = row['is_noise']
            score = row['trend_score']
            
            # Parse bài viết đại diện từ JSON
            rep_posts = safe_parse_posts(row['representative_posts'])
            main_post = rep_posts[0] if rep_posts else {'source': 'System', 'content': 'Analyzing...'}
            
            s_name = normalize_source(main_post.get('source', 'Unknown'))
            s_cls, s_tag_cls = get_source_class(s_name)
            type_cls = " item-highlight" if not is_noise else " item-noise"
            
            # Nội dung hiển thị
            content_display = main_post.get('content', '')[:110]
            title_display = row.get('trend_name') if not is_noise else "Dữ liệu đang được gom nhóm..."
            
            st.markdown(f"""
            <div class="live-feed-item {s_cls}{type_cls}">
                <span class="time-stamp">Score: {score:.1f}</span>
                <span class="topic-tag {'noise-tag' if is_noise else ''}">{topic if not is_noise else 'DIỄN BIẾN MỚI'}</span>
                <span class="source-tag {s_tag_cls}">{s_name}</span>
                <div style="margin-top:8px; font-weight:600;">{content_display}...</div>
            </div>
            """, unsafe_allow_html=True)

# --- COLUMN 2: SPATIAL MAP (Thay UMAP bằng Gravity Map) ---
with col_m:
    st.subheader("🧩 Trend Gravity Map")
    
    # Vẽ bản đồ phân tán dựa trên điểm số (News vs Social)
    # Đây là cách hiển thị không gian hiệu quả cho Streaming data thay vì UMAP tốn kém
    if not df_full.empty:
        plot_df = df_full.copy()
        
        # Tạo toạ độ giả lập dựa trên điểm số để phân bố lên biểu đồ
        # X: News Score, Y: Facebook Score
        plot_df['x'] = plot_df['score_n'] + np.random.normal(0, 0.5, len(plot_df)) # Jitter để đỡ đè nhau
        plot_df['y'] = plot_df['score_f'] + np.random.normal(0, 0.5, len(plot_df))
        
        # Legend
        plot_df['map_topic'] = plot_df['display_topic'].apply(lambda x: "Quét dữ liệu..." if x == "Discovery" else x)
        
        fig_map = px.scatter(
            plot_df, x='x', y='y', 
            color='map_topic', 
            size='post_count',
            template="plotly_dark",
            hover_name='trend_name',
            labels={'x': 'News Signal', 'y': 'Social Signal'},
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        # Tắt grid để trông giống map không gian hơn
        fig_map.update_layout(
            showlegend=False, 
            height=500, 
            margin=dict(l=0, r=0, t=0, b=0), 
            xaxis={'visible': False}, 
            yaxis={'visible': False},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, width='stretch', use_container_width=True)

# --- COLUMN 3: ANALYTICS ---
with col_s:
    st.subheader("📊 Analytics")
    
    # Pie Chart: Topic Distribution
    if not identified_clusters_df.empty:
        top_t = identified_clusters_df['display_topic'].value_counts().head(8).reset_index()
        top_t.columns = ['Topic', 'Count']
        
        fig_t = px.pie(top_t, values='Count', names='Topic', hole=0.4, template="plotly_dark")
        fig_t.update_layout(
            height=350, 
            margin=dict(l=0,r=0,t=0,b=0),
            showlegend=True,
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_t, width='stretch', use_container_width=True)
    else: 
        st.caption("Đang phân tích mẫu tin...")

    # Bar Chart: Source Distribution (Ước lượng từ representative_posts)
    # Vì DB Streaming lưu theo cụm, ta đếm xu hướng theo nguồn tin chính
    if not df_full.empty:
        # Extract source from first rep post
        def get_prime_source(json_val):
            try:
                posts = json.loads(json_val)
                if posts: return normalize_source(posts[0].get('source', 'Unknown'))
            except: pass
            return 'Unknown'
            
        source_counts = df_full['representative_posts'].apply(get_prime_source).value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        
        fig_s = px.bar(
            source_counts.head(10), 
            x='Count', y='Source', 
            orientation='h', 
            template="plotly_dark", 
            color='Source', 
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_s.update_layout(
            showlegend=False, 
            height=300, 
            margin=dict(l=0,r=0,t=0,b=0), 
            yaxis={'categoryorder':'total ascending'},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_s, width='stretch', use_container_width=True)

# --- REAL-TIME LOOP ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()