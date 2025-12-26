import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import glob
import json
import plotly.express as px
from datetime import datetime, timedelta
from umap import UMAP

# Setup
st.set_page_config(page_title="Evolutionary Social Intel", layout="wide", page_icon="🌐")

# Premium Styling
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
    
    .sent-pos { color: #2ea043; font-weight: bold; }
    .sent-neg { color: #f85149; font-weight: bold; }
    .sent-neu { color: #8b949e; }
    
    .time-stamp { font-size: 0.8em; color: #8b949e; float: right; }
    [data-testid="stMetricValue"] { text-align: center; }
</style>
""", unsafe_allow_html=True)

def normalize_source(source):
    s = str(source).strip()
    if s.lower().startswith('face'): return 'FACEBOOK'
    return s.upper()

def get_source_class(source):
    s = normalize_source(source).lower()
    if 'facebook' in s: return 'source-fb', 'st-fb'
    if 'nld' in s: return 'source-nld', 'st-nld'
    if 'thanhnien' in s or 'tn' in s: return 'source-tn', 'st-tn'
    if any(x in s for x in ['news', 'vietnamnet', 'vnexpress', 'tuoitre']): return 'source-news', 'st-news'
    return '', 'st-gen'

def get_sentiment_html(sent):
    s = str(sent).lower()
    if 'positive' in s: return '<span class="sent-pos">POSITIVE</span>'
    if 'negative' in s: return '<span class="sent-neg">NEGATIVE</span>'
    return '<span class="sent-neu">NEUTRAL</span>'

# ================= DATA PROVIDERS =================

# 1. FILE PROVIDER (Simulation)
def scan_files():
    patterns = ["demo/data/*.parquet", "demo/data/*.json", "crawlers/results/*.json"]
    found = []
    for p in patterns: found.extend(glob.glob(p))
    return sorted(list(set(found)))

@st.cache_data
def load_simulation_data(paths):
    if not paths: return None, None, {}
    dfs = []
    for p in paths:
        try:
            if p.endswith('.parquet'): d = pd.read_parquet(p)
            else:
                with open(p, 'r', encoding='utf-8') as f: d = pd.DataFrame(json.load(f))
            if 'time' in d.columns: d['time'] = pd.to_datetime(d['time'], utc=True, errors='coerce')
            dfs.append(d)
        except: continue
    if not dfs: return None, None, {}
    df = pd.concat(dfs, ignore_index=True).dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    df['source'] = df['source'].apply(normalize_source)
    total_counts = df['final_topic'].value_counts().to_dict()
    embeddings = None
    if len(paths) == 1 and paths[0] == "demo/data/results.parquet" and os.path.exists("demo/data/post_embeddings.npy"):
        embeddings = np.load("demo/data/post_embeddings.npy")
        if 'original_index' in df.columns: embeddings = embeddings[df['original_index'].values]
    return df, embeddings, total_counts

@st.cache_data
def get_projection(emb):
    if emb is None: return None
    st.info("🔄 Pre-computing Spatial Map (UMAP)...")
    return UMAP(n_components=2, random_state=42).fit_transform(emb)

# 2. REAL-TIME PROVIDER (Placeholder for Kafka/Spark)
def load_realtime_buffer():
    return None

# ================= APP LOGIC =================

st.sidebar.title("🌐 Hybrid Intelligence")

# 1. MODE SELECTION
mode = st.sidebar.selectbox("System Mode:", ["🛡️ Simulation (File Replay)", "⚡ Real-time (Kafka/Spark)"], index=0)

df_full = None
emb_full = None
topic_totals = {}
projection = None

if "Simulation" in mode:
    # --- SIMULATION MODE ---
    st.sidebar.subheader("🗂️ Data Sources")
    available = scan_files()
    default_selection = ["demo/data/results.parquet"] if "demo/data/results.parquet" in available else []
    selected = st.sidebar.multiselect("Select Files:", available, default=default_selection)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Event Evolution")
    evolution_threshold = st.sidebar.slider("Discovery Threshold %", 0.0, 1.0, 0.5)

    df_full, emb_full, topic_totals = load_simulation_data(selected)
    projection = get_projection(emb_full) if emb_full is not None else None

    # Controls
    st.sidebar.markdown("---")
    speed = st.sidebar.select_slider("Playback Speed (s)", options=[0.01, 0.1, 0.2, 0.5, 1.0], value=0.1)
    batch = st.sidebar.number_input("Batch Size", 1, 1000, 100)
    show_map = st.sidebar.toggle("Show Cluster Map", value=True)
    
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ START", use_container_width=True, type="primary"): st.session_state.sim_running = True
    if c2.button("⏸️ STOP", use_container_width=True): st.session_state.sim_running = False
    if st.sidebar.button("🔄 RESET", use_container_width=True):
        st.session_state.sim_index = 0
        st.session_state.sim_running = False
        st.rerun()

else:
    # --- REAL-TIME MODE ---
    st.sidebar.success("⚡ Connected to Real-time Stream")
    st.sidebar.markdown("Status: **Waiting for signals...**")
    st.sidebar.info("Integration Point: Kafka Consumer would inject dataframes here.")

# State Management
if 'sim_index' not in st.session_state: st.session_state.sim_index = 0
if 'sim_running' not in st.session_state: st.session_state.sim_running = False

# ================= MAIN UI =================

st.title("🛰️ Federated Intelligence Platform")

if "Simulation" in mode and df_full is None:
    st.warning("⚠️ Please select data sources in the sidebar.")
    st.stop()

if "Real-time" in mode:
    st.stop()

# --- SIMULATION RENDER LOGIC ---

vis_df = df_full.iloc[:st.session_state.sim_index].copy()

def process_evolving_topic(row, visible_counts):
    topic = row['final_topic']
    is_noise = topic in ['Unassigned', 'None', 'Discovery'] or "Noise" in str(topic)
    if is_noise: return topic, True
    total = topic_totals.get(topic, 1)
    so_far = visible_counts.get(topic, 0)
    if so_far < (evolution_threshold * total): return "Discovery", True
    return topic, False

if not vis_df.empty:
    v_counts = vis_df['final_topic'].value_counts().to_dict()
    vis_df[['display_topic', 'is_noise']] = vis_df.apply(
        lambda r: pd.Series(process_evolving_topic(r, v_counts)), axis=1
    )
    identified_df = vis_df[vis_df['is_noise'] == False]
else:
    vis_df['display_topic'] = None
    vis_df['is_noise'] = True
    identified_df = pd.DataFrame(columns=vis_df.columns)

prog_col, time_col = st.columns([4, 1])
with prog_col:
    st.progress(min(st.session_state.sim_index / max(len(df_full), 1), 1.0), text=f"Data Replay: {st.session_state.sim_index:,} posts")
with time_col:
    st.markdown(f"**Live Time:** `{datetime.now().strftime('%H:%M:%S')}`")

tab1, tab2 = st.tabs(["📊 Live Monitor", "📌 Identified Events & Insights"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Posts", f"{len(vis_df):,}")
    m2.metric("Identified Events", f"{identified_df['display_topic'].nunique() if not identified_df.empty else 0:,}")
    m3.metric("Active Sources", f"{vis_df['source'].nunique() if not vis_df.empty else 0:,}")
    pos_idx = "0.0%"
    if not vis_df.empty and 'sentiment' in vis_df.columns:
        p_val = vis_df['sentiment'].astype(str).str.lower().eq('positive').sum() / len(vis_df) * 100
        pos_idx = f"{p_val:.1f}%"
    m4.metric("Positivity Index", pos_idx)

    st.markdown("---")
    
    if show_map and projection is not None:
        col_f, col_m, col_s = st.columns([1, 1.2, 0.8])
    else:
        col_f, col_s = st.columns([1.6, 1]); col_m = None

    with col_f:
        st.subheader("📡 Live Stream")
        latest = vis_df.tail(15).iloc[::-1]
        if latest.empty: st.info("Press **START** to begin simulation.")
        else:
            for _, row in latest.iterrows():
                topic = row['display_topic']
                is_noise = row['is_noise']
                s_name = row['source']
                s_cls, s_tag_cls = get_source_class(s_name)
                type_cls = " item-highlight" if not is_noise else " item-noise"
                sent_html = get_sentiment_html(row.get('sentiment', 'neutral'))
                st.markdown(f"""
                <div class="live-feed-item {s_cls}{type_cls}">
                    <span class="time-stamp">{row['time'].strftime('%H:%M:%S')}</span>
                    <div style="margin-bottom: 4px;">
                        <span class="topic-tag {'noise-tag' if is_noise else ''}">{topic if not is_noise else 'DIỄN BIẾN MỚI'}</span>
                        <span class="source-tag {s_tag_cls}">{s_name}</span>
                    </div>
                    <div style="margin-top:8px; font-weight:600;">{row.get('title') if pd.notna(row.get('title')) else str(row.get('post_content',''))[:110]}</div>
                    <div style="margin-top:6px; font-size:0.8em;">{sent_html}</div>
                </div>
                """, unsafe_allow_html=True)

    if col_m:
        with col_m:
            st.subheader("🧩 Spatial Map")
            proj_visible = projection[:st.session_state.sim_index]
            plot_df = vis_df.copy()
            plot_df['x'] = proj_visible[:, 0]
            plot_df['y'] = proj_visible[:, 1]
            plot_df['map_topic'] = plot_df['display_topic'].apply(lambda x: "Scanning..." if x == "Discovery" else x)
            fig_map = px.scatter(
                plot_df, x='x', y='y', color='map_topic', template="plotly_dark",
                hover_name='final_topic', color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_map.update_layout(showlegend=False, height=500, margin=dict(l=0, r=0, t=0, b=0), xaxis={'visible': False}, yaxis={'visible': False})
            st.plotly_chart(fig_map, width='stretch', key=f"map_p_{st.session_state.sim_index}")

    with col_s:
        st.subheader("📊 Stats")
        if not identified_df.empty:
            t_data = identified_df['display_topic'].value_counts().head(8).reset_index()
            t_data.columns = ['Topic', 'Count']
            fig_t = px.pie(t_data, values='Count', names='Topic', hole=0.4, template="plotly_dark")
            fig_t.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_t, width='stretch', key=f"t_p_{st.session_state.sim_index}")
        else: st.caption("Waiting for significant events...")

        if not vis_df.empty:
            s_data = vis_df['source'].value_counts().reset_index()
            s_data.columns = ['Source', 'Count']
            fig_s = px.bar(s_data, x='Count', y='Source', orientation='h', template="plotly_dark", color='Source')
            fig_s.update_layout(showlegend=False, height=250, margin=dict(l=0,r=0,t=0,b=0), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_s, width='stretch', key=f"s_p_{st.session_state.sim_index}")

with tab2:
    st.subheader("📌 Intelligence Reports (Interactive)")
    if identified_df.empty:
        st.info("⌛ Analyzing patterns...")
    else:
        # Prepare Data
        summary_map = {}
        sent_map = {}
        for t in identified_df['display_topic'].unique():
            subset = identified_df[identified_df['display_topic'] == t]
            val = "N/A"
            if 'summary' in subset.columns:
                val = subset['summary'].dropna().iloc[0] if not subset['summary'].dropna().empty else "No summary"
            if val in ["N/A", "No summary"]:
                if 'title' in subset.columns: val = subset['title'].fillna(subset['post_content']).iloc[0]
            summary_map[t] = str(val)[:300]
            # Sentinel
            if 'sentiment' in subset.columns:
                counts = subset['sentiment'].str.lower().value_counts()
                sent_map[t] = counts.index[0].upper() if not counts.empty else "NEUTRAL"
            else: sent_map[t] = "N/A"

        # --- INTERACTIVE TABLE LOGIC ---
        
        # 1. Construct the Summary DataFrame
        trend_stats = identified_df.groupby('display_topic').agg(
            first_seen=('time', 'min'),
            last_seen=('time', 'max'),
            post_count=('display_topic', 'count')
        ).reset_index()
        trend_stats['Top Source'] = identified_df.groupby('display_topic')['source'].agg(lambda x: x.value_counts().index[0]).values
        trend_stats['Dominant Sentiment'] = trend_stats['display_topic'].map(sent_map)
        trend_stats['Quick Summary'] = trend_stats['display_topic'].map(summary_map)
        trend_stats = trend_stats.sort_values('post_count', ascending=False)
        
        # Interactive Dataframe
        st.markdown("### 📋 Event Intelligence Log")
        st.caption("👈 Click on a row to view full details (Drill-down)")
        
        event_df = trend_stats.rename(columns={'display_topic': 'Event Name', 'first_seen': 'First Detected', 'post_count': 'Volume'})
        
        selection = st.dataframe(
            event_df,
            use_container_width=True,
            column_config={
                "First Detected": st.column_config.DatetimeColumn(format="HH:mm:ss"),
                "last_seen": st.column_config.DatetimeColumn(format="HH:mm:ss"),
                "Volume": st.column_config.NumberColumn(format="%d posts"),
                "Dominant Sentiment": st.column_config.TextColumn(help="Majority sentiment"),
                "Quick Summary": st.column_config.TextColumn(width="medium")
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if selection and selection.selection.rows:
            sel_idx = selection.selection.rows[0]
            sel_topic = event_df.iloc[sel_idx]['Event Name']
            
            st.markdown("---")
            e_df = identified_df[identified_df['display_topic'] == selected_event]
            st.markdown(f"### 📂 Dossier: {selected_event}")
            
            # Close button (simulated by rerunning with empty selection - UI constraint, so we just provide context)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Volume", f"{len(e_df)} posts")
            k2.metric("First Seen", e_df['time'].min().strftime('%H:%M:%S'))
            k3.metric("Last Activity", e_df['time'].max().strftime('%H:%M:%S'))
            k4.metric("Sentiment", sent_map[selected_event])
            
            st.info(f"**Briefing**: {summary_map[selected_event]}")
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Top Sources**")
                s_metrics = e_df['source'].value_counts()
                st.bar_chart(s_metrics, color="#238636")
            with c_right:
                st.markdown("**Field Evidence (Top 3)**")
                for _, p in e_df.head(3).iterrows():
                    st.markdown(f"""
                    <div style="background-color: #161b22; padding: 10px; border-radius: 8px; margin-bottom: 5px; border: 1px solid #30363d;">
                        <span style="color: #8b949e; font-size: 0.8em;">[{p['source']}]</span>
                        <div style="font-size: 0.9em;">{p.get('title', p.get('post_content'))[:150]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("Select a row above to see details.")

if st.session_state.sim_running and "Simulation" in mode:
    if st.session_state.sim_index < len(df_full):
        time.sleep(speed)
        st.session_state.sim_index = min(st.session_state.sim_index + batch, len(df_full))
        st.rerun()
    else: st.session_state.sim_running = False; st.success("✅ Demo Completed!"); st.rerun()
