import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import plotly.express as px
from datetime import datetime
from sqlalchemy import create_engine, text
import math

DB_URL = "postgresql://user:password@localhost:5432/trend_db"

st.set_page_config(page_title="Phân tích Xu hướng", layout="wide", page_icon="🌐")

TAXONOMY_MAP = {
    "T1": "Khủng hoảng & Rủi ro",
    "T2": "Chính sách & Quản trị",
    "T3": "Rủi ro Uy tín",
    "T4": "Cơ hội Thị trường",
    "T5": "Văn hóa & Giải trí",
    "T6": "Vận hành & Dịch vụ",
    "T7": "Tin định kỳ"
}

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
        cursor: pointer;
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
    .topic-tag.analyzed {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .analyzed-card {
        border-left: 8px solid #7c3aed !important;
        background: linear-gradient(to right, #1e293b, #251052) !important;
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.15) inset;
    }
    .status-badge {
        font-size: 0.65rem;
        padding: 3px 8px;
        border-radius: 20px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-left: 10px;
        text-transform: uppercase;
    }
    .badge-verified {
        background: #7c3aed;
        color: white;
        border: 1px solid #a78bfa;
    }
    .badge-scanning {
        background: #334155;
        color: #94a3b8;
        border: 1px solid #475569;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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
    
    .post-card {
        background: #1e293b; 
        padding: 12px; 
        margin: 8px 0; 
        border-radius: 8px;
    }
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

def process_evolution(row, threshold):
    score = row['trend_score']
    if score < threshold:
        return pd.Series(["DIỄN BIẾN MỚI", True])
    return pd.Series([row['trend_name'], False])

def get_db_engine():
    return create_engine(DB_URL)

def load_realtime_data():
    engine = get_db_engine()
    query = text("SELECT * FROM detected_trends ORDER BY created_at DESC LIMIT 1000")
    return pd.read_sql(query, engine)



st.sidebar.title("🌐 Bảng điều khiển")
score_threshold = st.sidebar.slider("Ngưỡng điểm nóng", 0.0, 100.0, 30.0)
auto_refresh = st.sidebar.toggle("Tự động cập nhật", value=True)
refresh_rate = st.sidebar.select_slider("Tần suất (giây)", options=[2, 5, 10, 30], value=2)
sim_threshold = st.sidebar.slider("Độ tương đồng tối thiểu", 0.0, 1.0, 0.4, 0.05)

if st.sidebar.button("🗑️ Xóa dữ liệu"):
    with get_db_engine().begin() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends"))
    st.rerun()

@st.fragment(run_every=refresh_rate if auto_refresh else None)
def show_metrics():
    df_metrics = load_realtime_data()
    # Ensure columns exist even if empty
    if 'display_topic' not in df_metrics.columns:
        df_metrics['display_topic'] = ""
        df_metrics['is_noise'] = False

    if df_metrics.empty: 
        st.caption("⏳ Đang chờ dữ liệu...")
        return
    
    df_metrics[['display_topic', 'is_noise']] = df_metrics.apply(process_evolution, axis=1, threshold=score_threshold)
    id_df = df_metrics[df_metrics['is_noise'] == False]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cụm tin", f"{len(df_metrics):,}")
    m2.metric("Sự kiện", f"{id_df['trend_name'].nunique():,}")
    m3.metric("Điểm cao nhất", f"{df_metrics['trend_score'].max():.1f}")
    m4.metric("Thời gian", datetime.now().strftime("%H:%M:%S"))

show_metrics()

if 'last_df' not in st.session_state:
    st.session_state.last_df = load_realtime_data()

# Manual refresh for static components
if st.button("🔄 Cập nhật dữ liệu Chi tiết"):
    st.session_state.last_df = load_realtime_data()
    st.rerun()

df_full = st.session_state.last_df
if df_full.empty:
    st.warning("📡 Hệ thống đang khởi động... Dữ liệu sẽ tự động xuất hiện tại luồng Live.")
    # Do NOT stop here so fragments and tabs can render and poll for data

tab_live, tab_intel, tab_sys = st.tabs(["🚀 Luồng Live", "🧠 Chi tiết & Phân tích", "📈 Hiệu suất Hệ thống"])

# --- TAB 1: LIVE MONITOR ---
@st.fragment(run_every=refresh_rate if auto_refresh else None)
def show_tab_live():
    st.subheader("📡 Luồng tin tức thời gian thực")
    df_live = load_realtime_data()
    if df_live.empty:
        st.info("Đang chờ dữ liệu...")
        return
        
    df_live[['display_topic', 'is_noise']] = df_live.apply(process_evolution, axis=1, threshold=score_threshold)
    latest_trends = df_live.head(20)
    
    for _, row in latest_trends.iterrows():
        topic = row['display_topic']
        is_noise = row['is_noise']
        score = row['trend_score']
        
        raw_posts = row['representative_posts']
        posts = json.loads(raw_posts) if isinstance(raw_posts, str) else (raw_posts or [])
        main_post = posts[0] if posts else {'source': 'System', 'content': 'Không có nội dung'}
            
        s_name = normalize_source(main_post.get('source', 'Unknown'))
        s_cls, s_tag_cls = get_source_class(s_name)
        type_cls = " item-highlight" if not is_noise else " item-noise"
        
        # Check if analyzed (has summary)
        summary = row.get('summary')
        is_analyzed = summary and len(str(summary)) > 20 and str(summary) != "Waiting for analysis..."
        
        tag_class = "topic-tag"
        card_extra_cls = ""
        status_html = ""
        
        if is_noise: 
            tag_class += " noise-tag"
        elif is_analyzed: 
            tag_class += " analyzed"
            card_extra_cls = " analyzed-card"
            status_html = f'<span class="status-badge badge-verified">🤖 ANALYZED</span>'
        else:
            status_html = f'<span class="status-badge badge-scanning">🔍 SCANNING...</span>'
        
        icon = "📌 " if not is_analyzed and not is_noise else ""

        st.markdown(f"""<div class="live-feed-item {s_cls}{type_cls}{card_extra_cls}">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<span class="{tag_class}">{icon}{topic}</span>
{status_html}
</div>
<span class="time-stamp">ĐIỂM: {score:.1f}</span>
</div>
<div><span class="source-tag {s_tag_cls}">{s_name}</span></div>
<div class="post-content">{main_post['content'][:250]}...</div>
</div>""", unsafe_allow_html=True)

with tab_live:
    show_tab_live()

with tab_intel:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🔍 Chi tiết Sự kiện")
        
        # Prepare data
        if not df_full.empty:
            df_full[['display_topic', 'is_noise']] = df_full.apply(process_evolution, axis=1, threshold=score_threshold)
            identified_df = df_full[df_full['is_noise'] == False]
        else:
            identified_df = pd.DataFrame(columns=df_full.columns)
        
        show_all = st.checkbox("Xem cả các diễn biến mới (Cụm tin chưa đạt ngưỡng)")
        
        target_df = df_full if show_all else identified_df
        
        # Prepare filtered data for counts
        summaries = target_df['summary'].fillna('')
        analyzed_mask = (summaries.str.len() > 20) & (summaries != "Waiting for analysis...")
        count_all = len(target_df)
        count_analyzed = analyzed_mask.sum()
        count_pending = count_all - count_analyzed

        # Add Status Filter with Counts
        status_filter = st.radio(
            "Lọc theo trạng thái:",
            options=[
                f"Tất cả ({count_all})", 
                f"✨ Đã xử lý ({count_analyzed})", 
                f"🔍 Chờ xử lý ({count_pending})"
            ],
            horizontal=True,
            index=0
        )
        
        if "Đã xử lý" in status_filter:
            target_df = target_df[analyzed_mask]
        elif "Chờ xử lý" in status_filter:
            target_df = target_df[~analyzed_mask]
        
        if not target_df.empty:
            # Create labels with status icons
            def get_selector_label(row):
                summary = row.get('summary')
                is_analyzed = summary and len(str(summary)) > 20 and str(summary) != "Waiting for analysis..."
                icon = "✨" if is_analyzed else "🔍"
                return f"{icon} {row['trend_name']}"

            # We need to keep track of the mapping from label to original name
            label_to_name = {get_selector_label(r): r['trend_name'] for _, r in target_df.iterrows()}
            options = list(label_to_name.keys())
            
            selected_label = st.selectbox(
                "Chọn sự kiện hoặc cụm tin:",
                options=options,
                index=0
            )
            selected_trend = label_to_name[selected_label]
            
            trend_data = df_full[df_full['trend_name'] == selected_trend].iloc[0]
            score = trend_data['trend_score']
            is_event = score >= score_threshold
            
            # Map Category
            cat_code = trend_data.get('category', 'N/A') or 'N/A'
            cat_display = TAXONOMY_MAP.get(cat_code, cat_code)
            
            st.markdown(f"### {trend_data['trend_name']}")
            st.markdown(f"**Loại hình:** {cat_display} ({cat_code})")
            
            # Status & Reasoning Alert
            if not is_event:
                st.warning(f"**Trạng thái:** Đang theo dõi (Chưa đạt ngưỡng sự kiện)\n\n**Lý do:** Điểm hiện tại ({score:.1f}) thấp hơn Ngưỡng điểm nóng ({score_threshold:.1f}). Cần thêm bài viết hoặc tương tác để trở thành Sự kiện chính thức.")
            else:
                st.success(f"**Trạng thái:** Sự kiện chính thức (Đã đạt ngưỡng {score_threshold:.1f})")

            # Metrics Row (3 columns - Category is shown above)
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Điểm số", f"{score:.1f}")
            cm2.metric("Bài viết", f"{trend_data.get('post_count', 0):,}")
            cm3.metric("Cảm xúc", trend_data.get('sentiment', 'N/A') or 'N/A')
            
            st.markdown("---")
            
            summary = trend_data.get('summary', '')
            advice_state = trend_data.get('advice_state', '')
            advice_biz = trend_data.get('advice_business', '')
            reasoning = trend_data.get('reasoning', '')
            
            has_summary = summary and len(str(summary)) > 20 and str(summary) != "Waiting for analysis..."
            has_advice = (advice_state and str(advice_state).strip() and str(advice_state) != 'N/A') or \
                         (advice_biz and str(advice_biz).strip() and str(advice_biz) != 'N/A')
            
            if has_summary or has_advice:
                st.markdown("### 🤖 Phân tích AI")
                
                # Main Summary Box
                if has_summary:
                    with st.container():
                        st.markdown("**📋 Tóm tắt sự kiện:**")
                        # Replace \n with markdown line breaks for proper rendering
                        formatted_summary = str(summary).replace('\n', '  \n')
                        st.markdown(formatted_summary)
                
                st.markdown("")  # Spacer
                
                # Advice Section in columns
                if has_advice:
                    adv_col1, adv_col2 = st.columns(2)
                    
                    with adv_col1:
                        if advice_state and str(advice_state).strip() and str(advice_state) != 'N/A':
                            st.info(f"**💡 Khuyến nghị cho Nhà nước:**\n\n{advice_state}")
                    
                    with adv_col2:
                        if advice_biz and str(advice_biz).strip() and str(advice_biz) != 'N/A':
                            st.success(f"**💼 Khuyến nghị cho Doanh nghiệp:**\n\n{advice_biz}")
                
                # AI Reasoning (Expandable)
                if reasoning and str(reasoning) != 'N/A' and str(reasoning).strip():
                    with st.expander("🧐 Xem lý do phân loại từ AI"):
                        st.caption(reasoning)
            else:
                st.warning("⏳ Đang chờ phân tích từ AI...")
            
            st.markdown("#### 📰 Các bài viết liên quan")
            raw_posts = trend_data.get('representative_posts', '[]')
            all_posts = json.loads(raw_posts) if isinstance(raw_posts, str) else (raw_posts or [])
            
            # 1. Filter all posts by similarity threshold
            filtered_posts = []
            if all_posts:
                for post in all_posts:
                    sim_score = post.get('similarity', post.get('score', 0))
                    if sim_score >= sim_threshold:
                        filtered_posts.append(post)

            # 2. Pagination Logic
            PAGE_SIZE = 5
            total_items = len(filtered_posts)
            
            if total_items > 0:
                total_pages = math.ceil(total_items / PAGE_SIZE)
                
                # Use a unique key for the page selection based on the selected trend
                page_key = f"page_{selected_trend.replace(' ', '_')}"
                if page_key not in st.session_state:
                    st.session_state[page_key] = 1
                
                # Page selection UI
                p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                with p_col2:
                    current_page = st.number_input(
                        f"Trang (Tổng {total_pages})", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=st.session_state[page_key],
                        key=page_key
                    )
                
                st.caption(f"Hiển thị {min((current_page-1)*PAGE_SIZE + 1, total_items)} - {min(current_page*PAGE_SIZE, total_items)} / {total_items} bài viết")

                # 3. Display current page
                start_idx = (current_page - 1) * PAGE_SIZE
                end_idx = start_idx + PAGE_SIZE
                
                for post in filtered_posts[start_idx:end_idx]:
                    source = normalize_source(post.get('source', 'Unknown'))
                    content = post.get('content', '')[:500]
                    sim_score = post.get('similarity', post.get('score', 0))
                    sim_display = f"{float(sim_score):.2f}" if sim_score and float(sim_score) > 0 else "N/A"
                    time_str = str(post.get('time', ''))[:19]
                    border_color = '#3b82f6' if 'facebook' in source.lower() else '#f97316'
                    
                    st.markdown(f"""<div style="background: #1e293b; padding: 15px; margin: 10px 0; border-radius: 12px; border-left: 5px solid {border_color};">
<div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; font-weight: 600;">
<span>{source} • {time_str}</span>
<span>Độ tương đồng: {sim_display}</span>
</div>
<div style="margin-top: 10px; color: #e2e8f0; line-height: 1.5;">{content}...</div>
</div>""", unsafe_allow_html=True)
            else:
                if not all_posts:
                    st.warning("Không có bài viết nào trong cụm này.")
                else:
                    st.warning("Không có bài viết nào thỏa mãn ngưỡng tương đồng.")
    
    with col_right:
        st.subheader("📊 Thống kê")
        
        if 'category' in identified_df.columns and not identified_df.empty:
            cat_counts = identified_df['category'].value_counts().reset_index()
            cat_counts.columns = ['Mã', 'Số lượng']
            
            # Use TAXONOMY_MAP if available, otherwise fallback to code
            if 'TAXONOMY_MAP' in globals():
                cat_counts['Loại hình'] = cat_counts['Mã'].apply(lambda x: TAXONOMY_MAP.get(x, x))
            else:
                 cat_counts['Loại hình'] = cat_counts['Mã']
            
            fig_t = px.pie(cat_counts, values='Số lượng', names='Loại hình', hole=0.5, 
                           template="plotly_dark", title="Tỷ lệ Phân loại Sự kiện")
            fig_t.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0), showlegend=True)
            st.plotly_chart(fig_t, width="stretch")

        # 2. Topic Type Bar (Mapped)
        type_counts = df_full['topic_type'].value_counts().reset_index()
        type_counts.columns = ['Loại', 'Số lượng']
        
        # Also show categorized distribution if available
        if 'category' in df_full.columns:
            cat_counts = df_full['category'].value_counts().reset_index()
            cat_counts.columns = ['Mã', 'Số lượng']
            cat_counts['Loại hình'] = cat_counts['Mã'].apply(lambda x: TAXONOMY_MAP.get(x, x))
            
            fig_s = px.bar(cat_counts, x='Số lượng', y='Loại hình', orientation='h', 
                           template="plotly_dark", color='Loại hình', title="Phân loại theo Mục tiêu")
            fig_s.update_layout(showlegend=False, height=350, margin=dict(l=0,r=0,t=40,b=0), 
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_s, width="stretch")

with tab_sys:
    @st.fragment(run_every=refresh_rate if auto_refresh else None)
    def show_system_stats():
        st.subheader("⚙️ Chỉ số Vận hành Hệ thống")
        df_sys = load_realtime_data()
        
        if df_sys.empty:
            st.info("Chưa có dữ liệu hệ thống.")
            return

        total_posts = df_sys['post_count'].sum()
        total_trends = len(df_sys)
        active_trends = len(df_sys[df_sys['trend_score'] >= score_threshold])
        
        # Row 1: Big Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Tổng bài viết xử lý", f"{total_posts:,}", delta=None)
            st.caption("Tổng số tin bài đã được gán vào các cụm")
        with c2:
            st.metric("Tổng số cụm tin", f"{total_trends:,}")
            st.caption("Các nhóm tin có sự tương đồng nội dung")
        with c3:
            st.metric("Sự kiện tiêu điểm", f"{active_trends:,}")
            st.caption(f"Cụm tin vượt ngưỡng {score_threshold}")

        st.markdown("---")

        # Row 2: Charts
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("#### 📊 Phân bổ tin bài theo Chủ đề")
            fig_bar = px.bar(
                df_sys.sort_values('post_count', ascending=False).head(10),
                x='post_count',
                y='trend_name',
                orientation='h',
                color='trend_score',
                template="plotly_dark",
                labels={'post_count': 'Số lượng bài', 'trend_name': 'Chủ đề'},
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_bar, width="stretch")

        with col_plot2:
            st.markdown("#### ⏳ Trạng thái Xử lý (LLM)")
            summaries = df_sys['summary'].fillna('')
            analyzed = (summaries.str.len() > 20) & (summaries != "Waiting for analysis...")
            
            status_df = pd.DataFrame({
                'Trạng thái': ['Đã phân tích (Deep)', 'Chờ xử lý (Fast Path)'],
                'Số lượng': [analyzed.sum(), (~analyzed).sum()]
            })
            
            fig_pie = px.pie(
                status_df, 
                values='Số lượng', 
                names='Trạng thái',
                color='Trạng thái',
                color_discrete_map={'Đã phân tích (Deep)': '#7c3aed', 'Chờ xử lý (Fast Path)': '#334155'},
                hole=0.4,
                template="plotly_dark"
            )
            fig_pie.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_pie, width="stretch")

        # System Health Note
        st.success(f"✅ Hệ thống đang chạy ở chế độ **Real-time Injection** (Simulation).")
        st.info(f"💡 Tốc độ nạp dữ liệu: ~1 bài/3 giây. Tự động cập nhật mỗi {refresh_rate} giây.")

    show_system_stats()

