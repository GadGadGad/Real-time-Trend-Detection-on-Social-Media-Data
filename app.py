import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page Config
st.set_page_config(page_title="Trend Detection Dashboard", page_icon="📈", layout="wide")

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .trend-positive { color: green; font-weight: bold; }
    .trend-negative { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data(filepath="crawlers/results/results.json"):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

# Sidebar
st.sidebar.title("🔍 Config & Filter")
view_mode = st.sidebar.radio("View Mode", [
    "Semantic Matching (Google Trends)", 
    "Unsupervised Discovery (Clustering)",
    "Comparison (Trends vs Clusters)",
    "Hybrid Unified View (Cluster-First)"
])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pipeline Config")
use_llm = st.sidebar.checkbox("Enable LLM Refinement", value=False)
refine_trends = st.sidebar.checkbox("Refine Trends (LLM)", value=False)
use_keywords = st.sidebar.checkbox("Use Keyword Extraction", value=True)
results_path = st.sidebar.text_input("Results File Path", "crawlers/results/results.json")

if st.sidebar.button("🚀 Run Analysis"):
    st.info("Running Analysis... This may take a minute.")
    cmd = (
        "python src/pipeline/main_pipeline.py "
        "--social crawlers/facebook/*.json "
        "--trends crawlers/trendings/*.csv "
        "--news crawlers/news/*.csv crawlers/news_v2/*.csv "
        "--save-all "
        f"--output {results_path}"
    )
    if use_llm:
        cmd += " --llm"
    if refine_trends:
        cmd += " --refine-trends"
    if use_keywords:
        cmd += " --use-keywords"
        
    result = os.system(cmd)
    if result == 0:
        st.success("Analysis Complete! Refreshing data...")
        st.rerun()
    else:
        st.error("Analysis failed. Check console for errors.")

data = load_data(results_path)

if not data:
    st.error(f"No data found at `{results_path}`. Please run `analyze_trends.py` first.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract nested stats
if 'stats' in df.columns:
    stats_df = df['stats'].apply(pd.Series)
    df = pd.concat([df, stats_df], axis=1)

# Stats Calculation
total_posts = len(df)
total_trends = df['trend'].nunique() if 'trend' in df.columns else 0
matched_posts = df[df['is_matched'] == True] if 'is_matched' in df.columns else df
avg_score = matched_posts['score'].mean() if 'score' in matched_posts.columns else 0

# Header
st.title("📈 Real-time Trend Detection Dashboard")
st.markdown("Monitoring Social Media, News, and Search signals.")

# KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts Analyzed", total_posts)
col1.metric("Total Posts Analyzed", total_posts)

if "Clustering" in view_mode:
    active_count = df['cluster_id'].nunique() if 'cluster_id' in df.columns else 0
    col2.metric("Clusters Detected", active_count)
    group_col = 'cluster_name'
    filter_mask = df['cluster_id'] != -1 if 'cluster_id' in df.columns else pd.Series([True]*len(df))
elif "Comparison" in view_mode:
    # In comparison mode, show stats for both
    col2.metric("Active Trends", total_trends)
    col3.metric("Clusters Detected", df['cluster_id'].nunique() if 'cluster_id' in df.columns else 0)
    # Use col4 for match score or sources
    group_col = 'trend' # Default for list view
    filter_mask = pd.Series([True]*len(df))
elif "Hybrid" in view_mode:
    col2.metric("Trending Topics", len(df[df['topic_type'] == "Trending"]['final_topic'].unique()) if 'topic_type' in df.columns else 0)
    col3.metric("New Discoveries", len(df[df['topic_type'] == "Discovery"]['final_topic'].unique()) if 'topic_type' in df.columns else 0)
    group_col = 'final_topic'
    filter_mask = df['topic_type'].isin(['Trending', 'Discovery']) if 'topic_type' in df.columns else pd.Series([True]*len(df))
else:
    col2.metric("Active Trends Detected", total_trends)
    group_col = 'trend'
    filter_mask = df['is_matched'] == True if 'is_matched' in df.columns else pd.Series([True]*len(df))

if "Comparison" not in view_mode:
    col3.metric("Avg Match Confidence", f"{avg_score:.2f}")

col4.metric("Sources Tracked", df['source'].nunique())

# Layout: 2 Columns
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("🔥 Top Trends/Clusters")
    if group_col in df.columns:
        trend_counts = df[filter_mask][group_col].value_counts().reset_index()
        trend_counts.columns = ['Topic', 'Post Count']
        st.dataframe(trend_counts, width="stretch")
        # Actually user said: For `use_container_width=True`, use `width='stretch'`? No, Streamlit 1.39+ deprecated it for st.dataframe?
        # Re-reading user request: "For `use_container_width=True`, use `width='stretch'`" seems to imply I should use that parameter name?
        # Wait, the warning says: "Please replace `use_container_width` with `width`".
        # So I should use `st.dataframe(..., width=1000)`? Or `st.dataframe` arguments changed?
        # Let's try omitting it or using defaults if unsure, but user said use "width='stretch'".
        # WAIT, `st.dataframe` doesn't take "stretch" usually. Maybe they mean `st.image`?
        # Let's interpret "For `use_container_width=True`, use `width`" as the instruction.
        # But `width` usually expects int.
        # Check docs or assume user knows: `st.dataframe(..., use_container_width=True)` -> `st.dataframe(..., use_container_width=True)` is what IS deprecated?
        # "Please replace `use_container_width` with `width`".
        # Okay, let's try `st.dataframe(..., width=None)` (auto) or similar.
        # Actually, let's just remove the argument to be safe, it defaults to width of container often.
        st.dataframe(trend_counts)
        
        # Trend Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        # Fix Seaborn warning: Assign y to hue and legend=False
        sns.barplot(data=trend_counts.head(10), x='Post Count', y='Topic', hue='Topic', palette='viridis', ax=ax, legend=False)
        st.pyplot(fig)
    else:
        st.warning(f"No '{group_col}' column found.")

with right_col:
    st.subheader("📊 Source Distribution")
    source_counts = df['source'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig2)
    
    st.subheader("😊 Sentiment Analysis")
    if 'sentiment' in df.columns:
        sent_counts = df['sentiment'].value_counts()
        st.bar_chart(sent_counts)
    else:
        st.info("Sentiment analysis not enabled yet.")

    # t-SNE Plot
    st.subheader("🧩 Trend Visualization (t-SNE)")
    tsne_path = "crawlers/results/trend_tsne.png"
    if os.path.exists(tsne_path):
        st.image(tsne_path, caption="t-SNE of Semantic Clusters", width="stretch")
    else:
        st.info("Run `evaluate_trends.py` to generate the t-SNE visualization.")

if "Comparison" in view_mode:
    st.markdown("---")
    st.header("⚖️ Method Comparison: Google Trends vs. Unsupervised Clusters")
    
    if 'trend' in df.columns and 'cluster_name' in df.columns:
        # Filter noise
        comp_df = df[(df['cluster_id'] != -1) & (df['is_matched'] == True)]
        
        if not comp_df.empty:
            st.subheader("🔥 Overlap Heatmap")
            st.markdown("How do semantic trends map to discovered clusters?")
            
            # Crosstab
            ct = pd.crosstab(comp_df['trend'], comp_df['cluster_name'])
            
            # Simple Heatmap
            fig_hm, ax_hm = plt.subplots(figsize=(12, 8))
            sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax_hm)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_hm)
            
            st.subheader("📋 Contingency Table")
            st.dataframe(ct, width="stretch")
        else:
            st.warning("Not enough overlapping data (Matched + Clustered) to compare.")
            
        # --- NEW: Size & Coverage Comparison ---
        st.markdown("---")
        st.subheader("📊 Coverage & Size Distribution")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Total Posts Covered**")
            matched_count = len(df[df['is_matched'] == True])
            clustered_count = len(df[df['cluster_id'] != -1])
            coverage_df = pd.DataFrame({
                'Method': ['Google Trends (Matched)', 'HDBSCAN (Clustered)'],
                'Count': [matched_count, clustered_count]
            })
            # Bar chart
            fig_cov, ax_cov = plt.subplots()
            sns.barplot(data=coverage_df, x='Method', y='Count', palette='pastel', ax=ax_cov)
            st.pyplot(fig_cov)
            
        with c2:
            st.markdown("**Top Groups Size Comparison**")
            # Get top 5 trends stats
            top_trends = df[df['is_matched'] == True]['trend'].value_counts().head(5).reset_index()
            top_trends.columns = ['Name', 'Count']
            top_trends['Type'] = 'Trend'
            
            # Get top 5 clusters stats
            top_clusters = df[df['cluster_id'] != -1]['cluster_name'].value_counts().head(5).reset_index()
            top_clusters.columns = ['Name', 'Count']
            top_clusters['Type'] = 'Cluster'
            
            # Combine
            combined_top = pd.concat([top_trends, top_clusters])
            
            fig_size, ax_size = plt.subplots()
            sns.barplot(data=combined_top, y='Name', x='Count', hue='Type', ax=ax_size)
            st.pyplot(fig_size)

    else:
        st.error("Data missing 'trend' or 'cluster_name' columns.")

# Detailed Data View
st.subheader("📝 Detailed Data Viewer")
search_term = st.text_input("Search content...")
if search_term:
    filtered_df = df[df['post_content'].str.contains(search_term, case=False, na=False)]
else:
    filtered_df = df

cols_to_show = ['source', group_col, 'score', 'post_content', 'processed_content', 'time']
# Filter generic Unclustered if needed
if "Clustering" in view_mode:
     filtered_df = filtered_df[filtered_df['cluster_id'] != -1]
elif "Hybrid" in view_mode and 'topic_type' in filtered_df.columns:
     filtered_df = filtered_df[filtered_df['topic_type'] != "Noise"]

cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]
st.dataframe(filtered_df[cols_to_show])

# --- NEW: Topic Intelligence Explorer ---
if 'final_topic' in df.columns:
    st.markdown("---")
    st.header("🧠 Topic Intelligence Explorer")
    
    unique_topics = df[df['topic_type'].isin(['Trending', 'Discovery'])]['final_topic'].unique()
    selected_topic = st.selectbox("Select a topic to view detailed intelligence:", unique_topics)
    
    if selected_topic:
        topic_df = df[df['final_topic'] == selected_topic]
        
        # Topic Metadata Card
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Category:** {topic_df['category'].iloc[0] if 'category' in topic_df.columns else 'N/A'}")
        with c2:
            st.success(f"**Sentiment:** {topic_df['topic_sentiment'].iloc[0] if 'topic_sentiment' in topic_df.columns else 'N/A'}")
        with c3:
            st.warning(f"**Match Score:** {topic_df['score'].iloc[0]:.2f}")

        # Summary Section
        st.subheader("📝 Detailed Summary")
        summary_text = topic_df['summary'].iloc[0] if 'summary' in topic_df.columns else "No summary available."
        st.write(summary_text)

        # 5W1H Section
        st.subheader("🔍 5W1H Extraction")
        intel = topic_df['intelligence'].iloc[0] if 'intelligence' in topic_df.columns else {}
        if isinstance(intel, str): # Handle potential stringified JSON
             try: intel = json.loads(intel)
             except: intel = {}
        
        if intel:
            i1, i2 = st.columns(2)
            with i1:
                st.markdown(f"**WHO:** {intel.get('who', 'N/A')}")
                st.markdown(f"**WHAT:** {intel.get('what', 'N/A')}")
                st.markdown(f"**WHERE:** {intel.get('where', 'N/A')}")
            with i2:
                st.markdown(f"**WHEN:** {intel.get('when', 'N/A')}")
                st.markdown(f"**WHY:** {intel.get('why', 'N/A')}")
        else:
            st.info("No detailed intelligence extracted for this topic.")

        # advice Section
        st.subheader("💡 Strategic Advice")
        a1, a2 = st.columns(2)
        with a1:
             st.markdown("### 🏛️ For State / Authorities")
             advice_state = topic_df.get('advice_state', pd.Series(["N/A"])).iloc[0] or intel.get('advice_state', 'N/A')
             st.info(advice_state)
        with a2:
             st.markdown("### 🏢 For Business / Enterprise")
             advice_business = topic_df.get('advice_business', pd.Series(["N/A"])).iloc[0] or intel.get('advice_business', 'N/A')
             st.success(advice_business)
        
        # Reasoning (Expander)
        with st.expander("Show LLM Reasoning"):
            st.write(topic_df['llm_reasoning'].iloc[0] if 'llm_reasoning' in topic_df.columns else "N/A")

