"""
Investigation Script: Analyze why multiple clusters matched to the same trend
"""
import pandas as pd
import json
import os

# Load results (adjust path as needed)
RESULTS_PATH = "demo_data/results.parquet"  # or your output path

def investigate_trend(df, trend_name):
    """Deep dive into a specific trend to see why multiple reasonings exist."""
    
    print(f"\n{'='*80}")
    print(f"🔍 INVESTIGATING TREND: {trend_name}")
    print(f"{'='*80}")
    
    # Filter posts matching this trend
    matches = df[df['final_topic'].str.contains(trend_name, case=False, na=False)]
    
    print(f"\n📊 Total posts matched: {len(matches)}")
    
    # Check for concatenated reasonings (split by " | ")
    all_reasonings = []
    for r in matches['llm_reasoning'].dropna().unique():
        if ' | ' in str(r):
            parts = str(r).split(' | ')
            all_reasonings.extend(parts)
            print(f"\n⚠️  MERGED REASONING DETECTED ({len(parts)} parts):")
            for i, part in enumerate(parts, 1):
                print(f"   [{i}] {part[:150]}...")
        else:
            all_reasonings.append(str(r))
    
    print(f"\n📊 Unique reasoning components: {len(set(all_reasonings))}")
    
    # Group by reasoning to see which posts belong to which
    print(f"\n📰 SAMPLE POSTS PER REASONING:")
    for reasoning in matches['llm_reasoning'].unique():
        subset = matches[matches['llm_reasoning'] == reasoning]
        print(f"\n   --- Reasoning: {reasoning[:80]}... ---")
        print(f"   Post count: {len(subset)}")
        for _, row in subset.head(3).iterrows():
            content = row.get('content', row.get('post_content', ''))[:150]
            print(f"      • {content}...")
    
    # Check if there are distinct cluster patterns
    if 'cluster_id' in matches.columns or 'cluster_name' in matches.columns:
        cluster_col = 'cluster_id' if 'cluster_id' in matches.columns else 'cluster_name'
        print(f"\n🧩 CLUSTERS MERGED INTO THIS TREND:")
        for cluster in matches[cluster_col].unique():
            cluster_posts = matches[matches[cluster_col] == cluster]
            print(f"   • Cluster {cluster}: {len(cluster_posts)} posts")
    
    return matches

def list_multi_reasoning_trends(df):
    """Find all trends that have multiple reasonings (potential merge issues)."""
    
    # Group by final_topic and count unique reasonings
    reasoning_counts = df.groupby('final_topic')['llm_reasoning'].apply(
        lambda x: len(set(str(r) for r in x if pd.notna(r)))
    )
    
    multi = reasoning_counts[reasoning_counts > 1].sort_values(ascending=False)
    
    print(f"\n{'='*80}")
    print(f"⚠️  TRENDS WITH MULTIPLE REASONINGS (Potential Incorrect Merges)")
    print(f"{'='*80}")
    
    for topic, count in multi.head(20).items():
        posts = len(df[df['final_topic'] == topic])
        print(f"   [{count} reasonings] {topic[:60]}... ({posts} posts)")
    
    return multi

# Main execution
if __name__ == "__main__":
    # Load data
    if os.path.exists(RESULTS_PATH):
        df = pd.read_parquet(RESULTS_PATH)
        print(f"✅ Loaded {len(df)} results from {RESULTS_PATH}")
    else:
        print(f"❌ File not found: {RESULTS_PATH}")
        print("   Update RESULTS_PATH to your actual results file.")
        exit(1)
    
    # List all problematic trends
    multi = list_multi_reasoning_trends(df)
    
    # Deep dive into specific trend
    target_trend = "video đoàn văn sáng"  # Change this to investigate other trends
    investigate_trend(df, target_trend)
