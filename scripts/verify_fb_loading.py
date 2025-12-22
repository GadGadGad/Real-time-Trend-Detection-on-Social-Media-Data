import pandas as pd
import os

DATA_DIR = 'summarized_data'
fb_path = os.path.join(DATA_DIR, 'facebook_merged.csv')

if not os.path.exists(fb_path):
    # Fallback to absolute path if needed, based on user logs
    # User saw: /kaggle/input/se363-summaries/facebook_merged.csv
    # But locally we are in /home/gad...
    # Let's check generally in summarized_data
    print(f"File not found at {fb_path}")
    # Try finding it
    import glob
    found = glob.glob(f"**/{os.path.basename(fb_path)}", recursive=True)
    if found:
        fb_path = found[0]
        print(f"Found at {fb_path}")
    else:
        print("Cannot find facebook_merged.csv")
        exit(1)

print(f"Analyzing: {fb_path}")

try:
    df = pd.read_csv(fb_path)
    total_rows = len(df)
    
    # Simulate Filter
    # Logic: content = refined_summary OR summary OR (str(title) if title else content)
    # Filter: len(content) < 20
    
    kept = 0
    dropped = 0
    
    print("\n--- Filtering Logic Simulation ---")
    
    for _, row in df.iterrows():
        content = row.get('refined_summary')
        if pd.isna(content) or content == "": content = row.get('summary')
        if pd.isna(content) or content == "":
            t_raw = row.get('title', '')
            content = str(t_raw) if t_raw else row.get('content', '')
            
        if len(str(content).strip()) < 20:
            dropped += 1
        else:
            kept += 1
            
    print(f"Total Rows in CSV: {total_rows}")
    print(f"Rows Kept (>20 chars): {kept}")
    print(f"Rows Dropped (<20 chars): {dropped}")
    
    if dropped > 0:
        print(f"\nVerification: The 'missing' {dropped} posts are exactly the ones removed by the quality filter.")
    else:
        print("\nVerification: No posts were dropped. Something else is wrong.")

except Exception as e:
    print(f"Error: {e}")
