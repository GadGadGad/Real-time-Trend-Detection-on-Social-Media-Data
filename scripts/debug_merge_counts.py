import pandas as pd
import glob
import os
import json

DATA_DIR = "summarized_data"
RAW_FB_DIR = "crawlers/facebook"

print("--- DEBUGGING FACEBOOK DATA COUNTS ---")

# 1. Check RAW JSONs
fb_files = glob.glob(os.path.join(RAW_FB_DIR, "*.json"))
# Also check typical kaggle path if local not found
if not fb_files:
    fb_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/facebook/*.json")

raw_count = 0
for f in fb_files:
    try:
        with open(f, 'r') as jf:
            data = json.load(jf)
            raw_count += len(data)
    except: pass
print(f"RAW JSON Count: {raw_count}")

# 2. Check Target Files (Input for Merge)
target_path = os.path.join(DATA_DIR, "all_facebook_summarized.csv")
if not os.path.exists(target_path):
    # Try alternate name
    target_path = os.path.join(DATA_DIR, "facebook_summarized_content.csv")

if os.path.exists(target_path):
    try:
        df_target = pd.read_csv(target_path)
        print(f"TARGET CSV ({os.path.basename(target_path)}) Count: {len(df_target)}")
    except Exception as e:
        print(f"TARGET CSV Error: {e}")
else:
    print("TARGET CSV: Not Found")

# 3. Check Refined Output (Input for Merge)
refined_path = os.path.join(DATA_DIR, "facebook_refined.csv")
if os.path.exists(refined_path):
    try:
        df_refined = pd.read_csv(refined_path, on_bad_lines='skip', engine='python')
        print(f"REFINED CSV Count: {len(df_refined)}")
        
        # Check validity
        if 'refined_summary' in df_refined.columns:
            valid = df_refined['refined_summary'].notna() & (df_refined['refined_summary'] != "")
            print(f"  -> Valid Refined Summaries: {valid.sum()}")
        elif 'refined' in df_refined.columns:
             valid = df_refined['refined'].notna() & (df_refined['refined'] != "")
             print(f"  -> Valid Refined Summaries (col='refined'): {valid.sum()}")
        else:
            print("  -> Column 'refined_summary' not found in refined csv")
            
        # Check Index
        if 'index' in df_refined.columns:
             print(f"  -> Index range: Min={df_refined['index'].min()}, Max={df_refined['index'].max()}")
        else:
             print("  -> 'index' column MISSING in refined csv")

    except Exception as e:
        print(f"REFINED CSV Error: {e}")
else:
    print("REFINED CSV: Not Found")

# 4. Check Final Merged Output
merged_path = os.path.join(DATA_DIR, "facebook_merged.csv")
if os.path.exists(merged_path):
    try:
        df_merged = pd.read_csv(merged_path)
        print(f"MERGED CSV Count: {len(df_merged)}")
        
        # Check how many have refined_summary populated
        if 'refined_summary' in df_merged.columns:
            filled = df_merged['refined_summary'].notna() & (df_merged['refined_summary'] != "")
            print(f"  -> Merged with Refined Summary: {filled.sum()}")
    except Exception as e:
        print(f"MERGED CSV Error: {e}")
else:
    print("MERGED CSV: Not Found")
