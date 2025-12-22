import json
import os

nb_path = 'notebooks/analysis-playground_v2.ipynb'
backup_path = 'notebooks/analysis-playground_v2.backup.ipynb'

if not os.path.exists(nb_path):
    print(f"Error: {nb_path} does not exist.")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Create backup
with open(backup_path, 'w') as f:
    json.dump(nb, f, indent=4)

print(f"Backed up notebook to {backup_path}")

# Logic to find and replace
news_cell_found = False

new_news_code = """# ==========================================
# LOAD MERGED NEWS SUMMARIES (UPDATED)
# ==========================================
import pandas as pd
import os

# Adjust path: 'summarized_data' is in project root, notebook is in 'notebooks/'
# We try both '../summarized_data' (if running from notebooks dir) and 'summarized_data' (if root)
DATA_DIR = '../summarized_data'
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'summarized_data'

NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']
dfs = []

print(f"Loading data from: {os.path.abspath(DATA_DIR)}")

for source in NEWS_SOURCES:
    merged_path = os.path.join(DATA_DIR, f'{source}_merged.csv')
    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path)
        # Ensure 'summary' column is refined if available
        # The notebook pipeline likely uses 'summary' column for downstream analysis
        if 'refined_summary' in df.columns:
            # Fill NaN or empty refined summaries with the original text or summary
            # Fallback chain: refined_summary -> summary -> text -> empty
            fallback = df['summary'] if 'summary' in df.columns else (df['text'] if 'text' in df.columns else '')
            df['summary'] = df['refined_summary'].fillna(fallback)
        
        # Add metadata if needed (e.g. source)
        if 'source' not in df.columns:
            df['source'] = source
            
        dfs.append(df)
        print(f'{source}: Loaded {len(df)} rows from merged file')
    else:
        print(f"Warning: {merged_path} not found. Skipping {source}.")
"""

new_fb_block = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==========================================\n",
        "# LOAD MERGED FACEBOOK DATA (ADDED)\n",
        "# ==========================================\n",
        "fb_path = os.path.join(DATA_DIR, 'facebook_merged.csv')\n",
        "if os.path.exists(fb_path):\n",
        "    df_fb = pd.read_csv(fb_path)\n",
        "    if 'refined_summary' in df_fb.columns:\n",
        "         fallback = df_fb['summary'] if 'summary' in df_fb.columns else (df_fb['text'] if 'text' in df_fb.columns else '')\n",
        "         df_fb['summary'] = df_fb['refined_summary'].fillna(fallback)\n",
        "    \n",
        "    df_fb['source'] = 'Facebook'\n",
        "    print(f'Facebook: Loaded {len(df_fb)} rows from merged file')\n",
        "    # Append to dfs if the intention is to analyze all together, or keep separate\n",
        "    # For now, we keep separate as df_fb, but user can merge if needed\n",
        "else:\n",
        "    print(f'Warning: {fb_path} not found')\n"
    ]
}

target_idx = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Look for the simpler raw loading loop that uses 'load_summaries_for_use'
        if "load_summaries_for_use" in source and "for source in NEWS_SOURCES:" in source:
            print(f"Found target News loading cell at index {i}")
            target_idx = i
            # Replace code
            # Need to format as list of strings with newlines for ipynb format
            # Using splitlines(keepends=True) to preserve structure or just adding \n
            # splitlines() removes \n, so we re-add it
            nb['cells'][i]['source'] = [line + '\n' for line in new_news_code.splitlines()]
            news_cell_found = True
            break

if news_cell_found:
    # Insert FB cell after target
    # We check if the next cell is already our inserted cell (to avoid duplication on re-runs)
    if target_idx + 1 < len(nb['cells']):
        next_source = "".join(nb['cells'][target_idx + 1].get('source', []))
        if "LOAD MERGED FACEBOOK DATA" in next_source:
             print("Facebook cell already exists. Skipping insertion.")
        else:
             nb['cells'].insert(target_idx + 1, new_fb_block)
             print("Inserted Facebook loading cell")
    else:
        nb['cells'].append(new_fb_block)
        print("Inserted Facebook loading cell at end")

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=4)
    print(f"Successfully updated {nb_path}")

else:
    print("Could not find the exact News loading cell to replace.")
    print("Looking for cell containing: 'load_summaries_for_use' AND 'for source in NEWS_SOURCES:'")
