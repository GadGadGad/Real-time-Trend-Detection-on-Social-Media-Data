import json
import os

NB_PATH = "notebooks/analysis-playground.ipynb"

INSPECT_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- STEP 2.5: INSPECT EXTRACTION ---\n",
        "print(\"🔍 Inspecting Keywords & Metadata on random samples...\")\n",
        "import random\n",
        "\n",
        "kw_debug_extractor = KeywordExtractor()\n",
        "samples = random.sample(posts, min(5, len(posts)))\n",
        "\n",
        "for i, p in enumerate(samples):\n",
        "    content = p.get('content', '')\n",
        "    # Re-run extraction to show current logic\n",
        "    db_keywords = kw_debug_extractor.extract_keywords(content)\n",
        "    date_str = p.get('time') or p.get('published_at', 'Unknown')\n",
        "    \n",
        "    print(f\"\\n--- Post {i+1} [{p.get('source', 'Unknown')}] ---\")\n",
        "    print(f\"📅 Date: {date_str}\")\n",
        "    print(f\"🔑 Keywords: {db_keywords}\")\n",
        "    print(f\"📝 Content (First 150 chars): {content[:150].replace(chr(10), ' ')}...\")"
    ]
}

def main():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    inserted = False
    
    for cell in nb['cells']:
        new_cells.append(cell)
        src = "".join(cell.get('source', []))
        # Insert after Step 2 (Preprocessing)
        if "# --- STEP 2: Preprocessing & Keywords ---" in src and not inserted:
            print("✅ Found Step 2. Inserting Inspection Step 2.5...")
            new_cells.append(INSPECT_CELL)
            inserted = True
            
    if inserted:
        nb['cells'] = new_cells
        with open(NB_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("✅ Notebook updated with Keyword Inspection.")
    else:
        print("⚠️ Could not find Step 2 to insert inspection.")

if __name__ == "__main__":
    main()
