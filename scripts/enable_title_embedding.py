import json

notebook_path = 'notebooks/analysis-playground.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# The code to inject
new_logic = [
    "# --- STEP 2: Preprocessing & Keywords ---\n",
    "USE_TITLE_EMBEDDING = True  # [NEW] Set True to embed Titles (News) instead of full Content\n",
    "\n",
    "print(\"📝 Preprocessing posts...\")\n",
    "if USE_TITLE_EMBEDDING:\n",
    "    print(\"ℹ️ Mode: TITLE Embedding (using Title for News, Content for FB)\")\n",
    "    post_contents = []\n",
    "    for p in posts:\n",
    "        # Prefer Title for News, fall back to Content for FB\n",
    "        text = p.get('title', '')\n",
    "        if not text or len(str(text)) < 5:\n",
    "            text = p.get('content', '')\n",
    "        post_contents.append(str(text)[:EMBEDDING_CHAR_LIMIT])\n",
    "else:\n",
    "    print(\"ℹ️ Mode: CONTENT Embedding (using full Content)\")\n",
    "    post_contents = [p.get('content', '')[:EMBEDDING_CHAR_LIMIT] for p in posts]\n",
    "\n",
    "# Debug Check\n",
    "print(f\"Sample (Head): {post_contents[0][:50]}...\")\n",
    "\n",
    "if USE_KEYWORDS:\n",
    "    from src.core.extraction.keyword_extractor import KeywordExtractor\n",
    "    print(\"🔑 Extracting high-signal keywords...\")\n",
    "    kw_extractor = KeywordExtractor()\n",
    "    post_contents_enriched = kw_extractor.batch_extract(post_contents)\n",
    "else:\n",
    "    post_contents_enriched = post_contents\n"
]

# Find the cell to replace
found = False
for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell.get('source', []))
    if "# --- STEP 2: Preprocessing & Keywords ---" in source_str:
        # Replace this cell's source
        nb['cells'][i]['source'] = new_logic
        found = True
        print(f"✅ Updated Step 2 cell at index {i}")
        break

if not found:
    print("❌ Could not find Step 2 cell. Inserting new cell...")
    # Find "1. Load Data" and insert after
    for i, cell in enumerate(nb['cells']):
        if "1. Load Data" in "".join(cell.get('source', [])):
            nb['cells'].insert(i + 2, {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": new_logic
            })
            print(f"✅ Inserted new logic at index {i+2}")
            break

# Save
with open(notebook_path, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Notebook updated to support USE_TITLE_EMBEDDING")
