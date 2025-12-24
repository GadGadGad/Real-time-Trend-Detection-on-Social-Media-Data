import nbformat
import sys
import os

NOTEBOOK_PATH = '/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project/notebooks/analysis-playground-v1.ipynb'

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Notebook not found at {NOTEBOOK_PATH}")
        return

    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    found = False
    
    # Target content to identify the cell
    target_snippet = "classification_results = llm.classify_batch(topics_to_classify)"
    
    # New content for the cell (Complete replacement)
    complete_cell_content = """# Initialize
if 'LLM_PROVIDER' in locals() and LLM_PROVIDER != "none":
    from src.core.llm.llm_refiner import LLMRefiner
    print(f"🚀 Refining Clusters with {LLM_PROVIDER}...")
    
    llm = LLMRefiner(provider=LLM_PROVIDER, api_key=GEMINI_API_KEY, model_path=LLM_MODEL_PATH, debug=True)
    
    # Prepare clusters for refinement
    cluster_groups = {}
    for m in matches_hybrid:
        if m['topic_type'] == 'Noise': continue
        cid = m.get('cluster_id')
        if cid is None: continue
        if cid not in cluster_groups:
            cluster_groups[cid] = {
                'label': cid,
                'name': m['trend'],
                'sample_posts': [],
                'topic_type': m['topic_type'],
                'keywords': trends.get(m['trend'], {}).get('keywords', [])
            }
        if len(cluster_groups[cid]['sample_posts']) < 5:
            cluster_groups[cid]['sample_posts'].append({'content': m['post_content'], 'time': m['time']})

    to_refine = list(cluster_groups.values())
    
    # --- PHASE 3: REFINEMENT (Titles Only) ---
    if to_refine:
        print(f"   🤖 Phase 3: Batch Refining {len(to_refine)} clusters...")
        refined_results = llm.refine_batch(to_refine)
        
        # Create a map of refined titles/reasoning
        refine_map = {}
        for cid, res in refined_results.items():
            refine_map[int(cid)] = res
        
        # --- PHASE 5: CLASSIFICATION (A/B/C) ---
        print(f"   ⚖️ Phase 5: Classifying refined topics...")
        topics_to_classify = []
        for cid, res in refined_results.items():
            topics_to_classify.append({
                "id": cid,                          # CORRECTED: Added ID to match API
                "label": res.get('refined_title'),  # CORRECTED: Renamed key to match API
                "reasoning": res.get('reasoning', "") # CORRECTED: Added context
            })
        
        classification_results = llm.classify_batch(topics_to_classify)
        
        # Merge classification into refine_map
        for topic, class_res in classification_results.items():
            for cid, ref_res in refine_map.items():
                if ref_res.get('refined_title') == topic:
                    ref_res['category'] = class_res.get('category', 'B')
                    ref_res['event_type'] = class_res.get('event_type', 'Specific')
                    break

        # --- PHASE 4: SEMANTIC DEDUPLICATION (Optional) & SAVING ---
        # 1. Deduplication (Optional)
        all_topics = list(set([res['refined_title'] for res in refine_map.values()]))
        canonical_map = {}

        if len(all_topics) > 1:
            print(f"🔗 Phase 4: Deduplicating {len(all_topics)} topics...")
            canonical_map = llm.deduplicate_topics(all_topics)
            
            # Update refine_map with canonical names
            for cid, res in refine_map.items():
                orig = res['refined_title']
                if orig in canonical_map and canonical_map[orig] != orig:
                    res['refined_title'] = canonical_map[orig]
        
        # Apply updates to matches_hybrid
        new_matches = []
        updated_count = 0
        noise_count = 0
        
        for m in matches_hybrid:
            cid = m.get('cluster_id')
            if cid is not None and int(cid) in refine_map:
                res = refine_map[int(cid)]
                
                # Get classification data (now separate)
                category = res.get('category', 'B')
                event_type = res.get('event_type', 'Specific')
                
                # Filter Noise logic
                is_routine_c = (category == 'C' and m['trend_score'] < 90)
                
                if event_type == 'Generic' or is_routine_c:
                   if m['trend_score'] < 80 or is_routine_c:
                       noise_count += 1
                       continue
                
                # Update valid match
                m['final_topic'] = res['refined_title']
                m['category'] = category
                m['llm_reasoning'] = res['reasoning']
                m['topic_type'] = 'Trending' if m['topic_type'] == 'Discovery' else m['topic_type']
                updated_count += 1
            
            new_matches.append(m)
            
        matches_hybrid = new_matches
        print(f"   ✅ Refined {updated_count} posts. Filtered out {noise_count} noise posts.")

        # 3. Save
        import json
        output_path = "notebook_refined_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matches_hybrid, f, ensure_ascii=False, indent=2)
        print(f"\\n💾 Saved {len(matches_hybrid)} refined posts to {output_path}")

        # [EVAL] Refinement
        print("\\n📊 Evaluating Refinement...")
        from src.evaluation.metrics import evaluate_refinement
        ref_stats = evaluate_refinement([], refined_results)
        print(f"   Refined {ref_stats['total_clusters']} clusters.")
        print(f"   Categories: {dict(ref_stats['categories'])}")"""

    for cell in nb.cells:
        if cell.cell_type == 'code':
            if target_snippet in cell.source:
                cell.source = complete_cell_content
                found = True
                print("Successfully patched 'Phase 5' cell.")
                break
    
    if found:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Saved patched notebook to {NOTEBOOK_PATH}")
    else:
        print("Target cell not found for patching. Check if the notebook content has changed.")

if __name__ == "__main__":
    patch_notebook()
