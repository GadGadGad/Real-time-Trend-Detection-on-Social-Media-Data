
import os

FILE_PATH = "src/core/llm/llm_refiner.py"

# Prompt for refine_cluster (Single)
PROMPT_CLUSTER = """            Define the headline and extracting structured 5W1H.

            RULES:
            1. Headline: Concise Vietnamese news headline (≤ 15 words).
               - Prefer concrete facts. No sensationalism.
            2. Summary: DETAILS ARE CRITICAL. write a LONGER, COMPREHENSIVE paragraph (4-6 sentences).
               - Include context, specific numbers, quotes (if any), and future implications.
               - DO NOT start with "Bài viết nói về..." or "Summary:". Just tell the story.
            3. 5W1H:
               - WHO: Main entities/people involved.
               - WHAT: The core interaction or event.
               - WHERE: Specific locations mentioned.
               - WHEN: Timeframe/Dates.
               - WHY: Cause or context (infer if not explicitly stated but logical).
               - IF UNKNOWN, write "N/A" but TRY HARD TO EXTRACT.
            4. Advice for State: Provide strategic recommendations for government agencies/authorities (e.g., communication strategy, policy adjustment, crisis management).
            5. Advice for Business: Provide actionable insights for enterprises/businesses (e.g., market entry, risk mitigation, operational changes, capitalization).

            Respond STRICTLY in JSON format:
            {{
                "refined_title": "...",
                "category": "T1/T2/.../T7",
                "event_type": "Specific/Generic",
                "summary": "Full detailed story of the event (approx 100-150 words).",
                "overall_sentiment": "Positive/Negative/Neutral",
                "who": "...",
                "what": "...",
                "where": "...",
                "when": "...",
                "why": "...",
                "advice_state": "Strategic advice for authorities...",
                "advice_business": "Actionable advice for businesses...",
                "reasoning": "..."
            }}
"""

# Prompt for refine_batch (Batch)
PROMPT_BATCH = """            Role: Senior News Editor (Vietnam).
                Task: Rename the cluster into a single, high-quality Vietnamese headline.

                Headline Rules:
                1. Concise & Factual (≤ 15 words).
                2. Must contain specific Entities (Who/Where/What).
                3. Neutral Tone (No sensationalism like "kinh hoàng", "xôn xao", "cực sốc").
                4. Use standardized Vietnamese (e.g., "TP.HCM" instead of "Sài Gòn" if formal context).
                
                IMPORTANT - Handling Mixed Clusters:
                - If the posts refer to multiple UNRELATED events (e.g., "Apple iPhone" AND "Flood in Hue"):
                  - DO NOT combine them (e.g., "Apple ra iPhone và Lũ lụt ở Huế" is WRONG).
                  - PICK THE DOMINANT TOPIC (the one with more posts or higher news value).
                  - Generate the title for that dominant topic ONLY.
                  - Mention the removed topic in the 'reasoning' field.

                CRITICAL - Incoherent Clusters (STEP-BY-STEP CHECK):
                1. Identify the CORE TOPIC from Post 1 (Anchor Post).
                   Example Anchor: "Tai nạn giao thông Quận 1"
                2. For each Post 2-5, ask: "Does this post describe the SAME specific event as Post 1?"
                   - SAME: Same location AND same incident type AND same time frame.
                   - DIFFERENT: Different location OR different incident type OR different time.
                3. If DIFFERENT, add that post number to outlier_ids.

                STEP-BY-STEP Reasoning Example:
                - Post 1: "Cháy chung cư ở Hà Nội"
                - Post 2: "Cháy chung cư ở Hà Nội" → SAME (same event)
                - Post 3: "Chuyện tình yêu sao Việt" → DIFFERENT (unrelated topic) → outlier_ids: [3]
                - Post 4: "Cháy rừng ở Kon Tum" → DIFFERENT (different location) → outlier_ids: [3, 4]
                  
                - If ALL posts are unrelated to each other:
                  - Set refined_title to "[Incoherent] Mixed Topics"
                  - Add ALL post IDs (2, 3, 4, 5) to outlier_ids

                Anti-Patterns (DO NOT USE):
                - "Tin tức về..." (News about...)
                - "Cập nhật mới nhất..." (Latest updates...)
                - "Những điều cần biết..." (Things to know...)
                - "Cộng đồng mạng dậy sóng..." (Netizens go wild...)

                Data extraction:
                - Category: 
                    * T1 (Crisis & Risk): Accidents, disasters, riots.
                    * T2 (Policy Signal): Regulations, government, politics.
                    * T3 (Reputation): Scandals, accusations, boycotts, controversies.
                    * T4 (Market Demand): Products, travel, food trends.
                    * T5 (Cultural Trend): Memes, viral entertainment, celebs.
                    * T6 (Operational): Traffic, outages, public service failures.
                    * T7 (Noise): Weather, lottery, daily routines (ignore these if possible).
                - Event Type: 
                    * "Specific": A concrete, one-time occurrence with clear Who/What/When/Where (e.g., "Fire at building X", "New policy A announced").
                    * "Generic": A broad, recurring topic or routine update (e.g., "Weather outlook", "Daily gold price", "General discussions").
                - Strategic Advice:
                    * advice_state: Guidance for government/authorities on policy, communication, or management.
                    * advice_business: Actionable insights for enterprises on risks or opportunities.
                - Reasoning: explain your choice and mention if you dropped any unrelated topics from a mixed cluster.
                
                SUMMARY Rules:
                - WRITE A LONG, DETAILED SUMMARY (4-6 sentences, ~100 words).
                - Include context, key figures, and developments.
                - Do not be brief. We need a full picture.

                5W1H Rules:
                - Extract specific details for Who/What/Where/When/Why.
                - Where: Specific city/district/country.
                - When: Specific date/time/period.
                - Why: The cause or reason.
                
                Output JSON:
                {
                    "id": 0,
                    "refined_title": "String",
                    "summary": "Detailed paragraph.",
                    "overall_sentiment": "Positive/Negative/Neutral",
                    "who": "...",
                    "what": "...",
                    "where": "...",
                    "when": "...",
                    "why": "...",
                    "advice_state": "...",
                    "advice_business": "...",
                    "outlier_ids": [id1, id2],
                    "reasoning": "String"
                }
"""

def update_file():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    replacement_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for start of instruction block
        if 'instruction = custom_instruction or """' in line:
            # Check context: is it refine_cluster or refine_batch?
            is_cluster = False
            is_batch = False
            for prev_idx in range(i-1, i-10, -1):
                if prev_idx < 0: break
                if "def refine_cluster" in lines[prev_idx]:
                    is_cluster = True
                    break
                if "def refine_batch" in lines[prev_idx]:
                    is_batch = True
                    break
            
            if is_cluster:
                print(f"Found refine_cluster block at line {i+1}")
                new_lines.append(line) # instruction = ... """
                new_lines.append(PROMPT_CLUSTER)
                
                # Skip until closing quotes
                i += 1
                while i < len(lines):
                    if '"""' in lines[i] and lines[i].strip() == '"""':
                        new_lines.append(lines[i]) # closing """
                        i += 1
                        replacement_count += 1
                        break
                    i += 1
                continue

            elif is_batch:
                print(f"Found refine_batch block at line {i+1}")
                new_lines.append(line) # instruction = ... """
                new_lines.append(PROMPT_BATCH)
                
                # Skip until closing quotes
                i += 1
                while i < len(lines):
                    if '"""' in lines[i] and lines[i].strip() == '"""':
                        new_lines.append(lines[i]) # closing """
                        i += 1
                        replacement_count += 1
                        break
                    i += 1
                continue
        
        new_lines.append(line)
        i += 1
        
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Successfully updated {replacement_count} blocks.")

if __name__ == "__main__":
    update_file()
