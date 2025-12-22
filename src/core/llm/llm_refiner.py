import re
import os
import json
from rich.console import Console
from rich.progress import track
from dotenv import load_dotenv

load_dotenv()
console = Console()

class LLMRefiner:
    def __init__(self, provider="gemini", api_key=None, model_path=None, debug=False, batch_size=4):
        self.provider = provider
        self.enabled = False
        self.debug = debug
        
        if provider == "gemini":
            try:
                import google.generativeai as genai
                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
                if not self.api_key:
                    console.print("[yellow]⚠️ GEMINI_API_KEY not found. LLM Refinement disabled.[/yellow]")
                else:
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel("gemini-1.5-flash") # Default
                    self.enabled = True
            except ImportError:
                console.print("[red]❌ google-generativeai not installed.[/red]")

        elif provider == "kaggle" or provider == "local":
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
                
                model_id = model_path or "google/gemma-2-2b-it"
                self.model_id = model_id.lower()
                console.print(f"[bold cyan]🤖 Loading {model_id} via Transformers...[/bold cyan]")
                
                # Use 4-bit quantization to fit larger models in Kaggle T4/P100
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1024, # Increased for batching
                    temperature=0.1,
                    batch_size=batch_size
                )
                self.enabled = True
            except Exception as e:
                console.print(f"[red]❌ Failed to load local model: {e}[/red]")

    def _generate(self, prompt):
        return self._generate_batch([prompt])[0]

    def _generate_batch(self, prompts):
        if not prompts: return []
        
        if self.provider == "gemini":
            results = []
            for p in prompts:
                try:
                    results.append(self.model.generate_content(p).text)
                except Exception:
                    results.append("")
            return results
        else:
            formatted_prompts = []
            for prompt in prompts:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    if "qwen" in self.model_id:
                        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                formatted_prompts.append(formatted)
            
            # Using pipeline with a list of prompts (triggering batching)
            outputs = self.pipeline(
                formatted_prompts,
                max_new_tokens=1024,
                return_full_text=False,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                truncation=True
            )
            
            results = []
            for i, output in enumerate(outputs):
                res_text = output[0]['generated_text']
                
                # Robust extraction: remove known tags
                for tag in ["<start_of_turn>model\n", "<|im_start|>assistant\n"]:
                    if tag in res_text:
                        res_text = res_text.split(tag)[-1]
                
                clean_res = res_text.strip()
                if not clean_res:
                    # Retry single if batch failed for this item (unlikely but safe)
                    try:
                        raw_prompt = f"Task: {prompts[i]}\nResult (JSON):"
                        single_out = self.pipeline(raw_prompt, max_new_tokens=1024, return_full_text=False)
                        clean_res = single_out[0]['generated_text'].strip()
                    except: pass
                
                if self.debug:
                    console.print(f"[dim blue]DEBUG: Generated {len(clean_res)} chars for item {i}.[/dim blue]")
                results.append(clean_res)
            
            return results

    def _extract_json(self, text, is_list=False):
        """Robustly extract JSON from text even with markdown, newlines, or noise"""
        try:
            # Look for markdown blocks first
            code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if code_blocks:
                content = code_blocks[0]
            else:
                # Fallback to finding brackets
                char_start, char_end = ('[', ']') if is_list else ('{', '}')
                start = text.find(char_start)
                end = text.rfind(char_end) + 1
                if start == -1 or end == 0: 
                    # Try to find at least the start
                    if start != -1:
                        content = text[start:]
                    elif is_list and text.find('{') != -1:
                        # Case: Model forgot [ ] but started with {
                        start = text.find('{')
                        content = text[start:]
                    else:
                        return None
                else:
                    content = text[start:end]
            
            # --- SANITIZATION STEP ---
            # 1. Remove "..." if the model hallucinated it
            content = content.replace("...", "")
            
            # 2. Replace literal newlines with spaces (fixes multiline strings)
            content = content.replace('\n', ' ').replace('\r', '')
            
            # 3. Clean trailing commas (common LLM error)
            content = re.sub(r",\s*([\]}])", r"\1", content)

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 4. If still failing, it might be truncated. Try to close it.
                if is_list:
                     # Attempt to wrap bare list if brackets missing
                    if not content.strip().startswith('['):
                        content = "[" + content
                    
                    if not content.strip().endswith(']'):
                         # Try rudimentary fixing
                        try: return json.loads(content + "]")
                        except: pass
                        try: return json.loads(content + "}]")
                        except: pass

                # 5. Last resort: Use regex to extract object by object
                if is_list:
                    # Non-greedy match for complete objects
                    objects = re.findall(r"\{.*?\}", content)
                    if objects:
                        # Reconstruct valid list
                        fixed_json = "[" + ",".join(objects) + "]"
                        try: 
                            data = json.loads(fixed_json)
                            if self.debug: console.print(f"[dim green]DEBUG: Salvaged {len(data)} objects via regex.[/dim green]")
                            return data
                        except: pass
                
                if self.debug: console.print(f"[dim red]DEBUG: Sanitization failed on: {content[:100]}...[/dim red]")
                return None
        except Exception as e:
            if self.debug:
                console.print(f"[dim red]DEBUG: JSON Parse error: {e}[/dim red]")
                console.print(f"[dim yellow]DEBUG: Raw text was: {text}[/dim yellow]")
            return None

    def deduplicate_topics(self, topic_list):
        """
        Phase 4: Semantic Deduplication.
        Takes a list of topic names and returns a mapping {original: canonical}.
        """
        if not self.enabled or not topic_list:
            return {t: t for t in topic_list}

        unique_topics = list(set(topic_list))
        # No need to dedup if very few
        if len(unique_topics) < 2:
            return {t: t for t in topic_list}

        mapping = {t: t for t in topic_list}
        
        # Chunking for long lists (LLM context limit)
        # Reduced from 50 to 20 to prevent hallucinations/over-merging
        chunk_size = 20
        all_prompts = []
        chunks = []
        for i in range(0, len(unique_topics), chunk_size):
            chunk = unique_topics[i : i + chunk_size]
            chunks.append(chunk)
            chunk_str = "\n".join([f"- {t}" for t in chunk])
            
            prompt = f"""
                Role: Senior News Editor.

                Task:
                From the list below, identify headlines that refer to the EXACT SAME real-world event.

                Two headlines are the SAME EVENT ONLY IF:
                - Same location
                - Same timeframe
                - Same main actors/entities

                STRICT RULES:
                - DO NOT paraphrase or rewrite titles.
                - DO NOT invent a new canonical title.
                - Canonical title MUST be copied EXACTLY from one of the input lines.
                - If unsure, DO NOT merge.

                DO NOT merge:
                - Similar accidents in different locations
                - Same topic but different days
                - Generic vs specific headlines

                Input headlines:
                {chunk_str}

                Output format (JSON object ONLY):
                {{
                "Original Title": "Canonical Title (copied verbatim from input)"
                }}

                Include ONLY titles that are merged.
                Do NOT include titles that remain unique.
            """
            all_prompts.append(prompt)
            
        if all_prompts:
            batch_texts = self._generate_batch(all_prompts)
            for i, text in enumerate(batch_texts):
                try:
                    results = self._extract_json(text, is_list=False)
                    if results:
                        for orig, canon in results.items():
                            if orig in mapping:
                                mapping[orig] = canon
                        if self.debug: 
                            console.print(f"[green]DEBUG: Deduped batch {i}: found {len(results)} mappings.[/green]")
                except Exception as e:
                    console.print(f"[red]Dedup error in batch {i}: {e}[/red]")
        
        return mapping
        
        return mapping

    def refine_trends(self, trends_dict):
        """
        Phase 6: Google Trends Refinement.
        Filters out generic/useless trends and merges duplicates.
        Returns: { "filtered": [...], "merged": { "variant": "canonical" } }
        """
        if not self.enabled or not trends_dict:
            return None

        trend_list = list(trends_dict.keys())
        
        # Categorical Grouping: Put related trends in the same batch for better merging
        keyword_groups = {
            "Sports": ["đấu với", "vs", "cup", "bóng đá", "tỉ số", "bxh", "ngoại hạng", "trực tiếp"],
            "Marketplace": ["giá", "vàng", "bạc", "tiền lương", "cà phê", "xăng"],
            "Lottery": ["xổ số", "số miền", "xs", "quay thử"],
            "Game": ["code", "wiki", "the forge", "riot", "honkai", "pubg", "roblox"],
            "General": []
        }
        
        buckets = {k: [] for k in keyword_groups.keys()}
        for t in trend_list:
            assigned = False
            t_lower = t.lower()
            for cat, kws in keyword_groups.items():
                if any(kw in t_lower for kw in kws):
                    buckets[cat].append(t)
                    assigned = True
                    break
            if not assigned:
                buckets["General"].append(t)

        all_filtered = []
        all_merged = {}
        
        console.print(f"[cyan]🧹 Refining {len(trend_list)} Google Trends with Categorical Grouping...[/cyan]")
        
        all_prompts = []
        chunk_size = 30
        
        for cat, items in buckets.items():
            if not items: continue
            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                chunk_str = "\n".join([f"- {t}" for t in chunk])
                
                prompt = f"""
                    Role: Senior Editor.

                        Context: Google Trending Searches in Vietnam.
                        Category hint: {cat}

                        Task:
                        Step 1 – FILTER:
                        Identify terms that are NOT news-worthy.
                        Only remove if CLEARLY generic or utility-like.

                        Step 2 – MERGE:
                        Identify terms that refer to the SAME entity or SAME event.

                        MERGE RULES:
                        - Canonical term MUST appear verbatim in the input list.
                        - Merge multilingual sports terms (e.g. "vs", "đấu với").
                        - Do NOT merge related but different matches or events.

                        FILTER RULES:
                        - Remove prices, weather, schedules, generic queries.
                        - If unsure, KEEP.

                        Input list:
                        {chunk_str}

                        Output (JSON ONLY):
                        {{
                        "filtered": ["term_to_remove"],
                        "merged": {{
                            "variant_term": "canonical_term"
                        }}
                        }}

                """
                all_prompts.append(prompt)
                
        if all_prompts:
            batch_texts = self._generate_batch(all_prompts)
            for i, text in enumerate(batch_texts):
                try:
                    results = self._extract_json(text, is_list=False)
                    if results:
                        all_filtered.extend(results.get("filtered", []))
                        all_merged.update(results.get("merged", {}))
                except Exception as e:
                    console.print(f"[red]Trend Refine Error in batch {i}: {e}[/red]")
        
        return {"filtered": all_filtered, "merged": all_merged}

    def filter_noise_trends(self, trend_list):
        """
        Ad-hoc filter for specific list of trends.
        """
        if not self.enabled: return []
        
        console.print(f"[cyan]🧹 Intelligent Noise Filtering via LLM for {len(trend_list)} trends...[/cyan]")
        all_bad = []
        chunk_size = 50
        all_prompts = []
        
        for i in range(0, len(trend_list), chunk_size):
            chunk = trend_list[i:i+chunk_size]
            prompt = f"""
                You are a strict classifier for TRENDING SEARCH KEYWORDS.

                Your task:
                From the given list, identify ONLY items that are **Noise / Generic searches**.

                A keyword is considered NOISE if it belongs to ONE OR MORE of the following:

                1. Weather & Environment (non-event)
                - temperature, weather, forecast, rain, storm, air quality
                - examples: "nhiệt độ", "nhiệt độ tphcm", "aqi", "vùng áp thấp", "날씨"

                2. Generic utilities / daily queries
                - prices, schedules, dates, calendars, public services
                - examples: "giá xăng dầu", "ngày âm hôm nay", "phạt nguội xe máy", "dich vu cong"

                3. Very broad tech / platform terms WITHOUT a specific event or model
                - examples: "google", "facebook", "wifi", "mp3", "portal"
                - BUT NOT specific products like "iphone 17 pro", "samsung galaxy z trifold"

                4. Non-entity common phrases
                - generic nouns or vague phrases with no clear subject
                - examples: "hình ảnh", "video", "review", "tin nhanh", "random"

                5. Gambling / lottery / betting
                - examples: "xsmn", "bk8", "123b", "bet"

                6. Extremely short or meaningless tokens
                - 1–2 characters or symbols with no semantic meaning
                - examples: "s", "g", "ra", "cf"

                ---

                IMPORTANT – DO NOT mark as noise:
                - Named people (politicians, celebrities, athletes)
                - Countries, cities, regions
                - Sports matches, tournaments, teams (even if repetitive)
                - Movies, TV shows, anime, episodes
                - Games, apps, specific products, brands
                - Laws, decrees, official events
                - Natural disasters or incidents tied to a place/event

                If unsure, KEEP the keyword (do NOT classify as noise).

                ---

                Input list:
                {chunk}

                Output format:
                Return a JSON array of strings.
                ONLY include keywords that are confidently NOISE.
                Do NOT explain. Do NOT include markdown.

                Example output:
                ["nhiệt độ", "aqi", "ngày âm hôm nay"]
                FINAL CHECK:
                If a keyword could reasonably be a headline of a news article, it is NOT noise.
                """

            all_prompts.append(prompt)
            
        if all_prompts:
            if self.provider == 'gemini' and chunk_size > 1: # Optimize batch for Gemini
                 responses = [self._generate(p) for p in all_prompts] # Gemini SDK often better serial? actually _generate_batch handles it
            else:
                 responses = self._generate_batch(all_prompts)
                 
            for resp in responses:
                j = self._extract_json(resp, is_list=True)
                if j: all_bad.extend(j)
                
        return list(set(all_bad))
        

    def refine_cluster(self, cluster_name, posts, original_category=None, topic_type="Discovery", custom_instruction=None, keywords=None):
        if not self.enabled:
            return cluster_name, original_category, ""

        instruction = custom_instruction or """
        Role: Senior News Editor.

            Primary task:
            Rename the cluster into a concise Vietnamese news headline (≤10 words).

            SECONDARY tasks:
            - Assign category (A/B/C)
            - Assign event_type (Specific / Generic)

            RULES:
            - Base the headline ONLY on provided posts.
            - Prefer concrete facts over interpretation.
            - If no clear event → keep generic wording.

            DO NOT:
            - Add opinions
            - Add causes or consequences not stated
            - Guess missing details

            Reasoning:
            - One short sentence.
            - Mention ONLY entities explicitly seen in posts.

            Respond STRICTLY in JSON format:
            {{
                "refined_title": "...",
                "category": "A/B/C",
                "event_type": "Specific/Generic",
                "reasoning": "..."
            }}
        """

        context_texts = [p.get('content', '')[:300] for p in posts[:5]]
        context_str = "\n---\n".join(context_texts)
        
        # Extract metadata
        dates = sorted(list(set([str(p.get('time') or p.get('published_at', ''))[:10] for p in posts if p.get('time') or p.get('published_at')])))
        meta_info = f"Date Range: {dates[0]} to {dates[-1]}" if dates else "Date: Unknown"
        
        # Keywords
        kw_str = f"Keywords: {', '.join(keywords)}" if keywords else ""

        prompt = f"""
            Analyze this cluster of social media/news posts from Vietnam.
            Original Label: {cluster_name}
            Topic Type: {topic_type}
            {meta_info}
            {kw_str}
            
            Sample Posts:
            {context_str}

            {instruction}

            Respond STRICTLY in JSON format:
                {{
                    "refined_title": "...",
                    "category": "A/B/C",
                    "event_type": "Specific/Generic",
                    "reasoning": "..."
                }}
        """
        try:
            text = self._generate(prompt)
            data = self._extract_json(text, is_list=False)
            if data:
                return data.get('refined_title', cluster_name), data.get('category', original_category), data.get('reasoning', ""), data.get('event_type', "Specific")
            return cluster_name, original_category, "", "Specific"
        except Exception:
            return cluster_name, original_category, "", "Specific"

    def refine_batch(self, clusters_to_refine, custom_instruction=None):
        if not self.enabled or not clusters_to_refine:
            return {}

        instruction = custom_instruction or """
            Role: Senior News Editor (Vietnamese newsroom).
                Task:
                Write ONE concise, factual Vietnamese headline for EACH content cluster.

                The headline MUST:
                - Clearly describe the MAIN EVENT
                - Use SPECIFIC DETAILS found in the posts whenever available:
                - Time (date, part of day)
                - Location (city, district, landmark)
                - Named people, organizations, or teams
                - Be written in a neutral, journalistic tone
                - Avoid exaggeration, emotion, or speculation

                STRICT RULES:
                - DO NOT invent facts.
                - DO NOT generalize if specifics exist.
                - DO NOT use vague words like:
                "gây chú ý", "xôn xao", "dậy sóng", "nóng", "bất ngờ".
                - If multiple similar posts exist, summarize them into ONE concrete event.

                Examples:
                [BAD]: "Tai nạn giao thông"
                [BAD]: "Sự việc gây xôn xao dư luận"
                [GOOD]: "Tai nạn liên hoàn tại cầu Phú Mỹ chiều 8/8"
                [GOOD]: "Hà Nội ghi nhận AQI vượt ngưỡng nguy hại sáng 21/12"

                ---

                Classification (assign one):
                Category:
                - A: Critical (Accidents, Safety, Health, Natural disasters)
                - B: Social (Policy, Public affairs, Education, Sports, Viral events)
                - C: Market (Business, Technology, Entertainment, Brands)

                Event Type:
                - Specific: A named, unique event with identifiable time/place
                - Generic: Routine, recurring, or non-unique activity

                If details are missing:
                - Write the most specific headline POSSIBLE using available information.
                - Prefer location over time, and names over descriptions.

                Output format (JSON ONLY):
                {
                "refined_title": "...",
                "category": "A | B | C",
                "event_type": "Specific | Generic"
                }

        
        """

        # Chunking: Small LLMs (Gemma) or large batches can exceed context limits
        # We'll split into chunks of 3 clusters per request for local models (maximum stability)
        chunk_size = 3 if self.provider != "gemini" else 30
        all_results = {}

        all_results = {}
        
        # Use rich progress bar
        iterator = range(0, len(clusters_to_refine), chunk_size)
        all_prompts = []
        cluster_ids_per_chunk = []

        for i in iterator:
            chunk = clusters_to_refine[i : i + chunk_size]
            cluster_ids_per_chunk.append([c['label'] for c in chunk])
            
            batch_str = ""
            for c in chunk:
                # Increase context for better reasoning
                context_list = []
                for j, p in enumerate(c['sample_posts'][:3]): # Up to 3 posts
                    p_text = p.get('content', '')[:500] # Up to 500 chars
                    context_list.append(f"[Post {j+1}] {p_text}")
                
                context = "\n".join(context_list)
                
                # Extract date range for temporal context
                dates = []
                for p in c['sample_posts']:
                    d = p.get('published_at') or p.get('time')
                    if d: dates.append(str(d)[:10]) # YYYY-MM-DD
                
                date_context = ""
                if dates:
                    unique_dates = sorted(list(set(dates)))
                    if len(unique_dates) > 1:
                        date_context = f" [Timeframe: {unique_dates[0]} to {unique_dates[-1]}]"
                    else:
                        date_context = f" [Date: {unique_dates[0]}]"

                # Keywords significantly help grounding
                kw_str = f"Keywords: {', '.join(c.get('keywords', []))}" if c.get('keywords') else ""

                batch_str += f"### Cluster ID: {c['label']}\nName: {c['name']}{date_context}\n{kw_str}\nContext Samples:\n{context}\n\n"

            prompt = f"""
            Analyze these {len(chunk)} news/social clusters from Vietnam.
            {instruction}

            CRITICAL: 
            - Base your 'reasoning' ONLY on the provided Context and Keywords for THAT cluster. 
            - Do NOT mix facts between clusters. 
            - Ensure 'id' in JSON matches the 'Cluster ID'.

            Input Clusters:
            {batch_str}

            Respond STRICTLY in a JSON list of objects:
            [
            {{ "id": label_id, "refined_title": "...", "category": "A/B/C", "event_type": "Specific/Generic", "reasoning": "..." }},
            ...
            ]
            """
            all_prompts.append(prompt)

        if all_prompts:
            batch_texts = self._generate_batch(all_prompts)
            for i, text in enumerate(batch_texts):
                try:
                    results = self._extract_json(text, is_list=True)
                    if results:
                        for item in results:
                            if isinstance(item, dict) and 'id' in item:
                                all_results[item['id']] = item
                        
                        # Log a sample to show it's working
                        if results:
                            sample = results[0]
                            console.print(f"      ✨ [green]Refined {len(results)} clusters. Sample ID {sample.get('id')}: {sample.get('refined_title')}[/green]")
                    else:
                        console.print(f"[yellow]⚠️ Could not find JSON list in LLM response for chunk {i+1}[/yellow]")
                        if self.debug:
                            console.print(f"[dim yellow]DEBUG Raw Response: {text}[/dim yellow]")
                except Exception as e:
                    console.print(f"[red]Batch LLM error in chunk {i+1}: {e}[/red]")
        
        return all_results
        
        return all_results

    def summarize_text(self, text, max_words=100):
        """
        Summarize a long text into a concise paragraph.
        """
        if not self.enabled or not text: return text
        
        prompt = f"""
    Role: Senior Editor.
    Task: Summarize the following article in Vietnamese (max {max_words} words).
    Keep the main entities, numbers, and key events. Delete fluff.

    Input:
    {text[:4000]} # Limit input to avoid token overflow even on LLM side

    Result:
    """
        try:
            summary = self._generate(prompt)
            # Basic cleanup
            return summary.replace("Summary:", "").strip()
        except Exception:
            return text[:500] # Fallback to truncation
