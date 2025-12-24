import re
import os
import json
from rich.console import Console
from rich.progress import track
from dotenv import load_dotenv
import torch

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
                    # Allow specifying model via model_path (e.g. 'gemini-1.5-pro')
                    gemini_model = model_path or "models/gemma-3-27b-it"
                    console.print(f"[cyan]♊ Using Gemini Model: {gemini_model}[/cyan]")
                    self.model = genai.GenerativeModel(gemini_model)
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
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False, # Disable for max stability
                )

                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Enforce limit to fix truncation warning and prevent OOB
                self.tokenizer.model_max_length = 4096
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                # self.pipeline removal - using manual generate for stability
                self.enabled = True
            except Exception as e:
                console.print(f"[red]❌ Failed to load local model: {e}[/red]")

    def _generate(self, prompt):
        return self._generate_batch([prompt])[0]

    def _generate_batch(self, prompts):
        if not prompts: return []
        
        if self.provider == "gemini":
            import concurrent.futures
            
            def get_content(p):
                import time
                import re as _re
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        # Safety settings to minimize refusals
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]
                        response = self.model.generate_content(p, safety_settings=safety_settings)
                        
                        # Handle Finish Reasons e.g. RECITATION (4)
                        if response.candidates and response.candidates[0].finish_reason != 1: # 1 = STOP
                            if self.debug: 
                                console.print(f"[dim yellow]Gemini Finish Reason: {response.candidates[0].finish_reason}[/dim yellow]")
                            # Attempt to extract partial text if available
                            if hasattr(response, 'text'): 
                                try: return response.text
                                except: pass
                            return ""
                            
                        return response.text
                    except Exception as e:
                        error_str = str(e)
                        
                        # Handle 429 Rate Limit with retry
                        if "429" in error_str or "quota" in error_str.lower():
                            # Try to parse recommended wait time
                            wait_match = _re.search(r'retry in (\d+\.?\d*)', error_str.lower())
                            wait_time = float(wait_match.group(1)) if wait_match else 30.0
                            wait_time = min(wait_time + 5, 60)  # Add buffer, cap at 60s
                            
                            if attempt < max_retries - 1:
                                console.print(f"[yellow]⏳ Rate limited. Waiting {wait_time:.0f}s before retry {attempt+2}/{max_retries}...[/yellow]")
                                time.sleep(wait_time)
                                continue
                        
                        if self.debug: console.print(f"[dim red]Gemini Error: {e}[/dim red]")
                        return ""
                
                return ""  # All retries failed

            # Use ThreadPoolExecutor for parallel API calls
            # Reduced workers to prevent Rate Limits and "Stuck" behavior
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(get_content, prompts))
            return results
        else:
            results = []
            # Use progress bar for visible inference
            iterator = track(prompts, description="[cyan]🤖 Generating Responses...[/cyan]") if len(prompts) > 1 else prompts
            for prompt in iterator:
                try:
                    # Apply template
                    formatted = ""
                    try:
                        messages = [{"role": "user", "content": prompt}]
                        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception:
                        if "qwen" in self.model_id:
                            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                        else:
                            formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                            
                    # Manual Generation (Bare Metal Stability)
                    inputs = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                    # Decode only the new tokens
                    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                    res_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    results.append(res_text)
                    
                except Exception as e:
                    console.print(f"[red]Generation Error: {e}[/red]")
                    results.append("")
            
            return results


    def _extract_json(self, text, is_list=False):
        """Robustly extract JSON from text even with markdown, newlines, or noise"""
        if not text or not text.strip():
            return None
            
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
            # 0. Ensure we have clean text - strip leading/trailing whitespace
            content = content.strip()
            
            # 1. Remove "..." if the model hallucinated it (as placeholder for truncation)
            content = content.replace("...", "")
            content = content.replace("…", "")  # Unicode ellipsis
            
            # 2. Normalize whitespace (convert all whitespace including tabs/newlines to single spaces)
            content = re.sub(r'\s+', ' ', content)
            
            # 3. Clean trailing commas (common LLM error)
            content = re.sub(r",\s*([\]}])", r"\1", content)
            
            # 4. Fix common LLM error: single quotes instead of double
            # Only apply if no double quotes exist (likely all single-quoted)
            if '"' not in content and "'" in content:
                content = content.replace("'", '"')

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 5. If still failing, it might be truncated. Try to close it.
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
                        try: return json.loads(content + "\"}]")
                        except: pass

                # 6. Recovery for OBJECTS (is_list=False)
                if not is_list:
                    # Attempt to close truncated JSON objects
                    # Try common closing patterns
                    candidates = ['}', '"}', '"]}', '"]}}', '"]}}', '']
                    for suffix in candidates:
                        try:
                            return json.loads(content + suffix)
                        except: pass
                    
                    # Try closing open quotes first if odd number of quotes
                    if content.count('"') % 2 != 0:
                        for suffix in candidates:
                            try:
                                return json.loads(content + '"' + suffix)
                            except: pass

                # 7. IMPROVED Last resort for LISTS: Use greedy regex for complete objects
                if is_list:
                    # Try to find objects with proper brace balancing
                    objects = []
                    brace_depth = 0
                    current_obj = ""
                    in_string = False
                    prev_char = ""
                    
                    for char in content:
                        if char == '"' and prev_char != '\\':
                            in_string = not in_string
                        
                        if not in_string:
                            if char == '{':
                                if brace_depth == 0:
                                    current_obj = ""
                                brace_depth += 1
                            elif char == '}':
                                brace_depth -= 1
                                if brace_depth == 0:
                                    current_obj += char
                                    objects.append(current_obj)
                                    current_obj = ""
                                    prev_char = char
                                    continue
                        
                        if brace_depth > 0:
                            current_obj += char
                        prev_char = char
                    
                    # If we captured objects, try to parse them
                    if objects:
                        fixed_json = "[" + ",".join(objects) + "]"
                        try: 
                            data = json.loads(fixed_json)
                            if self.debug: console.print(f"[dim green]DEBUG: Salvaged {len(data)} objects via brace-balanced parsing.[/dim green]")
                            return data
                        except:
                            pass
                    
                    # Fallback to simple regex
                    simple_objects = re.findall(r'\{[^{}]+\}', content)
                    if simple_objects:
                        fixed_json = "[" + ",".join(simple_objects) + "]"
                        try: 
                            data = json.loads(fixed_json)
                            if self.debug: console.print(f"[dim green]DEBUG: Salvaged {len(data)} objects via simple regex.[/dim green]")
                            return data
                        except: pass
                
                # 8. NESTED ARRAYS: Salvage complete inner [...] arrays from truncated response
                # This handles: [["kw1"], ["kw2"], ["kw3  <- incomplete
                # We extract all complete inner arrays
                # Check for nested arrays with optional whitespace: [ [ or [[
                if is_list and re.search(r'\[\s*\[', content):
                    inner_arrays = []
                    bracket_depth = 0
                    current_arr = ""
                    in_string = False
                    prev_char = ""
                    
                    for char in content:
                        # Track string state (respecting escaped quotes)
                        if char == '"' and prev_char != '\\':
                            in_string = not in_string
                        
                        # Handle brackets (only when not inside a string)
                        if not in_string:
                            if char == '[':
                                bracket_depth += 1
                                if bracket_depth == 2:  # Starting an inner array
                                    current_arr = "["
                                elif bracket_depth > 2:  # Nested bracket inside inner array
                                    current_arr += char
                            elif char == ']':
                                if bracket_depth == 2:  # Completing an inner array
                                    current_arr += "]"
                                    inner_arrays.append(current_arr)
                                    current_arr = ""
                                elif bracket_depth > 2:  # Nested bracket inside inner array
                                    current_arr += char
                                bracket_depth -= 1
                            else:
                                # Non-bracket char outside string, add if in inner array
                                if bracket_depth >= 2:
                                    current_arr += char
                        else:
                            # Inside a string, add to current array if we're in an inner array
                            if bracket_depth >= 2:
                                current_arr += char
                        
                        prev_char = char
                    
                    # If we captured complete inner arrays, parse them
                    if inner_arrays:
                        fixed_json = "[" + ",".join(inner_arrays) + "]"
                        try:
                            data = json.loads(fixed_json)
                            if self.debug: console.print(f"[dim green]DEBUG: Salvaged {len(data)} nested arrays from truncated response.[/dim green]")
                            return data
                        except Exception as e:
                            if self.debug: console.print(f"[dim yellow]DEBUG: Nested array salvage failed to parse: {e}. Arrays: {inner_arrays[:3]}...[/dim yellow]")
                            pass
                
                if self.debug: console.print(f"[dim red]DEBUG: Sanitization failed on: {content[:200]}...[/dim red]")
                return None
        except Exception as e:
            if self.debug:
                console.print(f"[dim red]DEBUG: JSON Parse error: {e}[/dim red]")
                console.print(f"[dim yellow]DEBUG: Raw text was: {text[:300]}...[/dim yellow]")
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
        
        # Reduced from 50 to 20 to prevent hallucinations/over-merging for local models
        # But Gemini can handle 100+ easily
        chunk_size = 100 if self.provider == "gemini" else 20
        all_prompts = []
        chunks = []
        total_chunks = (len(unique_topics) + chunk_size - 1) // chunk_size
        for i in track(range(0, len(unique_topics), chunk_size), description="[cyan]Building dedup prompts...[/cyan]", total=total_chunks):
            chunk = unique_topics[i : i + chunk_size]
            chunks.append(chunk)
            chunk_str = "\n".join([f"- {t}" for t in chunk])
            
            prompt = f"""
                Role: Senior News Editor.

                Task:
                From the list below, identify headlines that refer to the EXACT SAME real-world event.

                Two headlines are the SAME EVENT ONLY IF they share:
                1. The SAME specific location (e.g., "Hanoi", "Nguyen Hue Walking Street")
                2. The SAME time/date (e.g., "Last night", "Oct 15")
                3. The SAME main entities (e.g., "Messi", "iPhone 16")

                strict matching rules:
                - "Traffic accident in District 1" != "Traffic accident in District 3" (Different location)
                - "Gold price increased today" != "Gold price increased yesterday" (Different time)
                - "Storm Yagi" == "Typhoon Yagi" (Match)

                STRICT OUTPUT RULES:
                - Canonical title MUST be an EXACT COPY of one of the input lines.
                - DO NOT create new titles.
                - DO NOT merge if unsure.
                - Return JSON object: {{ "Original Title": "Canonical Title" }}

                Input headlines:
                {chunk_str}

                Output format (JSON object ONLY):
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
        
        console.print(f"[cyan]🧹 Refining {len(trend_list)} Google Trends with Categorical Grouping (Provider: {self.provider})...[/cyan]")
        
        all_prompts = []
        chunk_size = 300 if self.provider == "gemini" else 100 # Gemini handles massive lists
        
        for cat, items in buckets.items():
            if not items: continue
            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                chunk_str = "\n".join([f"- {t}" for t in chunk])
                
                prompt = f"""
                    Role: Senior News Editor.
                        Context: Google Trending Searches in Vietnam.
                        Category hint: {cat}

                        Task:
                        1. FILTER: Remove terms that are clearly Generic, Utilities, or meaningless.
                           - NOISE: "xổ số", "kết quả", "thời tiết", "giá vàng", "lịch vạn niên", "random chars"
                           - KEEP: "bão Yagi", "iPhone 16", "Man Utd vs Liverpool", "Blackpink"

                        2. MERGE: Group key terms referring to the EXACT SAME event.
                           - MUST use one of the input terms as the canonical term.
                           - Example: "lịch thi đấu aff cup", "bxh aff cup" -> "AFF Cup 2024" (if present)
                           - Example: "giá xăng hôm nay", "giá xăng tăng" -> "Giá xăng dầu" (if present)

                        Input list:
                        {chunk_str}

                        Output (JSON ONLY):
                        {{
                        "filtered": ["term_to_remove", "term_to_remove"],
                        "merged": {{
                            "variant_term": "canonical_term"
                        }}
                        }}
                """
                all_prompts.append(prompt)
                
        if all_prompts:
            # Process one by one to show granular progress
            inference_batch_size = 1
            
            # Using rich progress track
            for i in track(range(0, len(all_prompts), inference_batch_size), description="[cyan]Processing Trend Batches...[/cyan]"):
                batch_prompts = all_prompts[i : i + inference_batch_size]
                batch_texts = self._generate_batch(batch_prompts)
                
                for text in batch_texts:
                    try:
                        results = self._extract_json(text, is_list=False)
                        if results:
                            all_filtered.extend(results.get("filtered", []))
                            all_merged.update(results.get("merged", {}))
                    except Exception as e:
                        console.print(f"[red]Trend Refine Parse Error: {e}[/red]")
        
        return {"filtered": all_filtered, "merged": all_merged}

    def filter_noise_trends(self, trend_list):
        """
        Ad-hoc filter for specific list of trends.
        """
        if not self.enabled: return []
        
        console.print(f"[cyan]🧹 Intelligent Noise Filtering via {self.provider} for {len(trend_list)} trends...[/cyan]")
        all_bad = []
        chunk_size = 500 if self.provider == "gemini" else 50 # Gemini handles 500+ items easily
        all_prompts = []
        total_chunks = (len(trend_list) + chunk_size - 1) // chunk_size
        
        for i in track(range(0, len(trend_list), chunk_size), description="[cyan]Building filter prompts...[/cyan]", total=total_chunks):
            chunk = trend_list[i:i+chunk_size]
            prompt = f"""
                Role: Classifier for Google Trends (Vietnam).
                Task: Return a list of keywords that are NOISE or GENERIC.

                DEFINITION OF NOISE (Remove these):
                1. Weather: "thời tiết", "nhiệt độ", "mưa", "bão" (generic), "aqi"
                2. Utilities: "giá vàng", "giá xăng", "lịch âm", "xổ số", "xsmn", "vietlott"
                3. Betting/Gambling: "bet88", "kubet", "soi cầu", "tỷ lệ cược"
                4. Generic Tech: "facebook", "gmail", "google", "login", "wifi"
                5. Vague/Meaningless: "hình ảnh", "video", "clip", "full", "hd", "review", "tin tức"
                6. Broad Concepts: "tình yêu", "cuộc sống", "học tập", "công việc"

                DEFINITION OF EVENTS (KEEP these):
                - Specific People: "Taylor Swift", "Phạm Minh Chính", "Quang Hải"
                - Specific Incidents: "Vụ cháy chung cư mini", "Bão Noru" (named storms)
                - Matches/Games: "MU vs Chelsea", "CKTG 2024"
                - Products: "iPhone 15", "VinFast VF3"

                Input keys:
                {chunk}

                Output: JSON Array of strings to REMOVE.
                Example: ["thời tiết", "xổ số miền bắc"]
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
            Role: Senior News Editor (Vietnam).
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

                Anti-Patterns (DO NOT USE):
                - "Tin tức về..." (News about...)
                - "Cập nhật mới nhất..." (Latest updates...)
                - "Những điều cần biết..." (Things to know...)
                - "Cộng đồng mạng dậy sóng..." (Netizens go wild...)

                Data extraction:
                - Category: A (Critical/Safety), B (Social/Politics/Sports), C (Entertainment/Business).
                - Event Type: "Specific" (One-time event) or "Generic" (Recurring/Topic).
                - Reasoning: explain your choice and mention if you dropped any unrelated topics from a mixed cluster.

                Output JSON:
                {
                    "refined_title": "String",
                    "reasoning": "String"
                }

        
        """

        # Chunking: Small LLMs (Gemma) or large batches can exceed context limits
        # [QUOTA OPTIMIZATION] For Gemini Free Tier, reduce chunk size to stay under token limits (e.g. 15k tokens/min)
        chunk_size = 3 if self.provider != "gemini" else 10
        all_results = {}
        
        # Build prompts
        all_prompts = []
        cluster_ids_per_chunk = []
        total_chunks = (len(clusters_to_refine) + chunk_size - 1) // chunk_size

        for i in track(range(0, len(clusters_to_refine), chunk_size), description="[cyan]Building cluster prompts...[/cyan]", total=total_chunks):
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

            RULES:
            1. Output ONLY a JSON array, nothing else
            2. Start your response with [ and end with ]
            3. Each cluster must have: id, refined_title, reasoning
            4. Focus purely on renaming the cluster into a high-quality headline.

            Input Clusters:
            {batch_str}

            Respond with ONLY this JSON (no other text):
            [[{{"id": 0, "refined_title": "Vietnamese headline", "reasoning": "why"}}]]
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
                                # Ensure minimal schema to prevent KeyErrors
                                sane_item = {
                                    'id': item['id'],
                                    'refined_title': item.get('refined_title', f"Cluster {item['id']}"),
                                    'reasoning': item.get('reasoning', 'No reasoning provided')
                                }
                                all_results[item['id']] = sane_item
                        
                        # Log a sample to show it's working
                        # Find first valid dict result for sample logging
                        valid_samples = [r for r in results if isinstance(r, dict) and 'id' in r]
                        if valid_samples:
                            sample = valid_samples[0]
                            console.print(f"      ✨ [green]Refined {len(valid_samples)} clusters. Sample ID {sample.get('id')}: {sample.get('refined_title')}[/green]")
                        else:
                            console.print(f"[yellow]⚠️ Chunk {i+1}: Parsed {len(results)} items but none were valid cluster dicts[/yellow]")
                    else:
                        console.print(f"[yellow]⚠️ Could not find JSON list in LLM response for chunk {i+1}[/yellow]")
                        if self.debug:
                            # Show first 500 chars of response for debugging
                            console.print(f"[dim yellow]DEBUG Raw Response (first 500 chars):[/dim yellow]")
                            console.print(f"[dim]{text[:500]}[/dim]")
                except Exception as e:
                    console.print(f"[red]Batch LLM error in chunk {i+1}: {type(e).__name__}: {e}[/red]")
                    # Show response preview for debugging even without debug mode
                    console.print(f"[dim red]Response preview: {text[:200] if text else 'empty'}...[/dim red]")
                    if self.debug:
                        import traceback
                        console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
        
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
