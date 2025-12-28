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
        self.model_name = (model_path or "").lower()  # Track model name for batch size decisions
        
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

    @property
    def is_high_capacity_model(self):
        """Check if using a high-capacity model (gemini API) vs local model (gemma).
        Gemini API can handle much larger batch sizes than local gemma models.
        """
        # If provider is gemini and model_name contains 'gemini' (not 'gemma'), it's high capacity
        if self.provider == "gemini":
            # Check if model_name explicitly mentions gemma (local-like model)
            if "gemma" in self.model_name:
                return False
            return True  # gemini-1.5-pro, gemini-2.0-flash, etc.
        return False  # kaggle/local providers

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
        
        # Gemini API handles 100+ items easily, gemma/local models need smaller batches
        chunk_size = 100 if self.is_high_capacity_model else 20
        all_prompts = []
        chunks = []
        total_chunks = (len(unique_topics) + chunk_size - 1) // chunk_size
        for i in track(range(0, len(unique_topics), chunk_size), description="[cyan]Building dedup prompts...[/cyan]", total=total_chunks):
            chunk = unique_topics[i : i + chunk_size]
            chunks.append(chunk)
            chunk_str = "\n".join([f"- {t}" for t in chunk])
            
            prompt = f"""
                Vai trò: Biên tập viên Tin tức Cao cấp.

                Nhiệm vụ:
                Từ danh sách dưới đây, hãy xác định các tiêu đề cùng đề cập đến MỘT sự kiện thực tế duy nhất.

                Hai tiêu đề là CÙNG MỘT SỰ KIỆN khi có ĐÚNG 3 yếu tố:
                1. CÙNG ĐỊA ĐIỂM: "Hà Nội" vs "Hà Nội" ✓ | "Hà Nội" vs "TP.HCM" ✗
                2. CÙNG THỜI GIAN: "hôm nay" vs "hôm nay" ✓ | "hôm nay" vs "tuần trước" ✗
                3. CÙNG THỰC THỂ CHÍNH: "Bão Yagi" vs "Bão số 3" ✓ | "Bão Yagi" vs "Bão Noru" ✗

                Ví dụ khớp/không khớp:
                - "Tai nạn Quận 1" ≠ "Tai nạn Quận 7" (Khác địa điểm)
                - "Giá vàng tăng hôm nay" ≠ "Giá vàng tuần trước" (Khác thời gian)
                - "Man Utd vs Liverpool" ≠ "Arsenal vs Chelsea" (Khác đội bóng)
                - "Bão Yagi" = "Cơn bão số 3 Yagi" (Cùng thực thể - OK để gộp)

                QUY TẮC ĐẦU RA (NGHIÊM NGẶT):
                - Tiêu đề chuẩn (Canonical Title) PHẢI là bản QUAY LẠI CHÍNH XÁC của một trong các dòng đầu vào.
                - KHÔNG tự tạo tiêu đề mới.
                - KHÔNG gộp nếu không chắc chắn.
                - Trả về đối tượng JSON: {{ "Tiêu đề gốc": "Tiêu đề chuẩn" }}

                Danh sách tiêu đề đầu vào:
                {chunk_str}

                Định dạng đầu ra (Chỉ JSON object):
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
        chunk_size = 300 if self.is_high_capacity_model else 100  # Gemini API handles massive lists
        
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
        chunk_size = 500 if self.is_high_capacity_model else 50  # Gemini API handles 500+ items easily
        all_prompts = []
        total_chunks = (len(trend_list) + chunk_size - 1) // chunk_size
        
        for i in track(range(0, len(trend_list), chunk_size), description="[cyan]Building filter prompts...[/cyan]", total=total_chunks):
            chunk = trend_list[i:i+chunk_size]
            prompt = f"""
                Vai trò: Bộ lọc phân loại cho Google Trends (Việt Nam).
                Nhiệm vụ: Trả về danh sách các từ khóa là RÁC (NOISE) hoặc CHUNG CHUNG (GENERIC).

                ĐỊNH NGHĨA RÁC (Cần loại bỏ):
                1. Thời tiết (Weather): "thời tiết hôm nay", "dự báo mưa", "aqi hà nội" (TRỪ bão có tên như "Bão Yagi")
                2. Tiện ích/Dịch vụ: "giá vàng", "giá xăng", "lịch âm", "xổ số", "xsmn", "vietlott"
                3. Cá cược/Cờ bạc: "bet88", "kubet", "soi cầu", "tỷ lệ cược"
                4. Công nghệ chung chung: "facebook", "gmail", "google", "login", "wifi"
                5. Mơ hồ/Vô nghĩa: "hình ảnh", "video", "clip", "full", "hd", "review", "tin tức"
                6. Khái niệm quá rộng: "tình yêu", "cuộc sống", "học tập", "công việc"

                ĐỊNH NGHĨA SỰ KIỆN (Cần GIỮ LẠI):
                - Nhân vật cụ thể: "Taylor Swift", "Phạm Minh Chính", "Quang Hải"
                - Vụ việc cụ thể: "Vụ cháy chung cư mini", "Bão Yagi" (bão có tên riêng)
                - Trận đấu/Giải đấu: "MU vs Chelsea", "CKTG 2024"
                - Sản phẩm: "iPhone 15", "VinFast VF3"

                Danh sách đầu vào:
                {chunk}

                Đầu ra: Mảng JSON chứa các chuỗi cần LOẠI BỎ.
                Ví dụ: ["thời tiết", "xổ số miền bắc"]
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
            Xác định tiêu đề và trích xuất cấu trúc 5W1H. PHẢI TRẢ LỜI BẰNG TIẾNG VIỆT.

            QUY TẮC:
            1. Tiêu đề (refined_title): Tiêu đề tin tức tiếng Việt súc tích (≤ 15 từ).
               - Ưu tiên các sự kiện cụ thể. Không giật gân.
            2. Tóm tắt (summary): CHI TIẾT LÀ CỰC KỲ QUAN TRỌNG. Viết một đoạn văn DÀI, TOÀN DIỆN (4-6 câu).
               - Bao gồm bối cảnh, con số cụ thể, trích dẫn (nếu có) và hệ quả tương lai.
               - KHÔNG bắt đầu bằng "Bài viết nói về..." hay "Tóm tắt:". Hãy kể câu chuyện trực tiếp.
               - PHẢI VIẾT BẰNG TIẾNG VIỆT.
            3. 5W1H (Trả lời bằng tiếng Việt):
               - WHO: Các thực thể/nhân vật chính liên quan.
               - WHAT: Tương tác hoặc sự kiện cốt lõi.
               - WHERE: Các địa điểm cụ thể được nhắc đến.
               - WHEN: Khung thời gian/Ngày tháng.
               - WHY: Nguyên nhân hoặc bối cảnh.
               - NẾU KHÔNG BIẾT, ghi "N/A" nhưng hãy CỐ GẮNG trích xuất.
            4. Lời khuyên cho Nhà nước (advice_state): Đưa ra các kiến nghị chiến lược cho cơ quan chức năng (ví dụ: chiến lược truyền thông, điều chỉnh chính sách, quản lý khủng hoảng). PHẢI VIẾT BẰNG TIẾNG VIỆT.
            5. Lời khuyên cho Doanh nghiệp (advice_business): Đưa ra các hiểu biết có thể hành động cho doanh nghiệp (ví dụ: thâm nhập thị trường, giảm thiểu rủi ro, thay đổi vận hành, tận dụng cơ hội). PHẢI VIẾT BẰNG TIẾNG VIỆT.
            
            6. PHÂN LOẠI DANH MỤC (category) - CHỌN ĐÚNG MỘT TRONG 7 LOẠI:
               - T1: Khủng hoảng & Rủi ro - Thiên tai, dịch bệnh, tai nạn thảm khốc, khủng bố, chiến tranh
                     VÍ DỤ: Động đất Nhật Bản, Dịch Covid, Cháy rừng, Xung đột Ukraine-Nga
               - T2: Chính sách & Quản trị - Luật mới, quyết định chính phủ, bầu cử, ngoại giao
                     VÍ DỤ: Quốc hội thông qua luật, Thủ tướng thăm nước ngoài, Chính sách thuế mới
               - T3: Rủi ro Uy tín - Scandal, bê bối, tham nhũng, lừa đảo, kiện tụng
                     VÍ DỤ: Quan chức bị bắt, Công ty bị phạt, Nghệ sĩ dính scandal
               - T4: Cơ hội Thị trường - Kinh tế, tài chính, bất động sản, đầu tư, startup
                     VÍ DỤ: VN-Index tăng, Nông nghiệp công nghệ cao, Khởi nghiệp thành công
               - T5: Văn hóa & Giải trí - Thể thao, phim ảnh, âm nhạc, lễ hội, du lịch
                     VÍ DỤ: SEA Games, Phim Việt đoạt giải, Lễ hội Xuân, Vietnam Idol
               - T6: Vận hành & Dịch vụ - Giao thông, y tế, giáo dục, tiện ích công
                     VÍ DỤ: Cao tốc kẹt xe, Bệnh viện quá tải, Trường học mở cửa, Mất điện
               - T7: Tin định kỳ - Thời tiết, dự báo, thống kê thường nhật
                     VÍ DỤ: Dự báo thời tiết, Giá vàng hôm nay, Tỷ giá USD

            Phản hồi NGHIÊM NGẶT theo định dạng JSON:
            {{
                "refined_title": "...",
                "category": "T1 hoặc T2 hoặc T3 hoặc T4 hoặc T5 hoặc T6 hoặc T7",
                "event_type": "Specific/Generic",
                "summary": "Câu chuyện chi tiết đầy đủ về sự kiện (khoảng 100-150 từ).",
                "overall_sentiment": "Positive/Negative/Neutral",
                "who": "...",
                "what": "...",
                "where": "...",
                "when": "...",
                "why": "...",
                "advice_state": "Lời khuyên chiến lược cho cơ quan chức năng...",
                "advice_business": "Lời khuyên thực tiễn cho doanh nghiệp...",
                "reasoning": "Giải thích tại sao chọn category này..."
            }}
        """

        context_texts = [p.get('content', '')[:300] for p in posts[:5]]
        context_str = "\n---\n".join(context_texts)
        
        # Extract metadata
        dates = sorted(list(set([str(p.get('time') or p.get('published_at', '')).split('T')[0] for p in posts if p.get('time') or p.get('published_at')])))
        meta_info = f"Date Range: {dates[0]} to {dates[-1]}" if dates else "Date: Unknown"
        
        # Keywords
        kw_str = f"Keywords: {', '.join(keywords)}" if keywords else ""

        prompt = f"""
            Phân tích cụm bài viết mạng xã hội/tin tức này từ Việt Nam. PHẢI TRẢ LỜI BẰNG TIẾNG VIỆT.
            Tên gốc: {cluster_name}
            Loại chủ đề: {topic_type}
            {meta_info}
            {kw_str}
            
            Bài viết mẫu:
            {context_str}

            {instruction}

            Định dạng trả về NGHIÊM NGẶT là JSON:
                {{
                    "refined_title": "Tiêu đề tiếng Việt",
                    "category": "T1/T2/.../T7",
                    "event_type": "Specific/Generic",
                    "summary": "Tóm tắt chi tiết bằng tiếng Việt...",
                    "overall_sentiment": "Positive/Negative/Neutral",
                    "who": "...",
                    "what": "...",
                    "where": "...",
                    "when": "...",
                    "why": "...",
                    "advice_state": "Lời khuyên cho Nhà nước bằng tiếng Việt...",
                    "advice_business": "Lời khuyên cho Doanh nghiệp bằng tiếng Việt...",
                    "reasoning": "Giải thích bằng tiếng Việt"
                }}
        """
        try:
            text = self._generate(prompt)
            data = self._extract_json(text, is_list=False)
            if data:
                return (
                    data.get('refined_title', cluster_name), 
                    data.get('category', original_category), 
                    data.get('reasoning', ""), 
                    data.get('event_type', "Specific"),
                    data.get('summary', ""),
                    data.get('overall_sentiment', 'Neutral'),
                    {
                        "who": data.get('who', 'N/A'),
                        "what": data.get('what', 'N/A'),
                        "where": data.get('where', 'N/A'),
                        "when": data.get('when', 'N/A'),
                        "why": data.get('why', 'N/A'),
                        "advice_state": data.get('advice_state', 'N/A'),
                        "advice_business": data.get('advice_business', 'N/A')
                    }
                )
            return cluster_name, original_category, "", "Specific", "", "Neutral", {}
        except Exception:
            return cluster_name, original_category, "", "Specific", "", "Neutral", {}

    def refine_batch(self, clusters_to_refine, custom_instruction=None, generate_summary=True):
        if not self.enabled or not clusters_to_refine:
            return {}

        instruction = custom_instruction or """
            Vai trò: Biên tập viên Tin tức Cao cấp (Việt Nam).
            Nhiệm vụ: Đặt lại tên cho cụm tin thành một tiêu đề tiếng Việt duy nhất, chất lượng cao. PHẢI TRẢ LỜI BẰNG TIẾNG VIỆT.

            Quy tắc Tiêu đề:
            1. Súc tích & Thực tế (≤ 15 từ).
            2. Phải chứa các Thực thể cụ thể (Who/Where/What).
            3. Giọng văn trung tính (Không giật gân).
            4. Sử dụng tiếng Việt chuẩn (ví dụ: "TP.HCM" thay vì "Sài Gòn" trong bối cảnh trang trọng).
            
            QUAN TRỌNG - Xử lý Cụm tin Hỗn hợp:
            - Nếu các bài viết đề cập đến nhiều sự kiện KHÔNG LIÊN QUAN (ví dụ: "Apple iPhone" VÀ "Lũ lụt ở Huế"):
              - KHÔNG kết hợp chúng (ví dụ: "Apple ra iPhone và Lũ lụt ở Huế" là SAI).
              - CHỌN CHỦ ĐỀ THỐNG TRỊ (chủ đề có nhiều bài viết hơn hoặc giá trị tin tức cao hơn).
              - Chỉ tạo tiêu đề cho chủ đề thống trị đó.
              - Đề cập đến chủ đề bị loại bỏ trong trường 'reasoning'.

            CẢNH BÁO - Cụm tin không nhất quán (KIỂM TRA TỪNG BƯỚC):
            1. Xác định CHỦ ĐỀ CỐT LÕI từ Bài viết 1 (Bài viết neo).
            2. Với mỗi Bài viết 2-5, hỏi: "Bài viết này có mô tả CÙNG MỘT sự kiện cụ thể như Bài viết 1 không?"
               - CÙNG: Cùng địa điểm VÀ cùng loại sự cố VÀ cùng khung thời gian.
               - KHÁC: Khác địa điểm HOẶC khác loại sự cố HOẶC khác thời gian.
            3. Nếu KHÁC, thêm số thứ tự bài viết đó vào outlier_ids.

            QUY TẮC TÓM TẮT (summary):
            - VIẾT MỘT ĐOẠN TÓM TẮT DÀI, CHI TIẾT (4-6 câu, ~100 từ).
            - Bao gồm bối cảnh, các nhân vật chính và diễn biến sự việc.
            - PHẢI VIẾT BẰNG TIẾNG VIỆT.

            QUY TẮC 5W1H (Trả lời bằng tiếng Việt):
            - Trích xuất chi tiết cụ thể cho Who/What/Where/When/Why.

            Lời khuyên Chiến lược (advice_state, advice_business):
            - PHẢI VIẾT BẰNG TIẾNG VIỆT.

            Kết quả trả về JSON:
            {
                "id": 0,
                "refined_title": "Chuỗi tiếng Việt",
                "summary": "Đoạn văn chi tiết bằng tiếng Việt.",
                "overall_sentiment": "Positive/Negative/Neutral",
                "who": "...",
                "what": "...",
                "where": "...",
                "when": "...",
                "why": "...",
                "advice_state": "...",
                "advice_business": "...",
                "outlier_ids": [id1, id2],
                "reasoning": "Giải thích bằng tiếng Việt"
            }
        """

        # Chunking: Small LLMs (Gemma) or large batches can exceed context limits
        # [QUOTA OPTIMIZATION] For Gemini Free Tier, reduce chunk size to stay under token limits (e.g. 15k tokens/min)
        chunk_size = 10 if self.is_high_capacity_model else 3  # Gemini API can handle more clusters
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
                for j, p in enumerate(c['sample_posts'][:5]): # Up to 5 posts
                    p_text = p.get('content', '')[:500] # Up to 500 chars
                    context_list.append(f"[Post {j+1}] {p_text}")
                
                context = "\n".join(context_list)
                
                # Extract metadata
                dates = []
                for p in c['sample_posts']:
                    d = p.get('published_at') or p.get('time')
                    if d: dates.append(str(d).split('T')[0]) # YYYY-MM-DD
                
                date_context = ""
                if dates:
                    unique_dates = sorted(list(set(dates)))
                    if len(unique_dates) > 1:
                        date_context = f" [Timeframe: {unique_dates[0]} to {unique_dates[-1]}]"
                    else:
                        date_context = f" [Date: {unique_dates[0]}]"

                # Keywords
                kw_str = f"Keywords: {', '.join(c.get('keywords', []))}" if c.get('keywords') else ""

                batch_str += f"### Cluster ID: {c['label']}\nName: {c['name']}{date_context}\n{kw_str}\nContext Samples (Post 1 is Anchor):\n{context}\n\n"

            json_template = '[ {{"id": 0, "refined_title": "Title", "summary": "Detailed summary...", "overall_sentiment": "...", "who": "...", "what": "...", "where": "...", "when": "...", "why": "...", "advice_state": "...", "advice_business": "...", "outlier_ids": [], "reasoning": "..."}} ]'
            
            prompt = f"""
            Analyze những {len(chunk)} news/social clusters này từ Việt Nam.
            {instruction}

            RULES:
            1. Output ONLY a JSON array, nothing else
            2. Start your response with [ and end with ]
            3. Each cluster must have: id, refined_title, summary, outlier_ids, reasoning
            4. outlier_ids are the post numbers (1, 2, 3, 4, 5) from context that DON'T match Post 1.

            Input Clusters:
            {batch_str}

            Respond with ONLY this JSON (no other text):
            {json_template}
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
                                    'summary': item.get('summary', 'No summary provided'),
                                    'overall_sentiment': item.get('overall_sentiment', 'Neutral'),
                                    'who': item.get('who', 'N/A'),
                                    'what': item.get('what', 'N/A'),
                                    'where': item.get('where', 'N/A'),
                                    'when': item.get('when', 'N/A'),
                                    'why': item.get('why', 'N/A'),
                                    'advice_state': item.get('advice_state', 'N/A'),
                                    'advice_business': item.get('advice_business', 'N/A'),
                                    'outlier_ids': item.get('outlier_ids', []),
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

    def classify_batch(self, topic_data_list):
        """
        Classify a batch of topics into Categories (A/B/C) and Event Types (Specific/Generic).
        
        Args:
            topic_data_list: List of dicts, each containing:
                - id: Unique ID
                - label: Refined title
                - reasoning: Reasoning from Phase 3
                
        Returns:
            Dict mapping id -> {category, event_type, reasoning}
        """
        if not self.enabled:
            return {item['id']: {'category': 'T5', 'event_type': 'Specific', 'reasoning': 'LLM Disabled'} for item in topic_data_list}

        results = {}
        batch_size = 10  # Process 10 classifications at a time
        
        # Prepare batches
        batches = [topic_data_list[i:i + batch_size] for i in range(0, len(topic_data_list), batch_size)]
        
        all_prompts = []
        all_items_ordered = []
        
        for batch in batches:
            # Construct prompt for the batch
            batch_items = []
            for item in batch:
                # Defensive check for required keys
                try:
                    batch_items.append({
                        "id": item.get('id', item.get('final_topic', 'unknown')),
                        "title": item.get('label', item.get('final_topic', 'Unknown Topic')),
                        "context": item.get('reasoning', '')[:200]
                    })
                except Exception:
                    continue
            
            if not batch_items:
                continue

            batch_str = json.dumps(batch_items, ensure_ascii=False, indent=2)
            
            prompt = f"""
            Role: Crisis & Event Classifier for Vietnam.
            
            Task: Classify each topic into one of the following 7 Usage Groups:
            - T1 (Crisis & Public Risk): Accidents, fires, natural disasters, epidemics, riots.
            - T2 (Policy & Governance): New regulations, policy announcements, government statements.
            - T3 (Reputation & Trust): Scandals, accusations, boycotts, controversies.
            - T4 (Market Opportunity): Product trends, lifestyle changes, tech adoption.
            - T5 (Cultural & Attention): Memes, celebrities, entertainment, viral noise.
            - T6 (Operational Pain): Traffic, power outages, public service failures.
            - T7 (Routine Signals): Weather updates, lottery, daily sports results.
               
            2. EVENT TYPE:
               - Specific: A concrete event with a distinct start/end and clear actors (e.g., "Bão Yagi", "Vụ cháy chung cư A", "Khai mạc hội nghị X").
               - Generic: Broad topics, recurring reports, or vague discussions (e.g., "Tình hình thời tiết", "Giá xăng hôm nay", "Chuyện đời thường", "Thông tin thị trường"). 
               - RULE: If it's a routine update without a "breaking" news point, mark as GENERIC.
               
            Input Topics:
            {batch_str}
            
            Output: JSON Object mapping ID -> Classification.
            Example:
            {{
                "0": {{ 
                    "category": "T1", 
                    "event_type": "Specific", 
                    "overall_sentiment": "Positive/Negative/Neutral",
                    "summary": "Short context of the event.",
                    "reasoning": "..." 
                }}
            }}
            """
            all_prompts.append(prompt)
            all_items_ordered.extend(batch)
            
        # Execute batch generation
        if not all_prompts: return {}
        
        console.print(f"[cyan]🛡️ Classifying {len(topic_data_list)} topics in {len(batches)} batches...[/cyan]")
        
        responses = self._generate_batch(all_prompts)
        
        # Process results
        current_idx = 0
        for i, resp in enumerate(responses):
            batch_items = batches[i]
            parsed = self._extract_json(resp, is_list=False)
            
            if not parsed:
                # Fallback if parsing fails
                for item in batch_items:
                    results[item['id']] = {"category": "T5", "event_type": "Specific", "reasoning": "Parse Error"}
                continue
                
            for item in batch_items:
                # ID might be int or str in JSON keys
                item_id = item.get('id') or item.get('final_topic', 'unknown')
                key = str(item_id)
                if key in parsed:
                    info = parsed[key]
                    results[item_id] = {
                        "category": info.get("category", "T5"),
                        "event_type": info.get("event_type", "Specific"),
                        "overall_sentiment": info.get("overall_sentiment", "Neutral"),
                        "summary": info.get("summary", ""),
                        "reasoning": info.get("reasoning", "")
                    }
                else:
                    results[item_id] = {"category": "T5", "event_type": "Specific", "overall_sentiment": "Neutral", "summary": "", "reasoning": "Missing in response"}

        return results

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
