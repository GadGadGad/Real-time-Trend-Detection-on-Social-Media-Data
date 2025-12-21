import re
import os
import json
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()

class LLMRefiner:
    def __init__(self, provider="gemini", api_key=None, model_path=None, debug=False):
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
                console.print(f"[bold cyan]🤖 Loading {model_id} via Transformers...[/bold cyan]")
                
                # Use 4-bit quantization to fit larger models in Kaggle T4/P100
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
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
                    temperature=0.1
                )
                self.enabled = True
            except Exception as e:
                console.print(f"[red]❌ Failed to load local model: {e}[/red]")

    def _generate(self, prompt):
        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text
        else:
            # Gemma chat format
            try:
                # Try standard template
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Manual Gemma fallback
                formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            # Setting return_full_text=False returns ONLY the generated part
            output = self.pipeline(
                formatted_prompt, 
                max_new_tokens=1024,
                return_full_text=False, 
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                truncation=True
            )
            res_text = output[0]['generated_text']
            
            if self.debug:
                console.print(f"[dim blue]DEBUG: Generated {len(res_text)} chars.[/dim blue]")
            
            # If empty, the model might be a "Base" model and hate the tags
            if not res_text.strip():
                if self.debug: console.print("[dim yellow]DEBUG: Empty response with tags. Retrying with raw prompt...[/dim yellow]")
                # Fallback to a plain text completion prompt
                raw_prompt = f"Task: {prompt}\nResult (JSON):"
                output = self.pipeline(raw_prompt, max_new_tokens=1024, return_full_text=False)
                res_text = output[0]['generated_text']

            return res_text.strip()

    def _extract_json(self, text, is_list=False):
        """Robustly extract JSON from text even with markdown or noise"""
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
                if start == -1 or end == 0: return None
                content = text[start:end]
            
            return json.loads(content)
        except Exception as e:
            if self.debug:
                console.print(f"[dim red]DEBUG: JSON Parse error: {e}[/dim red]")
                console.print(f"[dim yellow]DEBUG: Raw text was: {text}[/dim yellow]")
            return None

    def refine_cluster(self, cluster_name, posts, original_category=None, topic_type="Discovery", custom_instruction=None):
        if not self.enabled:
            return cluster_name, original_category, ""

        instruction = custom_instruction or """Tasks:
1. Create a professional, concise title for this event in Vietnamese (max 8 words). 
2. Classify into:
   - A: Critical (Accidents, Disasters, Safety)
   - B: Social (Policy, controversy, public sentiment)
   - C: Market (Commerce, Tech, Entertainment)
3. Provide a brief Vietnamese reasoning (1 sentence)."""

        context_texts = [p.get('content', '')[:300] for p in posts[:5]]
        context_str = "\n---\n".join(context_texts)

        prompt = f"""
Analyze this cluster of social media/news posts from Vietnam.
Original Label: {cluster_name}
Topic Type: {topic_type}
Sample Posts:
{context_str}

{instruction}

Respond STRICTLY in JSON format:
{{
  "refined_title": "...",
  "category": "A/B/C",
  "reasoning": "..."
}}
"""
        try:
            text = self._generate(prompt)
            data = self._extract_json(text, is_list=False)
            if data:
                return data.get('refined_title', cluster_name), data.get('category', original_category), data.get('reasoning', "")
            return cluster_name, original_category, ""
        except Exception:
            return cluster_name, original_category, ""

    def refine_batch(self, clusters_to_refine, custom_instruction=None):
        if not self.enabled or not clusters_to_refine:
            return {}

        instruction = custom_instruction or """For each cluster ID, provide a professional title, category, and reasoning.
Categories:
- A: Critical (Accidents, Disasters, Safety)
- B: Social (Policy, controversy, public sentiment)
- C: Market (Commerce, Tech, Entertainment)"""

        # Chunking: Small LLMs (Gemma) or large batches can exceed context limits
        # We'll split into chunks of 3 clusters per request for local models (maximum stability)
        chunk_size = 3 if self.provider != "gemini" else 30
        all_results = {}

        for i in range(0, len(clusters_to_refine), chunk_size):
            chunk = clusters_to_refine[i : i + chunk_size]
            
            batch_str = ""
            for c in chunk:
                # Limit context significantly for 2B models
                context = "\n".join([p.get('content', '')[:150] for p in c['sample_posts'][:2]])
                batch_str += f"- ID: {c['label']} | Name: {c['name']}\n  Context: {context}\n\n"

            prompt = f"""
Analyze these {len(chunk)} news/social clusters from Vietnam.
{instruction}

Input Clusters:
{batch_str}

Respond STRICTLY in a JSON list of objects:
[
  {{ "id": label_id, "refined_title": "...", "category": "A/B/C", "reasoning": "..." }},
  ...
]
"""
            try:
                text = self._generate(prompt)
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
                    console.print(f"[yellow]⚠️ Could not find JSON list in LLM response for chunk {i//chunk_size + 1}[/yellow]")
                    if self.debug:
                        console.print(f"[dim yellow]DEBUG Raw Response: {text}[/dim yellow]")
            except Exception as e:
                console.print(f"[red]Batch LLM error in chunk {i//chunk_size + 1}: {e}[/red]")
                if self.debug:
                    # In case text was not even generated or crashed before
                    try: console.print(f"[dim red]DEBUG Text: {text}[/dim red]")
                    except: pass
        
        return all_results
