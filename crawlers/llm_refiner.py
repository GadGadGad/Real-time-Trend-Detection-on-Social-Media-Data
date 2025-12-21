import os
import json
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()

class LLMRefiner:
    def __init__(self, provider="gemini", api_key=None, model_path=None):
        self.provider = provider
        self.enabled = False
        
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
                    max_new_tokens=512,
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
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = self.pipeline(formatted_prompt)
            return output[0]['generated_text'].split("model\n")[-1]

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
            start = text.find('{')
            end = text.rfind('}') + 1
            data = json.loads(text[start:end])
            return data.get('refined_title', cluster_name), data.get('category', original_category), data.get('reasoning', "")
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
        # We'll split into chunks of 15 clusters per request
        chunk_size = 15 if self.provider != "gemini" else 30
        all_results = {}

        for i in range(0, len(clusters_to_refine), chunk_size):
            chunk = clusters_to_refine[i : i + chunk_size]
            
            batch_str = ""
            for c in chunk:
                # Use slightly more context per post but only 2 posts for better speed/token ratio
                context = "\n".join([p.get('content', '')[:250] for p in c['sample_posts'][:2]])
                batch_str += f"### ID: {c['label']}\nLabel: {c['name']}\nType: {c['topic_type']}\nContext: {context}\n\n"

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
                start = text.find('[')
                end = text.rfind(']') + 1
                if start != -1 and end != -1:
                    results = json.loads(text[start:end])
                    for item in results:
                        all_results[item['id']] = item
                else:
                    console.print(f"[yellow]⚠️ Could not find JSON in LLM response for chunk {i//chunk_size + 1}[/yellow]")
            except Exception as e:
                console.print(f"[red]Batch LLM error in chunk {i//chunk_size + 1}: {e}[/red]")
        
        return all_results
