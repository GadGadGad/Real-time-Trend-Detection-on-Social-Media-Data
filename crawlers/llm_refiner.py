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
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                model_id = model_path or "google/gemma-2-2b-it"
                console.print(f"[bold cyan]🤖 Loading {model_id} via Transformers...[/bold cyan]")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                # Use 4-bit quantization to fit in Kaggle T4/P100
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
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

    def refine_cluster(self, cluster_name, posts, original_category=None, topic_type="Discovery"):
        if not self.enabled:
            return cluster_name, original_category, ""

        context_texts = [p.get('content', '')[:300] for p in posts[:5]]
        context_str = "\n---\n".join(context_texts)

        prompt = f"""
Analyze this cluster of social media/news posts from Vietnam.
Original Label: {cluster_name}
Topic Type: {topic_type}
Sample Posts:
{context_str}

Tasks:
1. Create a professional, concise title for this event in Vietnamese (max 8 words). 
2. Classify into:
   - A: Critical (Accidents, Disasters, Safety)
   - B: Social (Policy, controversy, public sentiment)
   - C: Market (Commerce, Tech, Entertainment)
3. Provide a brief Vietnamese reasoning (1 sentence).

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

    def refine_batch(self, clusters_to_refine):
        if not self.enabled or not clusters_to_refine:
            return {}

        batch_str = ""
        for c in clusters_to_refine:
            context = "\n".join([p.get('content', '')[:200] for p in c['sample_posts'][:3]])
            batch_str += f"### ID: {c['label']}\nLabel: {c['name']}\nType: {c['topic_type']}\nContext: {context}\n\n"

        prompt = f"""
Analyze these {len(clusters_to_refine)} news/social clusters from Vietnam.
For each cluster ID, provide a professional title, category, and reasoning.

Categories:
- A: Critical (Accidents, Disasters, Safety)
- B: Social (Policy, controversy, public sentiment)
- C: Market (Commerce, Tech, Entertainment)

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
            results = json.loads(text[start:end])
            return {item['id']: item for item in results}
        except Exception as e:
            console.print(f"[red]Batch LLM error: {e}[/red]")
            return {}
