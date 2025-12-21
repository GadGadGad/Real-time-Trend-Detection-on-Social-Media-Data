import os
import json
import google.generativeai as genai
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()

class LLMRefiner:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            console.print("[yellow]⚠️ GEMINI_API_KEY not found. LLM Refinement will be disabled.[/yellow]")
            self.enabled = False
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            self.enabled = True

    def refine_cluster(self, cluster_name, posts, original_category=None, topic_type="Discovery"):
        """
        Refine cluster name and category using LLM.
        """
        if not self.enabled:
            return cluster_name, original_category, ""

        # OPTIMIZATION: Only refine if it's a Discovery topic or a high-impact trend
        # If it's already a well-matched Google Trend, we might just want to keep the trend name
        # but refine the category/reasoning.
        
        # Sample posts for context (Top 5)
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
   - A: Critical (Natural disasters, accidents, public safety threats)
   - B: Social (Policy changes, social controversy, public sentiment)
   - C: Market (Technology, Entertainment, Commerce, Lifestyle)
3. Provide a brief Vietnamese reasoning (1 sentence).

Respond STRICTLY in JSON:
{{
  "refined_title": "...",
  "category": "A/B/C",
  "reasoning": "..."
}}
"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            start = text.find('{')
            end = text.rfind('}') + 1
            data = json.loads(text[start:end])
            
            return data.get('refined_title', cluster_name), data.get('category', original_category), data.get('reasoning', "")
        except Exception as e:
            # console.print(f"[red]LLM error for {cluster_name}: {e}[/red]")
            return cluster_name, original_category, ""

    def refine_batch(self, clusters_to_refine):
        """
        Refine multiple clusters in a single API call for efficiency.
        clusters_to_refine: List of dicts {label, name, topic_type, category, sample_posts}
        """
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
- B: Social (Policy, controversy, public emotion)
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
            response = self.model.generate_content(prompt)
            text = response.text
            start = text.find('[')
            end = text.rfind(']') + 1
            results = json.loads(text[start:end])
            
            # Map back to dict
            return {item['id']: item for item in results}
        except Exception as e:
            console.print(f"[red]Batch LLM error: {e}[/red]")
            return {}
