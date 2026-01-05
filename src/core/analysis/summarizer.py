import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rich.console import Console
import gc

console = Console()

# Available Vietnamese summarization models
SUMMARIZATION_MODELS = {
    'vit5-large': 'VietAI/vit5-large-vietnews-summarization',  # Best quality, ~1.2GB
    'vit5-base': 'VietAI/vit5-base-vietnews-summarization',    # Faster, ~900MB
    'bartpho': 'vinai/bartpho-syllable',                       # Alternative, good for short text
    'gemini': 'gemini-1.5-flash',                              # Cloud LLM, handles massive batches
}

class Summarizer:
    def __init__(self, model_name="VietAI/vit5-large-vietnews-summarization", device=None):
        # Allow shorthand model names
        if model_name in SUMMARIZATION_MODELS:
            model_name = SUMMARIZATION_MODELS[model_name]
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.enabled = False

    def load_model(self):
        if self.model_name.startswith('gemini'):
            from src.core.llm.llm_refiner import LLMRefiner
            self.model = LLMRefiner(provider="gemini", model_path=self.model_name)
            self.enabled = self.model.enabled
            if self.enabled:
                console.print(f"[green]✅ Summarizer using Gemini ({self.model_name})[/green]")
            return

        console.print(f"[cyan]📥 Loading Summarizer: {self.model_name}...[/cyan]")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.enabled = True
            console.print(f"[green]✅ Summarizer loaded on {self.device}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load Summarizer: {e}[/red]")
            self.enabled = False

    def unload_model(self):
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.enabled = False
            gc.collect()
            torch.cuda.empty_cache()
            console.print("[dim]🗑️ Summarizer unloaded to free GPU.[/dim]")

    def summarize_batch(self, texts, max_length=256):
        if not self.enabled: 
            self.load_model()
            if not self.enabled: return texts # Fallback
            
        if self.model_name.startswith('gemini'):
            summaries = []
            gemini_batch_size = 20 # Can handle more, but 20 is safe for response parsing
            for i in range(0, len(texts), gemini_batch_size):
                chunk = texts[i : i + gemini_batch_size]
                chunk_str = "\n".join([f"--- ARTICLE {idx+1} ---\n{t[:2000]}" for idx, t in enumerate(chunk)])
                prompt = (
                    "Role: Senior News Editor. Summarize each article into ONE concise Vietnamese paragraph (max 100 words).\n"
                    "Style: Neutral, journalistic, factual. No \"About this article...\" or \"The article discusses...\". Direct summary only.\n"
                    "Must Include: Key entities, numbers, dates, locations.\n"
                    "Output: A JSON list of strings. Each summary corresponds to the input article order.\n"
                    "Example: [\"Hà Nội đón không khí lạnh tăng cường từ đêm 15/10...\", \"Vingroup công bố báo cáo tài chính...\"]\n\n"
                    f"{chunk_str}\n\n"
                    "JSON Output:"
                )
                try:
                    res_text = self.model._generate(prompt)
                    chunk_summaries = self.model._extract_json(res_text, is_list=True)
                    if chunk_summaries and len(chunk_summaries) == len(chunk):
                        summaries.extend(chunk_summaries)
                    else:
                        console.print(f"[yellow]⚠️ Batch summary failed/mismatched for chunk {i//gemini_batch_size}. Retrying individually...[/yellow]")
                        for t in chunk:
                            summaries.append(self.model.summarize_text(t))
                except Exception as e:
                    console.print(f"[red]Gemini Batch Summarization Error: {e}[/red]")
                    summaries.extend([t[:256] for t in chunk]) # Truncation fallback
            return summaries

        console.print(f"[cyan]📝 Summarizing {len(texts)} articles with {self.model_name.split('/')[-1]}...[/cyan]")
        summaries = []
        batch_size = 4 # T4 GPU safe limit
        
        from rich.progress import track
        
        for i in track(range(0, len(texts), batch_size), description="[cyan]Summarizing batches...[/cyan]"):
            batch = texts[i : i + batch_size]
            formatted_batch = [f"vietnews: {text} </s>" for text in batch]
            
            inputs = self.tokenizer(
                formatted_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    early_stopping=True
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.extend(decoded)
            
        return summaries

    def sanity_check(self, texts, n_samples=3):
        """
        Run a quick sanity check on summarization quality.
        Shows original vs summary for n random samples.
        """
        import random
        
        if not self.enabled:
            self.load_model()
            if not self.enabled:
                console.print("[red]Cannot run sanity check - model failed to load[/red]")
                return
        
        # Pick random samples
        samples = random.sample(texts, min(n_samples, len(texts)))
        summaries = self.summarize_batch(samples, max_length=150)
        
        console.print("\n[bold cyan]📋 Summarization Sanity Check[/bold cyan]")
        console.print(f"Model: {self.model_name}\n")
        
        for i, (orig, summ) in enumerate(zip(samples, summaries)):
            compression = len(summ) / len(orig) * 100 if orig else 0
            console.print(f"[bold]--- Sample {i+1} ---[/bold]")
            console.print(f"[dim]Original ({len(orig)} chars):[/dim]")
            console.print(f"  {orig[:300]}...")
            console.print(f"[green]Summary ({len(summ)} chars, {compression:.0f}% of original):[/green]")
            console.print(f"  {summ}")
            console.print()
        
        # Quality metrics
        avg_compression = sum(len(s)/len(o) for o, s in zip(samples, summaries)) / len(samples) * 100
        avg_summary_len = sum(len(s) for s in summaries) / len(summaries)
        
        console.print("[bold]Quality Metrics:[/bold]")
        console.print(f"  Avg compression ratio: {avg_compression:.0f}%")
        console.print(f"  Avg summary length: {avg_summary_len:.0f} chars")
        console.print(f"  ✅ Good if compression < 30% and summaries are coherent")
