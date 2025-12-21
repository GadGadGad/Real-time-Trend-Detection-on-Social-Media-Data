import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rich.console import Console
import gc

console = Console()

class Summarizer:
    def __init__(self, model_name="VietAI/vit5-large-vietnews-summarization", device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.enabled = False

    def load_model(self):
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
            
        console.print(f"[cyan]📝 Summarizing {len(texts)} long articles with ViT5...[/cyan]")
        summaries = []
        batch_size = 4 # T4 GPU safe limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(decoded)
            
        return summaries
