
import os
import sys
import json
import argparse
from rich.console import Console

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.core.llm.llm_refiner import LLMRefiner

console = Console()

def debug_llm(provider, model_path, api_key=None):
    console.print(f"[bold cyan]🔍 Debugging LLM Provider: {provider}[/bold cyan]")
    console.print(f"[bold cyan]🤖 Model Path: {model_path}[/bold cyan]")
    
    if api_key:
        console.print(f"[dim]🔑 API Key provided (truncated): {api_key[:6]}...[/dim]")
    else:
        console.print("[dim]⚠️ No explicit API Key provided (checking env).[/dim]")

    try:
        refiner = LLMRefiner(provider=provider, model_path=model_path, api_key=api_key, debug=True)
        
        if not refiner.enabled:
            console.print("[red]❌ LLM Refiner failed to initialize (enabled=False). Check logs above.[/red]")
            return

        prompt = """
        Role: System Check.
        Task: Return a JSON object with a greeting.
        
        Output format:
        {
            "status": "ok",
            "message": "Hello from LLM"
        }
        """
        
        console.print("\n[yellow]🚀 Sending Test Prompt...[/yellow]")
        console.print(f"[dim]{prompt.strip()}[/dim]")
        
        # Call internal generate directly to skip batching logic first
        response = refiner._generate(prompt)
        
        console.print("\n[bold green]📥 Raw Response Received:[/bold green]")
        console.print("-" * 50)
        console.print(response)
        console.print("-" * 50)
        
        # Attempt extraction
        console.print("\n[yellow]🕵️ Attempting JSON Extraction...[/yellow]")
        data = refiner._extract_json(response, is_list=False)
        
        if data:
            console.print("[bold green]✅ JSON Parsed Successfully:[/bold green]")
            console.print(json.dumps(data, indent=2))
        else:
            console.print("[bold red]❌ JSON Parse Failed.[/bold red]")
            console.print("Common causes: Markdown code blocks, preamble text, or model hallucination.")

    except Exception as e:
        console.print(f"[bold red]❌ fatal Error during execution: {e}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug LLM API Call")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "kaggle", "local"], help="LLM Provider")
    parser.add_argument("--model", type=str, default=None, help="Model Path/Name (e.g. gemini-1.5-flash or gemma-2-9b)")
    parser.add_argument("--key", type=str, default=None, help="API Key (optional)")
    
    args = parser.parse_args()
    
    debug_llm(args.provider, args.model, args.key)
