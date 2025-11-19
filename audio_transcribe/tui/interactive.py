"""
Interactive mode for Audio Transcribe.
"""
import os
from pathlib import Path
from typing import Dict, Any, List
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from audio_transcribe.utils.config import ConfigManager
from audio_transcribe.utils.models import MODEL_REGISTRY, get_available_models, get_default_model

console = Console()

def run_interactive_mode(file_path: str = None) -> Dict[str, Any]:
    """
    Run the interactive configuration mode.
    
    Args:
        file_path: Optional path to a file/folder if provided via drag-and-drop
        
    Returns:
        Dictionary of configuration options to pass to the main processing function
    """
    console.print(Panel.fit("Audio Transcribe - Interactive Mode", style="bold blue"))
    
    config = ConfigManager()
    options = {}
    
    # 1. File Selection (if not provided)
    if not file_path:
        file_path = questionary.path("Path to audio/video file or folder:").ask()
        if not file_path:
            console.print("[red]No file selected. Exiting.[/red]")
            return None
    else:
        console.print(f"Processing: [bold]{file_path}[/bold]")
        
    # Determine if file or folder
    path_obj = Path(file_path)
    if path_obj.is_dir():
        options["folder"] = str(path_obj)
        options["file"] = None
    else:
        options["file"] = str(path_obj)
        options["folder"] = None
        
    # 2. API Selection
    api_choices = []
    available_apis = ["assemblyai", "elevenlabs", "groq", "openai"]
    default_api = config.get("default_api", "groq")
    
    for api in available_apis:
        key = config.get_api_key(api)
        if key:
            api_choices.append(questionary.Choice(api, value=api))
        else:
            api_choices.append(questionary.Choice(f"{api} (not configured)", value=api, disabled="No API key found"))
            
    if not any(not choice.disabled for choice in api_choices):
        console.print("[red]No APIs configured. Please run 'transcribe setup' first.[/red]")
        return None

    selected_api = questionary.select(
        "Select Transcription API:",
        choices=api_choices,
        default=default_api if config.get_api_key(default_api) else None
    ).ask()
    
    if not selected_api:
        return None
        
    options["api"] = selected_api
    
    # Save as default?
    if selected_api != default_api:
        config.set("default_api", selected_api)
        
    # 3. Model Selection
    # Try to fetch models dynamically first
    dynamic_models = []
    try:
        # Get API key from config to initialize instance
        api_key = config.get_api_key(selected_api)
        if api_key:
            from audio_transcribe.utils.api import get_api_instance
            api_instance = get_api_instance(selected_api, api_key=api_key)
            with console.status(f"[bold green]Fetching available models for {selected_api}...[/bold green]"):
                dynamic_models = api_instance.list_models()
    except Exception as e:
        console.print(f"[yellow]Could not fetch models dynamically: {e}[/yellow]")
    
    # Use dynamic models if available, otherwise fallback to registry
    if dynamic_models:
        models = dynamic_models
        # If default model is not in the list, use the first one
        default_model = get_default_model(selected_api)
        if default_model not in models:
            default_model = models[0]
    else:
        models = get_available_models(selected_api)
        default_model = get_default_model(selected_api)
    
    if models:
        selected_model = questionary.select(
            "Select Model:",
            choices=models,
            default=default_model
        ).ask()
        options["model"] = selected_model
    else:
        # If no models found (e.g. API error and no static models), let user type it
        selected_model = questionary.text(
            "Enter Model ID:",
            default=default_model or ""
        ).ask()
        if selected_model:
            options["model"] = selected_model
    
    # 4. Language
    language = questionary.text("Language code (leave empty for auto-detect):").ask()
    if language:
        options["language"] = language.strip()
    else:
        options["language"] = None
        
    # 5. Output Formats
    format_choices = [
        questionary.Choice("text", checked=True),
        questionary.Choice("srt", checked=True),
        questionary.Choice("word_srt", checked=False),
        questionary.Choice("davinci_srt", checked=False),
        questionary.Choice("json", checked=False)
    ]
    
    selected_formats = questionary.checkbox(
        "Select Output Formats:",
        choices=format_choices
    ).ask()
    
    if not selected_formats:
        console.print("[yellow]No output format selected. Defaulting to text and srt.[/yellow]")
        options["output"] = ["text", "srt"]
    else:
        options["output"] = selected_formats
        
    # 6. Advanced Options (Optional)
    if questionary.confirm("Configure advanced options?", default=False).ask():
        # Filler words
        if questionary.confirm("Remove filler words?", default=False).ask():
            options["remove_fillers"] = True
        else:
            options["remove_fillers"] = False
            
        # Speaker Diarization
        if selected_api in ["assemblyai", "elevenlabs"]:
            if questionary.confirm("Enable speaker diarization?", default=True).ask():
                options["speaker_labels"] = True
                options["diarize"] = True
                
                num_speakers = questionary.text("Number of speakers (optional):").ask()
                if num_speakers and num_speakers.isdigit():
                    options["num_speakers"] = int(num_speakers)
            else:
                options["speaker_labels"] = False
                options["diarize"] = False
                
        # DaVinci Specifics
        if "davinci_srt" in options["output"]:
            options["davinci_srt"] = True # Flag for logic
            if questionary.confirm("Output filler words as UPPERCASE lines?", default=False).ask():
                options["filler_lines"] = True
                options["silent_portions"] = 350 # Default for this mode
            else:
                options["filler_lines"] = False
                
    # Summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Input", options["file"] or options["folder"])
    table.add_row("API", options["api"])
    if "model" in options:
        table.add_row("Model", options["model"])
    table.add_row("Language", options["language"] or "Auto")
    table.add_row("Outputs", ", ".join(options["output"]))
    
    console.print(table)
    
    if not questionary.confirm("Start Transcription?").ask():
        console.print("Operation cancelled.")
        return None
        
    return options
