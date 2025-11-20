"""
Setup wizard for configuring Audio Transcribe.
"""
import questionary
from rich.console import Console
from rich.panel import Panel
from audio_transcribe.utils.config import ConfigManager
from audio_transcribe.utils.api import get_api_instance

console = Console()

def run_setup_wizard():
    """Run the interactive setup wizard."""
    console.print(Panel.fit("Audio Transcribe Setup", style="bold blue"))
    
    config = ConfigManager()
    apis = ["assemblyai", "elevenlabs", "groq", "openai"]
    
    while True:
        # Build menu choices
        choices = []
        for api in apis:
            key = config.get_api_key(api)
            status = "(Configured)" if key else "(Not Configured)"
            choices.append(questionary.Choice(f"Configure {api} {status}", value=api))
        
        choices.append(questionary.Choice("Configure Defaults", value="defaults"))
        choices.append(questionary.Choice("Exit", value="exit"))
        
        selected_action = questionary.select(
            "Main Menu:",
            choices=choices
        ).ask()
        
        if selected_action == "exit" or selected_action is None:
            console.print("Exiting setup.")
            break
            
        if selected_action == "defaults":
            configure_defaults(config)
        else:
            api_name = selected_action
            configure_api(api_name, config)

def configure_defaults(config: ConfigManager):
    """Configure default settings."""
    console.print("\n[bold]Configuring Defaults...[/bold]")
    
    while True:
        current_api = config.get("default_api", "Not set")
        current_lang = config.get("default_language", "Auto")
        current_outputs = config.get("default_output_formats", ["text", "srt"])
        current_outputs_str = ", ".join(current_outputs)
        
        action = questionary.select(
            "Defaults Menu:",
            choices=[
                questionary.Choice(f"Default API ({current_api})", value="api"),
                questionary.Choice(f"Default Language ({current_lang})", value="language"),
                questionary.Choice(f"Default Outputs ({current_outputs_str})", value="outputs"),
                questionary.Choice("Back", value="back")
            ]
        ).ask()
        
        if action == "back" or action is None:
            break
            
        if action == "api":
            apis = ["assemblyai", "elevenlabs", "groq", "openai"]
            selected = questionary.select("Select Default API:", choices=apis).ask()
            if selected:
                config.set("default_api", selected)
                console.print(f"[green]Default API set to {selected}[/green]")
                
        elif action == "language":
            lang = questionary.text("Enter Default Language Code (e.g. 'en', 'de') or leave empty for Auto:").ask()
            if lang is not None:
                val = lang.strip() if lang.strip() else None
                config.set("default_language", val)
                console.print(f"[green]Default Language set to {val if val else 'Auto'}[/green]")
                
        elif action == "outputs":
            choices = [
                questionary.Choice("text", checked="text" in current_outputs),
                questionary.Choice("srt", checked="srt" in current_outputs),
                questionary.Choice("word_srt", checked="word_srt" in current_outputs),
                questionary.Choice("davinci_srt", checked="davinci_srt" in current_outputs),
                questionary.Choice("json", checked="json" in current_outputs)
            ]
            selected = questionary.checkbox("Select Default Output Formats:", choices=choices).ask()
            if selected is not None:
                config.set("default_output_formats", selected)
                console.print(f"[green]Default Outputs updated[/green]")

def configure_api(api_name: str, config: ConfigManager):
    """Configure a specific API."""
    console.print(f"\n[bold]Configuring {api_name}...[/bold]")
    
    current_key = config.get_api_key(api_name)
    new_key = None
    
    if current_key:
        masked_key = f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else "****"
        console.print(f"Current key: {masked_key}")
        
        action = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Keep existing key", value="keep"),
                questionary.Choice("Update key", value="update"),
                questionary.Choice("Back", value="back")
            ]
        ).ask()
        
        if action == "back" or action == "keep" or action is None:
            return
            
    new_key = questionary.password(f"Enter API Key for {api_name}:").ask()
    
    if new_key:
        console.print("Validating key...")
        try:
            # Initialize API with new key to test it
            api_instance = get_api_instance(api_name, api_key=new_key)
            if api_instance.check_api_key():
                console.print(f"[green]✓ Key for {api_name} is valid![/green]")
                config.set_api_key(api_name, new_key)
                console.print(f"[green]Key saved for {api_name}[/green]")
                questionary.press_any_key_to_continue().ask()
            else:
                console.print(f"[red]✗ Key for {api_name} appears invalid.[/red]")
                if questionary.confirm("Save it anyway?").ask():
                    config.set_api_key(api_name, new_key)
                    console.print(f"[yellow]Key saved (unverified) for {api_name}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error validating key: {e}[/red]")
            if questionary.confirm("Save it anyway?").ask():
                config.set_api_key(api_name, new_key)
                console.print(f"[yellow]Key saved (unverified) for {api_name}[/yellow]")
