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
    
    choices = questionary.checkbox(
        "Which APIs do you want to configure?",
        choices=apis
    ).ask()
    
    if not choices:
        console.print("No APIs selected. Exiting setup.")
        return
        
    for api_name in choices:
        console.print(f"\n[bold]Configuring {api_name}...[/bold]")
        
        current_key = config.get_api_key(api_name)
        if current_key:
            console.print(f"Current key found: {current_key[:4]}...{current_key[-4:]}")
            if not questionary.confirm(f"Do you want to update the key for {api_name}?").ask():
                continue
                
        new_key = questionary.password(f"Enter API Key for {api_name}:").ask()
        
        if new_key:
            console.print("Validating key...")
            try:
                # Initialize API with new key to test it
                api_instance = get_api_instance(api_name, api_key=new_key)
                if api_instance.check_api_key():
                    console.print(f"[green]✓ Key for {api_name} is valid![/green]")
                    config.set_api_key(api_name, new_key)
                else:
                    console.print(f"[red]✗ Key for {api_name} appears invalid.[/red]")
                    if questionary.confirm("Save it anyway?").ask():
                        config.set_api_key(api_name, new_key)
            except Exception as e:
                console.print(f"[red]Error validating key: {e}[/red]")
                if questionary.confirm("Save it anyway?").ask():
                    config.set_api_key(api_name, new_key)
                    
    console.print("\n[bold green]Setup complete![/bold green]")
