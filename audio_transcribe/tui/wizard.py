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
        
        choices.append(questionary.Choice("Exit", value="exit"))
        
        selected_action = questionary.select(
            "Main Menu:",
            choices=choices
        ).ask()
        
        if selected_action == "exit" or selected_action is None:
            console.print("Exiting setup.")
            break
            
        api_name = selected_action
        configure_api(api_name, config)

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
