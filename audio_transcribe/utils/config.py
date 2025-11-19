"""
Configuration management for Audio Transcribe.
Handles loading and saving user preferences and API keys.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv, set_key

class ConfigManager:
    """Manages application configuration and user preferences."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.app_name = "audio_transcribe"
        self.config_dir = Path.home() / f".{self.app_name}"
        self.config_file = self.config_dir / "config.json"
        self.env_file = Path(".env")
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Load existing config
        self.config = self._load_config()
        
        # Load environment variables
        load_dotenv(self.env_file)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}

    def save_config(self) -> None:
        """Save current configuration to JSON file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and save."""
        self.config[key] = value
        self.save_config()

    def set_api_key(self, api_name: str, key: str) -> None:
        """
        Set an API key in the .env file.
        
        Args:
            api_name: Name of the API (e.g., 'openai', 'groq')
            key: The API key string
        """
        env_var = f"{api_name.upper()}_API_KEY"
        
        # Update current environment
        os.environ[env_var] = key
        
        # Update .env file
        try:
            # Create .env if it doesn't exist
            if not self.env_file.exists():
                self.env_file.touch()
                
            set_key(self.env_file, env_var, key)
            logger.info(f"Updated {env_var} in .env file")
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")

    def get_api_key(self, api_name: str) -> Optional[str]:
        """Get an API key from environment."""
        env_var = f"{api_name.upper()}_API_KEY"
        return os.getenv(env_var)
