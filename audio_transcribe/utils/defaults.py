"""
Centralized defaults management for Audio Transcribe.
"""
from typing import Dict, Any, Optional
from audio_transcribe.utils.config import ConfigManager

class DefaultsManager:
    """Centralized defaults management."""
    
    # API defaults
    API_DEFAULTS = {
        'groq': {'model': 'whisper-large-v3'},
        'openai': {'model': 'whisper-1'},
        'assemblyai': {'model': 'best', 'speaker_labels': True, 'disfluencies': True},
        'elevenlabs': {'model_id': 'scribe_v1'}
    }
    
    # Output format defaults
    FORMAT_DEFAULTS = {
        'chars_per_line': 80,
        'words_per_subtitle': 0,
        'silent_portions': 0,
        'padding_start': 0,
        'padding_end': 0,
        'show_pauses': False,
        'remove_fillers': False,
        'speaker_labels': True,
        'start_hour': 0
    }
    
    # Format presets
    PRESETS = {
        'davinci': {
            'chars_per_line': 500,
            'silent_portions': 250,
            'padding_start': -125,
            'remove_fillers': True,
            'show_pauses': True,
            'start_hour': 1
        }
    }
    
    @staticmethod
    def get_effective_params(api_name: str, user_params: Dict[str, Any], preset: str = None) -> Dict[str, Any]:
        """
        Merge defaults with user params using precedence:
        1. User explicit params (highest)
        2. Preset params (if preset specified)
        3. Config file params
        4. API defaults
        5. Format defaults (lowest)
        """
        # Load from config
        config = ConfigManager()
        config_defaults = config.config  # Access the config dictionary directly
        
        # Merge in order (lowest precedence first)
        effective = {}
        effective.update(DefaultsManager.FORMAT_DEFAULTS)
        effective.update(DefaultsManager.API_DEFAULTS.get(api_name, {}))
        effective.update(config_defaults)
        
        if preset:
            # If preset is a boolean (e.g. davinci_srt=True), map it to preset name
            if preset is True and 'davinci' in DefaultsManager.PRESETS:
                 effective.update(DefaultsManager.PRESETS['davinci'])
            elif isinstance(preset, str) and preset in DefaultsManager.PRESETS:
                effective.update(DefaultsManager.PRESETS[preset])
        
        # User params override everything (only if not None)
        # We filter out None values to allow defaults to shine through
        user_overrides = {k: v for k, v in user_params.items() if v is not None}
        effective.update(user_overrides)
        
        return effective
