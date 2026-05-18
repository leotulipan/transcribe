"""
Language utilities for transcription services.

This module provides functions for handling language codes across different APIs.
"""

from typing import Dict, Optional, Set
from loguru import logger

# Mapping of language codes
# Format: 
#   key: common name or code
#   value: {
#     'iso639_1': two-letter code (ISO 639-1),
#     'iso639_3': three-letter code (ISO 639-3),
#     'name': English name of language,
#     'native_name': Native name of language,
#     'assemblyai': preferred code for AssemblyAI,
#     'elevenlabs': preferred code for ElevenLabs,
#     'groq': preferred code for Groq,
#     'openai': preferred code for OpenAI,
#   }
# Not all keys are required for every language

LANGUAGE_MAP = {
    # European languages
    'en': {
        'iso639_1': 'en',
        'iso639_3': 'eng',
        'name': 'English',
        'native_name': 'English',
        'assemblyai': 'en',
        'elevenlabs': 'en',
        'groq': 'en',
        'openai': 'en'
    },
    'de': {
        'iso639_1': 'de',
        'iso639_3': 'deu',
        'name': 'German',
        'native_name': 'Deutsch',
        'assemblyai': 'de',
        'elevenlabs': 'de',
        'groq': 'de',
        'openai': 'de'
    },
    'fr': {
        'iso639_1': 'fr',
        'iso639_3': 'fra',
        'name': 'French',
        'native_name': 'Français',
        'assemblyai': 'fr',
        'elevenlabs': 'fr',
        'groq': 'fr',
        'openai': 'fr'
    },
    'es': {
        'iso639_1': 'es',
        'iso639_3': 'spa',
        'name': 'Spanish',
        'native_name': 'Español',
        'assemblyai': 'es',
        'elevenlabs': 'es',
        'groq': 'es',
        'openai': 'es'
    },
    'it': {
        'iso639_1': 'it',
        'iso639_3': 'ita',
        'name': 'Italian',
        'native_name': 'Italiano',
        'assemblyai': 'it',
        'elevenlabs': 'it',
        'groq': 'it',
        'openai': 'it'
    },
    'nl': {
        'iso639_1': 'nl',
        'iso639_3': 'nld',
        'name': 'Dutch',
        'native_name': 'Nederlands',
        'assemblyai': 'nl',
        'elevenlabs': 'nl',
        'groq': 'nl',
        'openai': 'nl'
    },
    'pl': {
        'iso639_1': 'pl',
        'iso639_3': 'pol',
        'name': 'Polish',
        'native_name': 'Polski',
        'assemblyai': 'pl',
        'elevenlabs': 'pl',
        'groq': 'pl',
        'openai': 'pl'
    },
    'pt': {
        'iso639_1': 'pt',
        'iso639_3': 'por',
        'name': 'Portuguese',
        'native_name': 'Português',
        'assemblyai': 'pt',
        'elevenlabs': 'pt',
        'groq': 'pt',
        'openai': 'pt'
    },
    'ru': {
        'iso639_1': 'ru',
        'iso639_3': 'rus',
        'name': 'Russian',
        'native_name': 'Русский',
        'assemblyai': 'ru',
        'elevenlabs': 'ru',
        'groq': 'ru',
        'openai': 'ru'
    },
    'sv': {
        'iso639_1': 'sv',
        'iso639_3': 'swe',
        'name': 'Swedish',
        'native_name': 'Svenska',
        'assemblyai': 'sv',
        'elevenlabs': 'sv',
        'groq': 'sv',
        'openai': 'sv'
    },
    
    # Asian languages
    'ja': {
        'iso639_1': 'ja',
        'iso639_3': 'jpn',
        'name': 'Japanese',
        'native_name': '日本語',
        'assemblyai': 'ja',
        'elevenlabs': 'ja',
        'groq': 'ja',
        'openai': 'ja'
    },
    'zh': {
        'iso639_1': 'zh',
        'iso639_3': 'zho',
        'name': 'Chinese',
        'native_name': '中文',
        'assemblyai': 'zh',
        'elevenlabs': 'zh',
        'groq': 'zh',
        'openai': 'zh'
    },
    'ko': {
        'iso639_1': 'ko',
        'iso639_3': 'kor',
        'name': 'Korean',
        'native_name': '한국어',
        'assemblyai': 'ko',
        'elevenlabs': 'ko',
        'groq': 'ko',
        'openai': 'ko'
    },
    
    # Add variants and aliases
    'deu': {
        'iso639_1': 'de',
        'iso639_3': 'deu',
        'name': 'German',
        'native_name': 'Deutsch',
        'assemblyai': 'de',
        'elevenlabs': 'de',
        'groq': 'de',
        'openai': 'de'
    },
    'german': {
        'iso639_1': 'de',
        'iso639_3': 'deu',
        'name': 'German',
        'native_name': 'Deutsch',
        'assemblyai': 'de',
        'elevenlabs': 'de',
        'groq': 'de',
        'openai': 'de'
    },
    'english': {
        'iso639_1': 'en',
        'iso639_3': 'eng',
        'name': 'English',
        'native_name': 'English',
        'assemblyai': 'en',
        'elevenlabs': 'en',
        'groq': 'en',
        'openai': 'en'
    },
    'eng': {
        'iso639_1': 'en',
        'iso639_3': 'eng',
        'name': 'English',
        'native_name': 'English',
        'assemblyai': 'en',
        'elevenlabs': 'en',
        'groq': 'en',
        'openai': 'en'
    },
    'fra': {
        'iso639_1': 'fr',
        'iso639_3': 'fra',
        'name': 'French',
        'native_name': 'Français',
        'assemblyai': 'fr',
        'elevenlabs': 'fr',
        'groq': 'fr',
        'openai': 'fr'
    },
    'french': {
        'iso639_1': 'fr',
        'iso639_3': 'fra',
        'name': 'French',
        'native_name': 'Français',
        'assemblyai': 'fr',
        'elevenlabs': 'fr',
        'groq': 'fr',
        'openai': 'fr'
    },
    'spanish': {
        'iso639_1': 'es',
        'iso639_3': 'spa',
        'name': 'Spanish',
        'native_name': 'Español',
        'assemblyai': 'es',
        'elevenlabs': 'es',
        'groq': 'es',
        'openai': 'es'
    },
    'spa': {
        'iso639_1': 'es',
        'iso639_3': 'spa',
        'name': 'Spanish',
        'native_name': 'Español',
        'assemblyai': 'es',
        'elevenlabs': 'es',
        'groq': 'es',
        'openai': 'es'
    },
}

def get_language_code(language_input: str, api_name: str) -> Optional[str]:
    """
    Convert a language code or name to the appropriate format for the specified API.
    
    Args:
        language_input: Language code (ISO-639-1, ISO-639-3) or name
        api_name: Name of the API to get the code for
    
    Returns:
        Converted language code appropriate for the API or None if not found
    """
    if not language_input:
        return None
        
    # Convert to lowercase for case-insensitive matching
    language_input = language_input.lower().strip()
    api_name = api_name.lower().strip()
    
    # Check if we have this language in our map
    if language_input in LANGUAGE_MAP:
        lang_data = LANGUAGE_MAP[language_input]
        
        # Return API-specific code if available
        if api_name in lang_data:
            return lang_data[api_name]
            
        # Fall back to ISO-639-1 if API-specific code not found
        return lang_data.get('iso639_1')
    
    # If not found directly, try to find a match based on name or aliases
    for lang_code, lang_data in LANGUAGE_MAP.items():
        if (lang_data.get('name', '').lower() == language_input or 
            lang_data.get('native_name', '').lower() == language_input):
            
            # Return API-specific code if available
            if api_name in lang_data:
                return lang_data[api_name]
                
            # Fall back to ISO-639-1
            return lang_data.get('iso639_1')
    
    # Log a warning but still return the input code as a fallback
    logger.warning(f"Unknown language code '{language_input}' for API '{api_name}'. Using as-is.")
    return language_input

def get_supported_languages() -> Set[str]:
    """
    Get a set of all supported language codes and names.
    
    Returns:
        Set of supported language identifiers
    """
    supported = set()
    
    for lang_code, lang_data in LANGUAGE_MAP.items():
        supported.add(lang_code)
        if 'name' in lang_data:
            supported.add(lang_data['name'].lower())
        if 'native_name' in lang_data:
            supported.add(lang_data['native_name'].lower())
        if 'iso639_1' in lang_data:
            supported.add(lang_data['iso639_1'])
        if 'iso639_3' in lang_data:
            supported.add(lang_data['iso639_3'])
    
    return supported

def is_language_supported(language: str) -> bool:
    """
    Check if a language is supported by the system.
    
    Args:
        language: Language code or name to check
    
    Returns:
        True if the language is supported, False otherwise
    """
    if not language:
        return False
        
    language = language.lower().strip()
    supported_languages = get_supported_languages()
    
    return language in supported_languages 