"""
Model definitions and registry for transcription APIs.
"""

MODEL_REGISTRY = {
    "openai": {
        "default": "whisper-1",
        "models": [
            "whisper-1"
        ]
    },
    "groq": {
        "default": "whisper-large-v3",
        "models": [
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "distil-whisper-large-v3-en"
        ]
    },
    "assemblyai": {
        "default": "best",
        "models": [
            "best",
            "nano"
        ]
    },
    "elevenlabs": {
        "default": "scribe_v1",
        "models": [
            "scribe_v1"
        ]
    }
}

def get_default_model(api_name: str) -> str:
    """Get the default model for a given API."""
    return MODEL_REGISTRY.get(api_name, {}).get("default", "")

def get_available_models(api_name: str) -> list:
    """Get list of available models for a given API."""
    return MODEL_REGISTRY.get(api_name, {}).get("models", [])
