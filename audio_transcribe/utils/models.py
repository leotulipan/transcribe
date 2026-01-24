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
    "openai_extended": {
        "default": "gpt-4o-transcribe",
        "models": [
            "whisper-1",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe-diarize"
        ],
        "note": "gpt-4o-transcribe-diarize supports speaker diarization"
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
        "default": "scribe_v2",
        "models": [
            "scribe_v1", "scribe_v2"
        ]
    },
    "gemini": {
        "default": "gemini-2.0-flash-exp",
        "models": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ],
        "note": "Text-only output (no timestamps)"
    },
    "mistral": {
        "default": "voxtral-mini-2507",
        "models": [
            "voxtral-mini-2507"
        ],
        "note": "Auto-detects language (cannot specify), segment-level timestamps only"
    }
}

def get_default_model(api_name: str) -> str:
    """Get the default model for a given API."""
    return MODEL_REGISTRY.get(api_name, {}).get("default", "")

def get_available_models(api_name: str) -> list:
    """Get list of available models for a given API."""
    return MODEL_REGISTRY.get(api_name, {}).get("models", [])
