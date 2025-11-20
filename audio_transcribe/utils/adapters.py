"""
Parameter adapter for normalizing API parameters.
"""
from typing import Dict, Any

class ParameterAdapter:
    """Adapter to normalize generic parameters to API-specific parameters."""
    
    @staticmethod
    def adapt_for_api(api_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate generic params to API-specific params.
        
        Args:
            api_name: Name of the API (e.g., 'elevenlabs', 'assemblyai')
            params: Dictionary of generic parameters
            
        Returns:
            Dictionary of API-specific parameters
        """
        adapted = params.copy()
        
        if api_name == "elevenlabs":
            if 'model' in adapted:
                adapted['model_id'] = adapted.pop('model')
            if 'num_speakers' in adapted:
                # ElevenLabs doesn't have num_speakers in the same way, ignore or adapt if needed
                pass
        
        elif api_name == "assemblyai":
            if 'num_speakers' in adapted:
                adapted['speakers_expected'] = adapted.pop('num_speakers')
            if 'diarize' in adapted:
                adapted['speaker_labels'] = adapted.pop('diarize')
        
        # Add more adaptations as needed
        return adapted
