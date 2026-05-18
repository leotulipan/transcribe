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
            # Handle speech_models array parameter
            if 'speech_models' in adapted and adapted['speech_models']:
                # Convert comma-separated string to list if needed
                if isinstance(adapted['speech_models'], str):
                    adapted['speech_models'] = [m.strip() for m in adapted['speech_models'].split(',')]
            # Handle keyterms_prompt parameter (comma-separated string to list)
            if 'keyterms_prompt' in adapted and adapted['keyterms_prompt']:
                if isinstance(adapted['keyterms_prompt'], str):
                    adapted['keyterms_prompt'] = [k.strip() for k in adapted['keyterms_prompt'].split(',')]
        
        # Add more adaptations as needed
        return adapted
