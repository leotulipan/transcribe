"""
Mixin for chunked transcription logic.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
from loguru import logger
from audio_transcribe.transcribe_helpers.chunking import split_audio

class ChunkingMixin(ABC):
    """Mixin for APIs that support chunked transcription."""
    
    def transcribe_with_chunking(self, audio_path: Union[str, Path], chunk_length: int = 600, overlap: int = 10, **kwargs) -> Any:
        """
        Generic chunking logic for any API.
        
        Args:
            audio_path: Path to the audio file
            chunk_length: Length of chunks in seconds
            overlap: Overlap between chunks in seconds
            **kwargs: Additional arguments passed to transcribe_chunk
            
        Returns:
            Merged transcription result
        """
        audio_path = Path(audio_path)
        logger.info(f"Splitting audio into chunks (length={chunk_length}s, overlap={overlap}s)...")
        
        chunks = split_audio(audio_path, chunk_length, overlap)
        results = []
        
        logger.info(f"Created {len(chunks)} chunks. Starting transcription...")
        
        try:
            for i, (chunk_path, start_time_seconds) in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk_path.name} (start: {start_time_seconds:.1f}s)")
                
                # Call abstract method to transcribe single chunk (start_time is in seconds)
                result = self.transcribe_chunk(chunk_path, chunk_index=i, start_time=start_time_seconds, **kwargs)
                
                if result:
                    results.append(result)
                else:
                    logger.warning(f"Chunk {i+1} returned no result.")
                    
        except Exception as e:
            logger.error(f"Error during chunked transcription: {e}")
            raise
        finally:
            # Cleanup chunks
            for chunk_path, _ in chunks:
                if chunk_path.exists():
                    try:
                        chunk_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete chunk {chunk_path}: {e}")
                        
        # Merge results
        if not results:
            logger.error("No results obtained from chunks.")
            return None
            
        return self.merge_chunk_results(results)
    
    @abstractmethod
    def transcribe_chunk(self, chunk_path: Path, chunk_index: int, start_time: float, **kwargs) -> Any:
        """
        Transcribe a single chunk. Must be implemented by subclass.
        
        Args:
            chunk_path: Path to the chunk audio file
            chunk_index: Index of the chunk
            start_time: Start time of the chunk in seconds
            **kwargs: Additional arguments
            
        Returns:
            Transcription result for the chunk
        """
        pass
    
    @abstractmethod
    def merge_chunk_results(self, results: List[Any]) -> Any:
        """
        Merge multiple chunk results into one. Must be implemented by subclass.
        
        Args:
            results: List of transcription results
            
        Returns:
            Merged transcription result
        """
        pass
