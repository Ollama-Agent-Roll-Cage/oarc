"""
Defines the TTSRequest model using Pydantic for handling text-to-speech request data.
This module encapsulates the configuration for voice name, speed, language, and text.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from oarc.utils.log import log


class TTSRequest(BaseModel):
    """
    Request model for text-to-speech operations.
    
    This model validates and stores parameters needed for text-to-speech synthesis,
    with configurable defaults instead of hardcoded values.
    """
    text: str
    voice_name: str = None
    speed: float = None
    language:str = None
    
    @classmethod
    def create(cls, text: str, defaults: Dict[str, Any] = None, **kwargs) -> 'TTSRequest':
        """
        Create a TTSRequest with configurable defaults.
        
        Args:
            text: The text to convert to speech
            defaults: Dictionary of default values to use if not provided in kwargs
            **kwargs: Override specific parameters
            
        Returns:
            TTSRequest: A configured request object
        """
        # Use empty dict if None to avoid mutable default issues
        defaults = defaults or {}
        
        # Default values if not provided in defaults dict
        default_voice = defaults.get('voice_name', 'c3po')
        default_speed = defaults.get('speed', 1.0)
        default_language = defaults.get('language', 'en')
        
        # Create parameters dict with defaults, overridden by any provided kwargs
        params = {
            'text': text,
            'voice_name': kwargs.get('voice_name', default_voice),
            'speed': kwargs.get('speed', default_speed),
            'language': kwargs.get('language', default_language)
        }
        
        # Log the configuration for debugging
        log.debug(f"Creating TTSRequest with: {params}")
        
        # Create and return the instance
        return cls(**params)