"""
Defines the TTSRequest model using Pydantic for handling text-to-speech request data.
This module encapsulates the configuration for voice name, speed, language, and text.
"""

from pydantic import BaseModel, field_validator
from oarc.utils.log import log


class TTSRequest(BaseModel):
    """
    Request model for text-to-speech operations.
    
    This model validates and stores parameters needed for text-to-speech synthesis.
    All fields are required with no defaults.
    """
    text: str
    voice_name: str 
    speed: float
    language: str
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty. Please provide text to convert to speech.")
        return v
    
    @field_validator('voice_name')
    @classmethod
    def validate_voice_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Voice name cannot be empty. Please specify a voice (e.g., 'c3po', 'darth_vader').")
        return v
    
    @field_validator('speed')
    @classmethod
    def validate_speed(cls, v):
        if v is None:
            raise ValueError("Speed cannot be None. Please specify a speed value (e.g., 1.0 for normal speed).")
        if v <= 0:
            raise ValueError(f"Speed must be positive, got {v}")
        return v
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if not v or not v.strip():
            raise ValueError("Language cannot be empty. Please specify a language code (e.g., 'en' for English).")
        return v
    
    @classmethod
    def create(cls, text: str, voice_name: str, speed: float, language: str) -> 'TTSRequest':
        """
        Create a TTSRequest with all required parameters.
        
        Args:
            text: The text to convert to speech
            voice_name: The name of the voice to use
            speed: The speed factor for speech synthesis
            language: The language code for speech synthesis
            
        Returns:
            TTSRequest: A configured request object
            
        Raises:
            ValueError: If any required parameters are missing or invalid
        """
        # Validate parameters before creating instance
        if not text:
            raise ValueError("Missing required parameter: 'text'")
        if not voice_name:
            raise ValueError("Missing required parameter: 'voice_name'")
        if speed is None:
            raise ValueError("Missing required parameter: 'speed'")
        if not language:
            raise ValueError("Missing required parameter: 'language'")
        
        # Create parameters dict
        params = {
            'text': text,
            'voice_name': voice_name,
            'speed': speed,
            'language': language
        }
        
        # Log the configuration for debugging
        log.debug(f"Creating TTSRequest with: {params}")
        
        # Create and return the instance
        return cls(**params)