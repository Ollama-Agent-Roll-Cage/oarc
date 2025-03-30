"""
Defines the TTSRequest model using Pydantic for handling text-to-speech request data.
This module encapsulates the configuration for voice name, speed, language, and text.
"""

import logging

from pydantic import BaseModel

# Create a proper logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class TTSRequest(BaseModel):
    text: str
    voice_name: str = "c3po"
    speed: float = 1.0
    language: str = "en"