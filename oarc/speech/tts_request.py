"""
Defines the TTSRequest model using Pydantic for handling text-to-speech request data.
This module encapsulates the configuration for voice name, speed, language, and text.
"""

from pydantic import BaseModel
from oarc.utils.log import log


class TTSRequest(BaseModel):
    text: str
    voice_name: str = "c3po"
    speed: float = 1.0
    language: str = "en"