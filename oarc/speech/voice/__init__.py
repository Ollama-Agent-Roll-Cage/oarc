"""
OARC Speech Voice Package.

This package provides voice-related functionality for the OARC speech system, including:
- Voice type definitions and enumerations
- Voice reference pack types and information
- Utility functions for voice processing

The package centralizes voice-related components to ensure consistent handling
across the OARC application.
"""

from .voice_type import VoiceType
from .voice_ref_pack_type import VoiceRefPackType, VoiceRefPackInfo
from .voice_utils import VoiceUtils

__all__ = [
    'VoiceType',
    'VoiceRefPackType',
    'VoiceRefPackInfo',
    'VoiceUtils'
]