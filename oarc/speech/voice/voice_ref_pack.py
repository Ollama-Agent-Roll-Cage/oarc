"""
Voice Reference Pack Types Enumeration.

This module defines the voice reference pack types available in OARC.
Each entry represents a voice model that can be used by the TTS system,
including its Hugging Face repository URL.
"""

from enum import Enum
from dataclasses import dataclass

from oarc.speech.voice.voice_type import VoiceType


@dataclass
class VoiceRefPack:
    """Information about a voice reference pack."""
    name: str
    repo_url: str
    creator: str
    voice_type: VoiceType
    description: str = ""


class VoiceRefPackType(Enum):
    """Enum for available voice reference packs.
    
    Each enum value contains metadata about a specific voice model,
    including its Hugging Face repository and creator.
    """
    # Base model
    XTTS_V2_BASE = VoiceRefPack(
        name="XTTS-v2",
        repo_url="https://hf.co/coqui/XTTS-v2",
        creator="Coqui",
        voice_type=VoiceType.XTTS_V2,
        description="Base XTTS v2 multilingual voice model"
    )
    
    # Character voices
    C3PO = VoiceRefPack(
        name="C3PO",
        repo_url="https://hf.co/Borcherding/XTTS-v2_C3PO",
        creator="Borcherding",
        voice_type=VoiceType.XTTS_V2,
        description="C3PO droid voice from Star Wars"
    )
    
    CARLI_G = VoiceRefPack(
        name="CarliG",
        repo_url="https://hf.co/Borcherding/XTTS-v2_CarliG",
        creator="Borcherding",
        voice_type=VoiceType.XTTS_V2,
        description="CarliG voice model"
    )
    
    # Sports voices
    PETER_DRURY = VoiceRefPack(
        name="PeterDrury",
        repo_url="https://hf.co/kodoqmc/XTTS-v2_PeterDrury",
        creator="kodoqmc",
        voice_type=VoiceType.XTTS_V2,
        description="Peter Drury sports commentator voice"
    )
    
    # Sci-fi voices
    SAN_TI = VoiceRefPack(
        name="San-Ti",
        repo_url="https://hf.co/kodoqmc/XTTS-v2_San-Ti",
        creator="kodoqmc",
        voice_type=VoiceType.XTTS_V2,
        description="San-Ti voice model"
    )
    
    @classmethod
    def get_by_name(cls, name: str):
        """Get voice reference pack by name."""
        for pack in cls:
            if pack.value.name.lower() == name.lower():
                return pack
        return None
    
    @classmethod
    def list_all_names(cls):
        """List all available voice reference pack names."""
        return [pack.value.name for pack in cls]
    
    @classmethod
    def get_repo_url(cls, name: str):
        """Get repository URL for a voice reference pack by name."""
        pack = cls.get_by_name(name)
        return pack.value.repo_url if pack else None