from enum import Enum

"""
OARC Voice Type.

This module defines the `VoiceType` enumeration, which represents the different
types of voice technologies supported by the OARC Speech Manager. It provides
a centralized and consistent way to refer to voice types across the application.
"""



class VoiceType(Enum):
    """
    Enumeration of supported voice types.
    """
    XTTS_V2 = "xtts_v2"
    XTTS_V1 = "xtts_v1"
    COQUI_TTS = "coqui_tts"
    CUSTOM = "custom"

    @staticmethod
    def is_valid(voice_type):
        """
        Check if the given voice type is valid.

        Args:
            voice_type (str): The voice type to validate.

        Returns:
            bool: True if the voice type is valid, False otherwise.
        """
        return voice_type in VoiceType._value2member_map_

    @staticmethod
    def list_types():
        """
        List all available voice types.

        Returns:
            list: A list of all supported voice types as strings.
        """
        return [voice_type.value for voice_type in VoiceType]