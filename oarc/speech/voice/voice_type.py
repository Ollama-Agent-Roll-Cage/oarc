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
    # Updated to match actual model prefixes used in URLs
    XTTS_V2 = "XTTS-v2"
    XTTS_V1 = "XTTS-v1"
    COQUI_TTS = "coqui-tts"
    CUSTOM = "custom-tts"

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