"""
Speech utilities for OARC.
This module provides utility functions for working with speech files and audio processing.
"""

import os
from oarc.utils.log import log
from oarc.utils.paths import Paths

class SpeechUtils:
    """Utility functions for speech processing."""

    @staticmethod
    def ensure_voice_reference_file(voice_name):
        """
        Ensure the voice reference file exists for TTS.
        Raises an error if the voice reference file doesn't exist.

        Args:
            voice_name (str): Name of the voice reference to ensure
            
        Returns:
            bool: True if the voice reference file exists
            
        Raises:
            FileNotFoundError: If the voice reference file doesn't exist
        """
        voice_ref_dir = os.path.join(Paths.get_voice_reference_dir(), voice_name)
        voice_ref_file = os.path.join(voice_ref_dir, "clone_speech.wav")
        
        if not os.path.exists(voice_ref_file):
            models_dir = Paths.get_model_dir()
            error_message = (
                f"Voice reference file not found: {voice_ref_file}\n"
                f"Please ensure you have created the voice reference directory at: {voice_ref_dir}\n"
                f"And added a valid 'clone_speech.wav' file in that directory.\n"
                f"For more information, see the documentation on setting up voice references."
            )
            log.error(error_message)
            raise FileNotFoundError(error_message)
        
        log.info(f"Voice reference file found: {voice_ref_file}")
        return True
