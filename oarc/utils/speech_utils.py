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
        If not, create a placeholder file so TTS can initialize.

        Args:
            voice_name (str): Name of the voice reference to ensure
            
        Returns:
            bool: True if the voice reference file exists or was successfully created
        """
        voice_ref_dir = os.path.join(Paths.get_voice_reference_dir(), voice_name)
        voice_ref_file = os.path.join(voice_ref_dir, "clone_speech.wav")
        
        if not os.path.exists(voice_ref_file):
            log.warning(f"Voice reference file not found: {voice_ref_file}")
            
            # Create directory if it doesn't exist
            os.makedirs(voice_ref_dir, exist_ok=True)
            
            # Look for sample files within the project
            project_root = Paths._PATHS['base']['project_root']
            sample_path = os.path.join(project_root, 'samples', 'voices', f'{voice_name}_sample.wav')
            
            if os.path.exists(sample_path):
                # Copy the sample file
                import shutil
                log.info(f"Copying sample voice reference from {sample_path}")
                shutil.copy(sample_path, voice_ref_file)
                log.info(f"Voice reference file created at {voice_ref_file}")
                return True
            else:
                # We need to create a simple dummy wav file
                log.warning("No sample voice file found. Creating a dummy wav file.")
                try:
                    import numpy as np
                    import soundfile as sf
                    
                    # Create a simple sine wave as a dummy audio file
                    sample_rate = 22050
                    duration = 3  # seconds
                    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                    
                    # Save as WAV
                    sf.write(voice_ref_file, audio, sample_rate)
                    log.info(f"Created dummy voice reference file at {voice_ref_file}")
                    return True
                except Exception as e:
                    log.error(f"Failed to create dummy voice file: {e}", exc_info=True)
                    log.error("TTS will likely fail to initialize")
                    return False
        
        return True
