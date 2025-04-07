"""
Basic test for Coqui TTS functionality.
This test uses the TTS API directly with the OARC path management system.
"""

import os
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.speech import SpeechManager
from oarc.speech.speech_utils import SpeechUtils

def test_basic_tts():
    """Test basic TTS functionality using the custom C3PO voice."""
    log.info("Starting basic TTS test with C3PO voice")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_output.wav")
    
    try:
        # Get paths using the OARC utility singleton
        paths = Paths()
        paths.log_paths()  # Log all paths for debugging

        # The voice name we want to use
        voice_name = "c3po"
        
        # First ensure the voice reference exists using the centralized utility function
        if not SpeechUtils.ensure_voice_reference_exists(voice_name):
            log.error(f"Cannot continue test: Voice reference for {voice_name} could not be found or downloaded")
            return False
            
        # Now that we know the voice reference exists, initialize the SpeechManager
        log.info(f"Initializing SpeechManager with voice: {voice_name}")
        speech_manager = SpeechManager(voice_name=voice_name)
        
        # Log paths for informational purposes
        coqui_dir = paths.get_coqui_path()
        custom_coqui_dir = paths.get_tts_paths_dict()['custom_coqui']
        voice_ref_dir = paths.get_voice_ref_path()
        
        log.info(f"Using regular Coqui directory: {coqui_dir}")
        log.info(f"Using custom Coqui directory: {custom_coqui_dir}")
        log.info(f"Using voice reference directory: {voice_ref_dir}")
        
        # Log the voice reference path being used
        voice_ref_path = speech_manager.voice_ref_path
        log.info(f"Using voice reference file: {voice_ref_path}")
        
        # Use the generate_speech_to_file method which handles all fallback cases internally
        # Test with overwrite=False to demonstrate non-conflicting filename generation
        log.info("Generating speech using SpeechManager's generate_speech_to_file method")
        success = speech_manager.generate_speech_to_file(
            text="Hello! I am C-3PO, human-cyborg relations!",
            output_file=output_file,
            speed=1.0,
            language="en",
            overwrite=False  # Don't overwrite existing files
        )
        
        if success:
            log.info(f"Speech generated successfully and saved")
            return True
        else:
            log.error("Failed to generate speech")
            return False
            
    except Exception as e:
        log.error(f"Error in TTS test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_basic_tts()