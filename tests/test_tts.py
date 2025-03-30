"""
Text-To-Speech and Speech-To-Text Testing Module

This module provides testing functionality for the Text-To-Speech (TTS) 
and Speech-To-Text (STT) components of the OARC package. It demonstrates
how to initialize and use these components with proper path management,
utilizing the centralized Paths utility for consistent environment setup.

Functions:
    test_tts():
        Tests the TextToSpeech component by generating speech from sample text
        
    test_stt():
        Tests the SpeechToText component with hotkey-triggered speech recognition
"""

import logging

from oarc.speech import TextToSpeech, SpeechToText
from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def test_tts():
    """Test TextToSpeech functionality.
    
    Initializes the TTS component with proper paths from the Paths utility,
    processes a sample text input, and generates audio output.
    """
    log.info("Starting TextToSpeech test")
    
    # Initialize Paths utility
    paths = Paths()
    
    # Get TTS paths dictionary
    developer_tools_dict = paths.get_tts_paths_dict()
    
    # Ensure all required directories exist
    paths.ensure_paths(developer_tools_dict)
    
    # Initialize the TextToSpeech component
    log.info("Initializing TextToSpeech with C-3PO voice")
    tts = TextToSpeech(
        developer_tools_dict=developer_tools_dict,
        voice_type="xtts_v2",
        voice_name="c3po"
    )
    
    # Test speech generation with a sample text
    test_text = "Hello! I am C-3PO, human-cyborg relations!"
    log.info(f"Processing text: '{test_text}'")
    audio = tts.process_tts_responses(test_text, "c3po")
    
    if audio is not None:
        log.info(f"Successfully generated audio of length: {len(audio)} samples")
    else:
        log.error("Failed to generate audio")
        
    log.info("TextToSpeech test completed")


def test_stt():
    """Test SpeechToText functionality.
    
    Initializes the SpeechToText component and starts a hotkey recognition loop
    for triggering speech recognition.
    """
    log.info("Starting SpeechToText test")
    
    # Initialize the SpeechToText component
    stt = SpeechToText()
    
    # Inform user about hotkey usage
    log.info("Press Ctrl+Shift to start recording...")
    print("Press Ctrl+Shift to start recording, Ctrl+Alt to finalize, or Shift+Alt to interrupt.")
    
    # Start the hotkey recognition loop
    stt.hotkeyRecognitionLoop()


if __name__ == "__main__":
    try:
        test_tts()
        test_stt()
    except KeyboardInterrupt:
        log.info("Tests interrupted by user")
    except Exception as e:
        log.error(f"Error in tests: {str(e)}", exc_info=True)