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

import os

# Use the centralized logging instead of creating our own logger
from oarc.utils.log import log
from oarc.speech import TextToSpeech, SpeechToText, SpeechManager
from oarc.utils.paths import Paths

def test_tts():
    """Test TextToSpeech functionality.
    
    Initializes the TTS component with proper paths from the Paths utility,
    processes a sample text input, and generates audio output.
    """
    log.info("Starting TextToSpeech test")
    
    # Initialize Paths utility - correctly uses singleton pattern
    paths = Paths()
    
    # Get TTS paths dictionary
    developer_tools_dict = paths.get_tts_paths_dict()
    
    # Ensure all required directories exist
    paths.ensure_paths(developer_tools_dict)
    
    # Initialize SpeechManager - explicitly specify voice name
    voice_name = "c3po"
    voice_type = "xtts_v2"
    speech_manager = SpeechManager(voice_name=voice_name, voice_type=voice_type)
    
    # Test speech generation with SpeechManager
    test_text = "Hello! I am C-3PO, human-cyborg relations!"
    log.info(f"Processing text with SpeechManager: '{test_text}'")
    
    try:
        audio = speech_manager.generate_speech(test_text)
        if audio is not None:
            log.info(f"Successfully generated audio with SpeechManager")
            
            # For output file creation
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "speech_manager_output.wav")
            
            # Save the audio file
            import soundfile as sf
            sf.write(output_file, audio, speech_manager.sample_rate)
            log.info(f"Saved audio to {output_file}")
    except Exception as e:
        log.error(f"SpeechManager test failed: {e}", exc_info=True)
    
    # Also test using the TextToSpeech class directly (which now uses SpeechManager internally)
    log.info("Initializing TextToSpeech with C-3PO voice")
    tts = TextToSpeech(
        developer_tools_dict=developer_tools_dict,
        voice_type="xtts_v2",
        voice_name=voice_name
    )
    
    # Test speech generation with a sample text
    log.info(f"Processing text with TextToSpeech: '{test_text}'")
    audio = tts.process_tts_responses(test_text, voice_name)
    
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