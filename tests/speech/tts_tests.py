"""
Text-To-Speech and Speech-To-Text Testing Module

This module provides testing functionality for the Text-To-Speech (TTS) 
and Speech-To-Text (STT) components of the OARC package using the async test harness.
It demonstrates how to initialize and use these components with proper path management.
"""

import os
import sys
import soundfile as sf
from typing import Optional

# Add the project root to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from oarc.utils.log import log
from oarc.speech import TextToSpeech, SpeechToText, SpeechManager
from oarc.utils.paths import Paths
from oarc.utils.const import SUCCESS, FAILURE

from tests.async_harness import AsyncTestHarness

# Constants
TEST_VOICE_NAME = "C3PO"  # Derived from HF_VOICE_REF_PACK_C3PO
TEST_VOICE_TYPE = "xtts_v2"
OUTPUT_FILE_NAME = "speech_manager_output.wav"
TEST_TEXT = "Hello! I am C-3PO, human-cyborg relations!"

class TTSAsyncTests(AsyncTestHarness):
    """Async test implementation for TTS and STT functionality."""

    def __init__(self):
        """Initialize the TTS/STT async test harness."""
        super().__init__("TTS/STT")
        self.speech_manager = None
        self.tts = None
        self.audio_output = None
        
    async def setup(self) -> bool:
        """Set up test environment."""
        try:
            log.info(f"Setting up {self.test_name} test environment")
            
            # Initialize Paths utility
            self.paths = Paths()  # singleton instance
            self.paths.log_paths()  # Log paths for debugging
            
            # Get TTS paths dictionary
            self.developer_tools_dict = self.paths.get_tts_paths_dict()
            
            # Ensure all required directories exist
            self.paths.ensure_paths(self.developer_tools_dict)
            
            # Set up the output path
            self.output_path = self.paths.get_test_output_dir()
            self.output_file = os.path.join(self.output_path, OUTPUT_FILE_NAME)
            
            # Get the reference wav file path for the voice
            self.voice_ref_path = self.paths.get_voice_ref_path()
            self.ref_wav = os.path.join(self.voice_ref_path, TEST_VOICE_NAME, "reference.wav")
            
            # Log reference file path
            log.info(f"Using voice reference file: {self.ref_wav}")
            if not os.path.exists(self.ref_wav):
                log.error(f"Reference file does not exist: {self.ref_wav}")
                return False
                
            return True
        except Exception as e:
            log.error(f"Error in test setup: {e}", exc_info=True)
            return False
    
    async def test_direct_tts_generation(self) -> bool:
        """Test TTS speech generation using direct API like tts_fast_tests.py."""
        try:
            # Import directly required models
            log.info("Initializing model directly using documented approach from repo")
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Get device
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Using device: {device}")
            
            # Get the model path
            custom_coqui_dir = self.paths.get_custom_coqui_dir()
            model_dir = os.path.join(custom_coqui_dir, f"XTTS-v2_{TEST_VOICE_NAME}")
            config_path = os.path.join(model_dir, "config.json")
            
            log.info(f"Using model directory: {model_dir}")
            log.info(f"Using config path: {config_path}")
            log.info(f"Using reference path: {self.ref_wav}")
            
            if not os.path.exists(config_path):
                log.error(f"Config file not found at {config_path}")
                return False
            
            # Load the config
            config = XttsConfig()
            config.load_json(config_path)
            
            # Initialize model from config
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
            model.to(device)
            
            # Generate speech
            log.info("Generating speech directly using the model")
            outputs = model.synthesize(
                TEST_TEXT,
                config,
                speaker_wav=self.ref_wav,
                gpt_cond_len=3,
                language="en",
            )
            
            # Save the audio output
            direct_output_file = os.path.join(self.output_path, "direct_tts_output.wav")
            sf.write(direct_output_file, outputs["wav"], 24000)  # XTTS v2 uses 24kHz
            
            log.info(f"Speech generated successfully and saved to {direct_output_file}")
            return True
        except Exception as e:
            log.error(f"Error generating speech directly: {e}", exc_info=True)
            return False
    
    async def test_speech_to_text_init(self) -> bool:
        """Test SpeechToText initialization."""
        try:
            log.info("Testing SpeechToText initialization")
            
            # Initialize the SpeechToText component
            stt = SpeechToText()
            
            # Verify initialization succeeded
            log.info("SpeechToText initialized successfully")
            
            # We don't actually test the hotkey recognition loop as it would
            # block the tests, but we verify that initialization works
            return True
            
        except Exception as e:
            log.error(f"SpeechToText initialization test failed: {e}", exc_info=True)
            return False
    
    async def run_tests(self) -> bool:
        """Run the TTS/STT test suite."""
        try:
            # Test direct TTS generation
            self.results["Direct TTS Generation"] = await self.test_direct_tts_generation()
            
            # Test SpeechToText initialization (without running the loop)
            self.results["SpeechToText Init"] = await self.test_speech_to_text_init()
            
            # Return overall success
            return all(self.results.values())
            
        except Exception as e:
            log.error(f"Error running tests: {e}", exc_info=True)
            return False

# Use the async harness runner
if __name__ == "__main__":
    AsyncTestHarness.run(TTSAsyncTests)
