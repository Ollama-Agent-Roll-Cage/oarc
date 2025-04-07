"""
Fast test for Coqui TTS functionality.
This test uses the TTS API directly with the OARC path management system.
"""

import os
import torch
import sys
from pathlib import Path

# Add the project root to the path to make imports work when running directly
# Fix: Use '../../' instead of '..' to go up two levels to the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from oarc.hf.hf_utils import HfUtils
from oarc.speech.voice.voice_utils import VoiceUtils
from oarc.utils.const import HF_VOICE_REF_PACK_C3PO, SUCCESS, FAILURE
from oarc.utils.log import log
from oarc.utils.paths import Paths
from tests.async_harness import AsyncTestHarness

TEST_VOICE_NAME = "C3PO"
TEST_OUTPUT_FILE_NAME = "test_output_C3PO.wav"

class TTSFastTests(AsyncTestHarness):
    """Fast test implementation for TTS functionality."""

    def __init__(self):
        """Initialize the TTS fast test harness."""
        super().__init__("TTS Fast")
        
    async def setup(self) -> bool:
        """Set up test environment."""
        try:
            log.info(f"Setting up {self.test_name} test environment")
            
            # Get paths using the OARC utility singleton
            self.paths = Paths()
            self.paths.log_paths()
            
            # Use the test output directory API
            self.output_dir = self.paths.get_test_output_dir()
            self.output_file = os.path.join(self.output_dir, TEST_OUTPUT_FILE_NAME)
            
            # The voice name we want to use
            self.voice_name = TEST_VOICE_NAME
            
            # Get paths to model files using Paths API
            self.coqui_dir = self.paths.get_coqui_path()
            self.custom_coqui_dir = self.paths.get_custom_coqui_dir()
            self.voice_ref_dir = self.paths.get_voice_ref_path()
            
            # Log paths for informational purposes
            log.info(f"Using Coqui directory: {self.coqui_dir}")
            log.info(f"Using custom Coqui directory: {self.custom_coqui_dir}")
            log.info(f"Using voice reference directory: {self.voice_ref_dir}")
            
            # Determine the specific model and reference paths
            self.model_dir = os.path.join(self.custom_coqui_dir, f"XTTS-v2_{self.voice_name}")
            self.model_path = os.path.join(self.model_dir, "model.pth")
            self.config_path = os.path.join(self.model_dir, "config.json")
            self.reference_path = os.path.join(self.voice_ref_dir, self.voice_name, "reference.wav")
            
            return True
        except Exception as e:
            log.error(f"Error in test setup: {e}", exc_info=True)
            return False

    async def test_model_download(self) -> bool:
        """Test downloading the TTS model if it doesn't exist."""
        try:
            # Check if C3PO model exists, if not, download it
            if not os.path.exists(self.model_path) or not os.path.exists(self.config_path):
                log.info(f"C3PO model not found at {self.model_dir}, downloading it...")
                
                # Download the C3PO model using HfUtils
                downloaded_model_path, success = HfUtils.download_voice_ref_pack(
                    HF_VOICE_REF_PACK_C3PO,  # Use constant from utils.const
                    f"XTTS-v2_{self.voice_name}",
                    target_type="model"  # Download as a full model
                )
                
                if not success:
                    log.error("Failed to download C3PO model")
                    return False
                    
                # Update the model directory path if it's different
                if downloaded_model_path and downloaded_model_path != self.model_dir:
                    self.model_dir = downloaded_model_path
                    self.model_path = os.path.join(self.model_dir, "model.pth")
                    self.config_path = os.path.join(self.model_dir, "config.json")
                    log.info(f"Using downloaded model at: {self.model_dir}")
            else:
                log.info(f"C3PO model already exists at: {self.model_dir}")
                
            return True
        except Exception as e:
            log.error(f"Error downloading model: {e}", exc_info=True)
            return False
            
    async def test_reference_file(self) -> bool:
        """Test ensuring the voice reference file exists."""
        try:
            # Check if reference file exists, if not, create it
            if not os.path.exists(self.reference_path):
                log.info(f"Reference file not found at {self.reference_path}, creating it...")
                
                # Create voice reference directory if it doesn't exist
                os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
                
                # Copy reference file from model directory
                model_reference = None
                for ref_name in ["reference.wav", "clone_speech.wav"]:
                    potential_ref = os.path.join(self.model_dir, ref_name)
                    if os.path.exists(potential_ref):
                        model_reference = potential_ref
                        break
                
                if model_reference:
                    # Use VoiceUtils to copy reference file from model to voice reference directory
                    VoiceUtils.copy_reference_from_model(self.model_dir, self.voice_name, self.paths)
                    log.info(f"Copied reference file to {self.reference_path}")
                else:
                    log.error(f"No reference file found in model directory {self.model_dir}")
                    return False
            else:
                log.info(f"Reference file already exists at: {self.reference_path}")
                
            return True
        except Exception as e:
            log.error(f"Error ensuring reference file: {e}", exc_info=True)
            return False
            
    async def test_verify_files(self) -> bool:
        """Test verifying that all required files exist."""
        try:
            # Verify files exist
            if not os.path.exists(self.model_path):
                log.error(f"Model file not found at {self.model_path}")
                return False
                
            if not os.path.exists(self.config_path):
                log.error(f"Config file not found at {self.config_path}")
                return False
                
            if not os.path.exists(self.reference_path):
                log.error(f"Reference file not found at {self.reference_path}")
                return False
                
            log.info(f"All required files verified: model, config, and reference")
            return True
        except Exception as e:
            log.error(f"Error verifying files: {e}", exc_info=True)
            return False
            
    async def test_tts_generation(self) -> bool:
        """Test TTS speech generation."""
        try:
            # Import directly required models
            log.info("Initializing model directly using documented approach from repo")
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Get device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Using device: {device}")
            
            # Load the config
            config = XttsConfig()
            config.load_json(self.config_path)
            
            # Initialize model from config
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=self.model_dir, eval=True)
            model.to(device)
            
            # Generate speech
            log.info("Generating speech directly using the model")
            message = "Hello! I am C-3PO, human-cyborg relations. I am fluent in over six million forms of communication. How may I assist you today?"
            
            outputs = model.synthesize(
                message,
                config,
                speaker_wav=self.reference_path,
                gpt_cond_len=3,
                language="en",
            )
            
            # Save the audio output
            import soundfile as sf
            sf.write(self.output_file, outputs["wav"], 24000)  # XTTS v2 uses 24kHz
            
            log.info(f"Speech generated successfully and saved to {self.output_file}")
            return True
        except Exception as e:
            log.error(f"Error generating speech: {e}", exc_info=True)
            return False
        
    async def run_tests(self) -> bool:
        """Run the TTS fast test suite."""
        try:
            # Step 1: Download model if needed
            self.results["Model Download"] = await self.test_model_download()
            if not self.results["Model Download"]:
                return False
                
            # Step 2: Ensure reference file exists
            self.results["Reference File"] = await self.test_reference_file()
            if not self.results["Reference File"]:
                return False
                
            # Step 3: Verify all required files exist
            self.results["Verify Files"] = await self.test_verify_files()
            if not self.results["Verify Files"]:
                return False
                
            # Step 4: Generate speech with the model
            self.results["TTS Generation"] = await self.test_tts_generation()
            if not self.results["TTS Generation"]:
                return False
                
            return True
        except Exception as e:
            log.error(f"Error in TTS tests: {e}", exc_info=True)
            return False

# Use the async harness runner
if __name__ == "__main__":
    AsyncTestHarness.run(TTSFastTests)
