"""
Basic test for Coqui TTS functionality.
This test uses the TTS API directly with the OARC path management system.
"""

import os
import torch
import sys

from oarc.hf.hf_utils import HfUtils
from oarc.speech.voice.voice_utils import VoiceUtils
from oarc.utils.const import (
    HF_VOICE_REF_PACK_C3PO, 
    SUCCESS,
    FAILURE
)
from oarc.utils.log import log
from oarc.utils.paths import Paths

TEST_VOICE_NAME = "C3PO"
TEST_OUTPUT_FILE_NAME = "test_output_C3PO.wav"

def test_basic_tts():
    """Test basic TTS functionality using the custom C3PO voice."""
    log.info("Starting basic TTS test with C3PO voice")
    
    # Get paths using the OARC utility singleton
    paths = Paths()
    
    # Use the test output directory API
    output_dir = paths.get_test_output_dir()
    output_file = os.path.join(output_dir, TEST_OUTPUT_FILE_NAME)
    
    try:
        # Get paths using the OARC utility singleton
        paths.log_paths()
        
        # The voice name we want to use
        voice_name = TEST_VOICE_NAME
        
        # Get paths to model files using Paths API
        coqui_dir = paths.get_coqui_path()
        custom_coqui_dir = paths.get_custom_coqui_dir()
        voice_ref_dir = paths.get_voice_ref_path()
        
        # Log paths for informational purposes
        log.info(f"Using Coqui directory: {coqui_dir}")
        log.info(f"Using custom Coqui directory: {custom_coqui_dir}")
        log.info(f"Using voice reference directory: {voice_ref_dir}")
        
        # Determine the specific model and reference paths
        model_dir = os.path.join(custom_coqui_dir, f"XTTS-v2_{voice_name}")
        model_path = os.path.join(model_dir, "model.pth")
        config_path = os.path.join(model_dir, "config.json")
        reference_path = os.path.join(voice_ref_dir, voice_name, "reference.wav")
        
        # Step 1: Check if C3PO model exists, if not, download it
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            log.info(f"C3PO model not found at {model_dir}, downloading it...")
            
            # Download the C3PO model using HfUtils
            downloaded_model_path, success = HfUtils.download_voice_ref_pack(
                HF_VOICE_REF_PACK_C3PO,  # Use constant from utils.const
                f"XTTS-v2_{voice_name}",
                target_type="model"  # Download as a full model
            )
            
            if not success:
                log.error("Failed to download C3PO model")
                return False
                
            # Update the model directory path if it's different
            if downloaded_model_path and downloaded_model_path != model_dir:
                model_dir = downloaded_model_path
                model_path = os.path.join(model_dir, "model.pth")
                config_path = os.path.join(model_dir, "config.json")
                log.info(f"Using downloaded model at: {model_dir}")
        else:
            log.info(f"C3PO model already exists at: {model_dir}")
        
        # Step 2: Check if reference file exists, if not, create it
        if not os.path.exists(reference_path):
            log.info(f"Reference file not found at {reference_path}, creating it...")
            
            # Create voice reference directory if it doesn't exist
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            
            # Copy reference file from model directory
            model_reference = None
            for ref_name in ["reference.wav", "clone_speech.wav"]:
                potential_ref = os.path.join(model_dir, ref_name)
                if os.path.exists(potential_ref):
                    model_reference = potential_ref
                    break
            
            if model_reference:
                # Use VoiceUtils to copy reference file from model to voice reference directory
                VoiceUtils.copy_reference_from_model(model_dir, voice_name, paths)
                log.info(f"Copied reference file to {reference_path}")
            else:
                log.error(f"No reference file found in model directory {model_dir}")
                return False
        else:
            log.info(f"Reference file already exists at: {reference_path}")
        
        log.info(f"Using model directory: {model_dir}")
        log.info(f"Using voice reference file: {reference_path}")
        
        # Verify files exist
        if not os.path.exists(model_path):
            log.error(f"Model file not found at {model_path}")
            return False
            
        if not os.path.exists(config_path):
            log.error(f"Config file not found at {config_path}")
            return False
            
        if not os.path.exists(reference_path):
            log.error(f"Reference file not found at {reference_path}")
            return False
        
        # --------------- Direct method from C3PO repo documentation ---------------
        
        # Import directly required models
        log.info("Initializing model directly using documented approach from repo")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {device}")
        
        # Load the config
        config = XttsConfig()
        config.load_json(config_path)
        
        # Initialize model from config
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        model.to(device)
        
        # Generate speech
        log.info("Generating speech directly using the model")
        message = "Hello! I am C-3PO, human-cyborg relations! Is this Borch? I miss my friend Borch? Have you seen borch, o dear I hope something had not happened to him?"
        
        outputs = model.synthesize(
            message,
            config,
            speaker_wav=reference_path,
            gpt_cond_len=3,
            language="en",
        )
        
        # Save the audio output
        import soundfile as sf
        sf.write(output_file, outputs["wav"], 24000)  # XTTS v2 uses 24kHz
        
        log.info(f"Speech generated successfully and saved to {output_file}")
        return True
    
    except Exception as e:
        log.error(f"Error in TTS test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    result = test_basic_tts()
    sys.exit(SUCCESS if result else FAILURE)