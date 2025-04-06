"""
Basic test for Coqui TTS functionality.
This test uses the TTS API directly with the OARC path management system.
"""

import os
import logging
import torch
from pathlib import Path

from TTS.api import TTS
from oarc.utils.paths import Paths

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

def test_basic_tts():
    """Test basic TTS functionality using the custom C3PO voice."""
    log.info("Starting basic TTS test with C3PO voice")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_output.wav")
    
    try:
        # Get paths using the OARC utility
        paths = Paths()
        paths.log_paths()  # Log all paths for debugging
        
        # Get paths for voice references
        voice_name = "c3po"
        voice_ref_dir = paths.get_voice_reference_dir()
        voice_ref_file = find_voice_reference_file(voice_ref_dir, voice_name)
        
        if not voice_ref_file:
            log.error(f"No voice reference file found for {voice_name}. Using default model instead.")
            # Fall back to a simple model
            log.info("Initializing default TTS model")
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        else:
            # Use XTTS v2 with the C3PO voice reference
            log.info(f"Using XTTS v2 with voice reference: {voice_ref_file}")
            
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Using device: {device}")
            
            # Initialize TTS with XTTS v2
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=device=="cuda")
        
        # Generate speech
        log.info("Generating speech from test text")
        if voice_ref_file:
            tts.tts_to_file(
                text="Hello! I am C-3PO, human-cyborg relations!",
                file_path=output_file,
                speaker_wav=voice_ref_file,
                language="en"
            )
        else:
            tts.tts_to_file(
                text="Hello! This is a test of the TTS system.",
                file_path=output_file
            )
        
        log.info(f"Speech generated successfully and saved to {output_file}")
        return True
    except Exception as e:
        log.error(f"Error in TTS test: {str(e)}", exc_info=True)
        return False

def find_voice_reference_file(voice_ref_dir, voice_name):
    """Find the voice reference file for the given voice name."""
    voice_dir = os.path.join(voice_ref_dir, voice_name)
    if not os.path.exists(voice_dir):
        log.warning(f"Voice directory not found: {voice_dir}")
        return None
    
    # Check for both clone_speech.wav and reference.wav
    for filename in ["clone_speech.wav", "reference.wav"]:
        ref_file = os.path.join(voice_dir, filename)
        if os.path.exists(ref_file):
            log.info(f"Found voice reference file: {ref_file}")
            return ref_file
    
    log.warning(f"No reference file found in {voice_dir}")
    return None

if __name__ == "__main__":
    test_basic_tts()