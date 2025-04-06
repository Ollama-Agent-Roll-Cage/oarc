"""
Basic test for Coqui TTS functionality.
This test uses the TTS API directly with the OARC path management system.
"""

import os
import logging
import torch
import json
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
        
        # Get paths for various directories
        voice_name = "c3po"
        voice_ref_dir = paths.get_voice_reference_dir()
        coqui_dir = paths.get_coqui_dir()
        custom_coqui_dir = paths.get_tts_paths_dict()['custom_coqui']
        
        log.info(f"Using regular Coqui directory: {coqui_dir}")
        log.info(f"Using custom Coqui directory: {custom_coqui_dir}")
        
        # First, check if we have a fine-tuned model in either directory
        fine_tuned_model_path = find_fine_tuned_model(coqui_dir, custom_coqui_dir, voice_name)
        
        # Get voice reference file as fallback
        voice_ref_file = find_voice_reference_file(voice_ref_dir, voice_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {device}")
        
        if fine_tuned_model_path:
            # Use fine-tuned model
            log.info(f"Using fine-tuned model from: {fine_tuned_model_path}")
            config_path = os.path.join(fine_tuned_model_path, "config.json")
            
            # Load the model
            tts = TTS(
                model_path=fine_tuned_model_path,
                config_path=config_path,
                progress_bar=False
            )
            tts.to(device)
            
            # Generate speech with fine-tuned model (no need for reference file)
            log.info("Generating speech from fine-tuned model")
            tts.tts_to_file(
                text="Hello! I am C-3PO, human-cyborg relations!",
                file_path=output_file
            )
        elif voice_ref_file:
            # Use XTTS v2 with the C3PO voice reference
            log.info(f"Using XTTS v2 with voice reference: {voice_ref_file}")
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            log.info(f"Initializing XTTS v2 model with reference voice: {model_name}")
            
            # Initialize TTS with XTTS v2
            tts = TTS(model_name=model_name)
            tts.to(device)
            
            # Generate speech with reference voice
            log.info("Generating speech with reference voice")
            tts.tts_to_file(
                text="Hello! I am C-3PO, human-cyborg relations!",
                file_path=output_file,
                speaker_wav=voice_ref_file,
                language="en"
            )
        else:
            # Fall back to a simple model
            log.warning(f"No voice model or reference found for {voice_name}. Using default model instead.")
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            log.info(f"Initializing default TTS model: {model_name}")
            tts = TTS(model_name=model_name)
            
            # Generate speech with default model
            tts.tts_to_file(
                text="Hello! This is a test of the TTS system.",
                file_path=output_file
            )
        
        log.info(f"Speech generated successfully and saved to {output_file}")
        return True
    except Exception as e:
        log.error(f"Error in TTS test: {str(e)}", exc_info=True)
        return False

def find_fine_tuned_model(coqui_dir, custom_coqui_dir, voice_name):
    """Find fine-tuned model in either regular or custom Coqui directories."""
    
    # Check if directories exist
    log.info(f"Checking if coqui_dir exists: {os.path.exists(coqui_dir)}")
    log.info(f"Checking if custom_coqui_dir exists: {os.path.exists(custom_coqui_dir)}")
    
    # Log directory contents for debugging
    if os.path.exists(custom_coqui_dir):
        log.info(f"Contents of custom_coqui_dir: {os.listdir(custom_coqui_dir)}")
    
    # Change the values to match your actual directory structure
    borcherding_dir = os.path.join(os.path.dirname(custom_coqui_dir), "Borcherding")
    if os.path.exists(borcherding_dir):
        log.info(f"Contents of Borcherding directory: {os.listdir(borcherding_dir)}")
    
    # Check various naming patterns in both directories
    possible_paths = [
        os.path.join(coqui_dir, f"XTTS-v2_{voice_name}"),
        os.path.join(coqui_dir, voice_name),
        os.path.join(custom_coqui_dir, f"XTTS-v2_{voice_name}"),
        os.path.join(custom_coqui_dir, voice_name),
        os.path.join(custom_coqui_dir, f"{voice_name}_xtts_v2"),
        
        # Add paths matching your actual directory structure
        os.path.join(os.path.dirname(custom_coqui_dir), "Borcherding", f"XTTS-v2_{voice_name}"),
        os.path.join(os.path.dirname(custom_coqui_dir), "Borcherding-XTTS-v2_C3PO"),
        
        # Try parent directory
        os.path.join(os.path.dirname(custom_coqui_dir), f"XTTS-v2_{voice_name}"),
        
        # Try HuggingFace cache
        os.path.join(os.path.dirname(custom_coqui_dir), "huggingface", "hub", f"models--Borcherding--XTTS-v2_{voice_name}")
    ]
    
    # Log all paths we're checking
    for path in possible_paths:
        log.info(f"Checking path: {path}")
    
    for path in possible_paths:
        if os.path.exists(path):
            config_path = os.path.join(path, "config.json")
            model_path = os.path.join(path, "model.pth")
            
            if os.path.exists(config_path) and os.path.exists(model_path):
                log.info(f"Found fine-tuned model at: {path}")
                
                # Extra validation - check that it's actually an XTTS model
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if 'model' in config and 'model_type' in config['model'] and 'xtts' in config['model']['model_type'].lower():
                        return path
                    else:
                        log.warning(f"Found model at {path} but it's not an XTTS model")
                except:
                    log.warning(f"Found model files at {path} but couldn't validate config")
                    return path  # Still return it and try using it anyway
            else:
                log.info(f"Found directory {path} but missing model files")
    
    log.warning(f"No fine-tuned model found for {voice_name}")
    return None

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