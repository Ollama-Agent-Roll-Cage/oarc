"""
Speech Manager for OARC.

This module provides centralized management for speech-related functionality
including Text-to-Speech initialization, voice reference handling, and audio
resource management. It serves as a central point for configuring and initializing
speech components with appropriate error handling and fallback mechanisms.
"""

import os
import torch
import numpy as np
import logging
import shutil
from TTS.api import TTS
from huggingface_hub import snapshot_download

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.decorators.singleton import singleton
from oarc.utils.setup.tts_utils import accept_coqui_license
from oarc.speech.speech_errors import TTSInitializationError
from oarc.utils.setup.cuda_utils import check_cuda_capable


@singleton
class SpeechManager:
    """
    Centralized manager for speech-related functionality.
    
    This class handles TTS model initialization, voice reference management,
    audio device configuration, and other shared speech processing resources.
    It provides consistent error handling and fallback mechanisms.
    
    Uses the singleton pattern to ensure only one instance is created
    across the application, saving resources and ensuring consistent state.
    """
    
    def __init__(self, voice_name="c3po", voice_type="xtts_v2"):
        """
        Initialize the speech manager with specified voice configuration.
        
        Args:
            voice_name (str): Name of the voice to use (default: "c3po")
            voice_type (str): Type of voice technology to use (default: "xtts_v2")
        """
        self.voice_name = voice_name
        self.voice_type = voice_type
        self.is_multi_speaker = None
        self.paths = Paths()  # Correctly gets the singleton instance via the decorator
        self.sample_rate = 22050
        
        # Configure device using more reliable CUDA detection
        is_cuda_available, cuda_version = check_cuda_capable()
        
        # Verify torch was actually compiled with CUDA if CUDA is detected
        if is_cuda_available:
            try:
                # Test if torch can actually use CUDA (simple tensor creation)
                test_tensor = torch.zeros(1).cuda()
                del test_tensor  # Clean up
                self.device = "cuda"
                log.info(f"Using CUDA {cuda_version} for speech processing.")
            except (AssertionError, RuntimeError) as e:
                # This will catch "Torch not compiled with CUDA enabled" and similar errors
                log.warning(f"CUDA detected but torch cannot use it: {e}")
                log.warning("Using CPU for speech processing instead.")
                is_cuda_available = False
                self.device = "cpu"
        else:
            self.device = "cpu"
            log.info("CUDA not available. Using CPU for speech processing.")
        
        # Set up path dictionary
        self.setup_paths()
        
        # Initialize TTS model
        self.tts = None
        self.voice_reference_path = None
        self.initialize_tts_model()
    
    def setup_paths(self):
        """
        Set up paths for speech-related resources.
        
        Configures all directory paths needed for TTS voices, models, and output files.
        """
        # Get path dictionary from the Paths singleton instance
        self.developer_tools_dict = self.paths.get_tts_paths_dict()
        
        # Extract individual paths for easier access
        self.current_dir = self.developer_tools_dict['current_dir']
        self.parent_dir = self.developer_tools_dict['parent_dir']
        self.speech_dir = self.developer_tools_dict['speech_dir']
        self.recognize_speech_dir = self.developer_tools_dict['recognize_speech_dir']
        self.generate_speech_dir = self.developer_tools_dict['generate_speech_dir']
        self.tts_voice_ref_wav_pack_path = self.developer_tools_dict['tts_voice_ref_wav_pack_path_dir']
        
        # Create any missing directories
        for path in [self.speech_dir, self.recognize_speech_dir, self.generate_speech_dir]:
            os.makedirs(path, exist_ok=True)

    # Voice Pack Management methods
    def download_voice_pack(self, repo_id, voice_name=None):
        """
        Download a voice pack from HuggingFace Hub
        
        Args:
            repo_id (str): HuggingFace repo ID (e.g., "Borcherding/XTTS-v2_C3PO")
                           or a full URL (https://huggingface.co/Borcherding/XTTS-v2_C3PO/tree/main)
            voice_name (str, optional): Name to save the voice pack as. 
                                       If None, will use last part of repo_id.
                                       
        Returns:
            str: Path to downloaded voice pack directory
            bool: Success status
        """
        # Handle full URLs by extracting the repo_id
        if repo_id.startswith("http"):
            from oarc.utils.speech_utils import SpeechUtils
            repo_id = SpeechUtils.extract_repo_id_from_url(repo_id)
        
        if not voice_name:
            voice_name = repo_id.split('/')[-1].lower()
            # Remove XTTS-v2_ prefix if present
            if voice_name.startswith("xtts-v2_"):
                voice_name = voice_name[8:]
            
        log.info(f"Downloading voice pack from {repo_id} as {voice_name}")
        
        # Get the voice reference directory
        voice_reference_dir = self.paths.get_voice_reference_dir()
        
        # Create voice directory
        voice_dir = os.path.join(voice_reference_dir, voice_name)
        os.makedirs(voice_dir, exist_ok=True)
        log.debug(f"Created voice directory at: {voice_dir}")
        
        try:
            # Download the repository files
            log.info(f"Starting HuggingFace Hub download for {repo_id}")
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=voice_dir,
                local_dir_use_symlinks=False,  # Full download, no symlinks
                repo_type="model"  # Explicitly specify this is a model repo
            )
            
            # List the downloaded files to help with debugging
            files = os.listdir(downloaded_path)
            log.info(f"Downloaded voice pack to {downloaded_path} with {len(files)} files: {', '.join(files)}")
            
            # Copy files from downloaded_path to voice_dir if they're different
            if downloaded_path != voice_dir:
                for filename in files:
                    source_file = os.path.join(downloaded_path, filename)
                    target_file = os.path.join(voice_dir, filename)
                    if os.path.isfile(source_file) and not os.path.exists(target_file):
                        shutil.copy2(source_file, target_file)
                log.info(f"Copied files from {downloaded_path} to {voice_dir}")
            
            # Verify the downloaded content
            if self.verify_voice_pack(voice_dir):
                log.info(f"Voice pack {voice_name} successfully downloaded and verified")
                
                # If reference.wav exists but clone_speech.wav doesn't, create a copy as clone_speech.wav
                reference_file = os.path.join(voice_dir, "reference.wav")
                clone_speech_file = os.path.join(voice_dir, "clone_speech.wav")
                if os.path.exists(reference_file) and not os.path.exists(clone_speech_file):
                    shutil.copy2(reference_file, clone_speech_file)
                    log.info(f"Copied reference.wav to clone_speech.wav for compatibility")
                
                return voice_dir, True
            else:
                log.error(f"Downloaded voice pack {voice_name} failed verification")
                return voice_dir, False
                
        except Exception as e:
            log.error(f"Error downloading voice pack from {repo_id}: {e}", exc_info=True)
            return None, False
    
    def verify_voice_pack(self, voice_dir):
        """
        Verify that a voice pack directory contains all required files
        
        Args:
            voice_dir (str): Path to voice pack directory
            
        Returns:
            bool: True if valid, False otherwise
        """
        log.info(f"Verifying voice pack at {voice_dir}")
        
        # Check for reference audio file (either reference.wav or clone_speech.wav)
        has_reference = (
            os.path.isfile(os.path.join(voice_dir, "reference.wav")) or
            os.path.isfile(os.path.join(voice_dir, "clone_speech.wav"))
        )
        
        # Check for required model files
        has_model = os.path.isfile(os.path.join(voice_dir, "model.pth"))
        has_config = os.path.isfile(os.path.join(voice_dir, "config.json"))
        has_vocab = os.path.isfile(os.path.join(voice_dir, "vocab.json"))
        
        # Log the verification results
        if not has_reference:
            log.error(f"Voice pack missing reference audio file (reference.wav or clone_speech.wav)")
        if not has_model:
            log.error(f"Voice pack missing model.pth file")
        if not has_config:
            log.error(f"Voice pack missing config.json file")
        if not has_vocab:
            log.error(f"Voice pack missing vocab.json file")
            
        # Return overall verification result
        is_valid = has_reference and has_model and has_config and has_vocab
        log.info(f"Voice pack verification {'successful' if is_valid else 'failed'}")
        
        return is_valid
    
    def list_available_voices(self):
        """
        List all available voice packs
        
        Returns:
            list: Names of available voice packs
        """
        try:
            voice_reference_dir = self.paths.get_voice_reference_dir()
            voices = [d for d in os.listdir(voice_reference_dir) 
                     if os.path.isdir(os.path.join(voice_reference_dir, d))]
            
            # Filter to only include verified voice packs
            verified_voices = []
            for voice in voices:
                voice_dir = os.path.join(voice_reference_dir, voice)
                if self.verify_voice_pack(voice_dir):
                    verified_voices.append(voice)
                    
            log.info(f"Found {len(verified_voices)} verified voice packs: {', '.join(verified_voices)}")
            return verified_voices
            
        except Exception as e:
            log.error(f"Error listing available voices: {e}")
            return []
            
    def get_voice_path(self, voice_name):
        """
        Get the full path to a voice pack
        
        Args:
            voice_name (str): Name of the voice pack
            
        Returns:
            str: Full path to the voice pack directory or None if not found
        """
        voice_reference_dir = self.paths.get_voice_reference_dir()
        voice_dir = os.path.join(voice_reference_dir, voice_name)
        if os.path.isdir(voice_dir) and self.verify_voice_pack(voice_dir):
            return voice_dir
        return None
    
    def delete_voice_pack(self, voice_name):
        """
        Delete a voice pack
        
        Args:
            voice_name (str): Name of the voice pack to delete
            
        Returns:
            bool: Success status
        """
        voice_reference_dir = self.paths.get_voice_reference_dir()
        voice_dir = os.path.join(voice_reference_dir, voice_name)
        
        if not os.path.isdir(voice_dir):
            log.error(f"Voice pack {voice_name} not found")
            return False
            
        try:
            shutil.rmtree(voice_dir)
            log.info(f"Deleted voice pack {voice_name}")
            return True
        except Exception as e:
            log.error(f"Error deleting voice pack {voice_name}: {e}")
            return False

    def discover_voice_packs(self):
        """
        Discover and verify available voice packs in the voice_reference_pack directory
        
        Returns:
            dict: Dictionary of voice name to voice directory path for verified voice packs
        """
        voice_packs = {}
        
        voice_ref_pack_dir = os.path.join(self.paths.get_coqui_dir(), 'voice_reference_pack')
        if not os.path.exists(voice_ref_pack_dir):
            os.makedirs(voice_ref_pack_dir, exist_ok=True)
            log.info(f"Created voice reference pack directory at {voice_ref_pack_dir}")
            return voice_packs
            
        # Check for subdirectories which might contain voice packs
        try:
            voice_dirs = [d for d in os.listdir(voice_ref_pack_dir) 
                          if os.path.isdir(os.path.join(voice_ref_pack_dir, d))]
            
            if not voice_dirs:
                log.info("No voice packs found in voice_reference_pack directory")
                return voice_packs
                
            log.info(f"Found {len(voice_dirs)} potential voice packs: {', '.join(voice_dirs)}")
            
            # Verify each potential voice pack
            for voice_name in voice_dirs:
                voice_dir = os.path.join(voice_ref_pack_dir, voice_name)
                
                # Check if this is a valid voice pack
                if self.verify_voice_pack(voice_dir):
                    voice_packs[voice_name] = voice_dir
                    log.info(f"Verified voice pack: {voice_name}")
                else:
                    log.warning(f"Directory {voice_name} is not a valid voice pack")
                    
            log.info(f"Found {len(voice_packs)} verified voice packs: {', '.join(voice_packs.keys())}")
            return voice_packs
            
        except Exception as e:
            log.error(f"Error discovering voice packs: {e}", exc_info=True)
            return {}

    def initialize_tts_model(self):
        """
        Initialize the appropriate TTS model.
        
        Loads or creates the TTS model based on the specified voice type and name.
        Handles fallbacks and error scenarios gracefully.
        """
        try:
            log.info(f"=========== INITIALIZING TEXT-TO-SPEECH ===========")
            
            # Get device configuration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Using device: {self.device}")
            
            # Get paths using the Paths singleton
            model_git_dir = self.paths.get_model_dir()
            coqui_dir = self.paths.get_coqui_dir()
            voice_ref_dir = self.paths.get_voice_reference_dir()
            
            log.info(f"Using model directory: {model_git_dir}")
            log.info(f"Using Coqui directory: {coqui_dir}")
            log.info(f"Using voice reference directory: {voice_ref_dir}")
            
            # Check for fine-tuned model first
            fine_tuned_model_path = os.path.join(coqui_dir, f'XTTS-v2_{self.voice_name}')
            
            # List available fine-tuned voices for logging
            available_voices = [d.replace('XTTS-v2_', '') for d in os.listdir(coqui_dir) 
                             if d.startswith('XTTS-v2_') and os.path.isdir(os.path.join(coqui_dir, d))]
            log.info(f"Available fine-tuned voices: {', '.join(available_voices) if available_voices else 'None'}")
            
            # Check for voice reference
            voice_dir = os.path.join(voice_ref_dir, self.voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            
            # Look for reference files using the same pattern as test_tts_fast.py
            self.voice_reference_path = None
            for filename in ["clone_speech.wav", "reference.wav"]:
                ref_path = os.path.join(voice_dir, filename)
                if os.path.exists(ref_path):
                    self.voice_reference_path = ref_path
                    log.info(f"Found voice reference file: {self.voice_reference_path}")
                    break
            
            # If fine-tuned model exists, use it
            if os.path.exists(fine_tuned_model_path):
                config_path = os.path.join(fine_tuned_model_path, "config.json")
                model_path = os.path.join(fine_tuned_model_path, "model.pth")
                
                if not os.path.exists(config_path) or not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing model files in {fine_tuned_model_path}")
                    
                log.info(f"Loading fine-tuned model from: {fine_tuned_model_path}")
                self.tts = TTS(
                    model_path=fine_tuned_model_path,
                    config_path=config_path,
                    progress_bar=False
                )
                self.tts.to(self.device)
                self.is_multi_speaker = False
                
                # Use the model's reference.wav if we don't have one already
                if not self.voice_reference_path:
                    model_ref_path = os.path.join(fine_tuned_model_path, "reference.wav")
                    if os.path.exists(model_ref_path):
                        self.voice_reference_path = model_ref_path
                        log.info(f"Using model's reference.wav: {self.voice_reference_path}")
                
                log.info(f"Loaded fine-tuned model for voice: {self.voice_name}")
                    
            else:
                # Use base model with reference voice
                log.info(f"No fine-tuned model found for {self.voice_name}, using base model with voice reference")
                
                # Use the same model name as in test_tts_fast.py
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
                log.info(f"Initializing XTTS v2 model: {model_name}")
                
                # Initialize with the correct model name
                self.tts = TTS(model_name=model_name)
                self.tts.to(self.device)
                self.is_multi_speaker = True
                
                # Verify we have a voice reference file
                if not self.voice_reference_path:
                    # Try copying from an alternative location if available
                    from oarc.utils.speech_utils import SpeechUtils
                    
                    # Use SpeechUtils to find a reference file
                    alt_ref_path = SpeechUtils.find_voice_reference_file(self.voice_name, self.paths)
                    
                    if alt_ref_path:
                        # Create voice_dir if needed and copy the file
                        os.makedirs(voice_dir, exist_ok=True)
                        clone_speech_path = os.path.join(voice_dir, "clone_speech.wav")
                        import shutil
                        shutil.copy2(alt_ref_path, clone_speech_path)
                        self.voice_reference_path = clone_speech_path
                        log.info(f"Copied voice reference from {alt_ref_path} to {clone_speech_path}")
                    else:
                        # No reference file found anywhere
                        available_refs = os.listdir(voice_ref_dir) if os.path.exists(voice_ref_dir) else 'None'
                        raise FileNotFoundError(
                            f"Voice reference file not found for {self.voice_name}. "
                            f"Please ensure voice references exist in: {voice_ref_dir}\n"
                            f"Available voices: {available_refs}"
                        )
            
            log.info(f"TTS Model initialized successfully on {self.device}")
            log.info(f"Using voice reference: {self.voice_reference_path}")
            log.info(f"=========== TEXT-TO-SPEECH INITIALIZED ===========")
            return True
                
        except Exception as e:
            log.error(f"Error initializing TTS model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error initializing TTS model: {str(e)}")

    def generate_speech(self, text, speed=1.0, language="en"):
        """
        Generate speech audio for the provided text.
        
        Args:
            text (str): The text to convert to speech
            speed (float): Speed factor for speech generation (default: 1.0)
            language (str): Language code for speech (default: "en")
            
        Returns:
            numpy.ndarray: Audio data as a numpy array, or empty array if generation fails
        """
        try:
            if not self.tts:
                log.warning("TTS model not initialized, returning silence")
                return np.array([], dtype=np.float32)
                
            # Clear VRAM cache
            torch.cuda.empty_cache()
            
            # Generate audio
            if self.is_multi_speaker:
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.voice_reference_path,
                    language=language,
                    speed=speed
                )
            else:
                audio = self.tts.tts(
                    text=text,
                    language=language,
                    speed=speed
                )
            
            # Convert to float32 numpy array
            audio_np = np.array(audio, dtype=np.float32)
            
            # Normalize audio
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max()
                
            return audio_np
            
        except Exception as e:
            log.error(f"Error generating speech: {str(e)}", exc_info=True)
            return np.array([], dtype=np.float32)

    def cleanup(self):
        """
        Clean up resources used by the speech manager.
        
        Releases memory and GPU resources to prevent leaks.
        """
        log.info("Cleaning up speech manager resources")
        torch.cuda.empty_cache()
        self.tts = None
        
        # Reset the singleton instance for proper cleanup
        self._reset_singleton()

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="OARC Speech Manager")
    parser.add_argument("--list-voices", action="store_true", help="List all available voice packs")
    args = parser.parse_args()
    
    # Initialize the speech manager
    speech_manager = SpeechManager()
    
    if args.list_voices:
        print("\n=== Available Voice Packs ===")
        voices = speech_manager.list_available_voices()
        if voices:
            for voice in voices:
                print(f"- {voice}")
        else:
            print("No voice packs found")
        
        # Display voice reference path to help with troubleshooting
        from oarc.utils.paths import Paths
        paths = Paths()
        print(f"\nVoice reference directory: {paths.get_voice_reference_dir()}")
        print(f"Expected path structure: {paths.get_voice_reference_dir()}/[voice_name]/clone_speech.wav\n")
    
    else:
        parser.print_help()
        
    sys.exit(0)
