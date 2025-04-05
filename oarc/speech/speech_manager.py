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
        Initialize the appropriate finetuned text to speech with Coqui TTS.
        
        This method configures the TTS engine, checking for fine-tuned models first,
        then falling back to base models with voice reference if needed. It includes
        error handling and device fallback mechanisms.
        
        Returns:
            bool: True if initialization was successful
        
        Raises:
            TTSInitializationError: If the TTS model could not be initialized
        """
        try:
            # Pre-accept the Coqui license and set TTS_HOME to prevent AppData storage
            accept_coqui_license()
            
            # Redirect TTS logs to our logging system before initializing
            for logger_name in ['TTS.utils.manage', 'TTS.tts.models', 'TTS']:
                external_logger = logging.getLogger(logger_name)
                external_logger.handlers.clear()
                external_logger.propagate = True  # Let it propagate to root logger

            log.info(f"=========== INITIALIZING TEXT-TO-SPEECH ===========")
            model_git_dir = self.paths.get_model_dir()
            log.info(f"Using model directory: {model_git_dir}")

            # Construct paths
            coqui_dir = os.path.join(model_git_dir, 'coqui')
            if not os.path.exists(coqui_dir):
                os.makedirs(coqui_dir, exist_ok=True)
                log.warning(f"Coqui directory not found, creating: {coqui_dir}")

            # List available fine-tuned voices
            available_voices = [d.replace('XTTS-v2_', '') for d in os.listdir(coqui_dir) 
                              if d.startswith('XTTS-v2_') and os.path.isdir(os.path.join(coqui_dir, d))]
            log.info(f"Available fine-tuned voices: {', '.join(available_voices) if available_voices else 'None'}")
            
            # Discover voice packs
            available_voice_packs = self.discover_voice_packs()
            if available_voice_packs:
                log.info(f"Available voice packs: {', '.join(available_voice_packs.keys())}")
            
            # Define the voice reference pack directory here so it's available throughout the method
            voice_ref_pack_dir = os.path.join(coqui_dir, 'voice_reference_pack')
            
            fine_tuned_model_path = os.path.join(coqui_dir, f'XTTS-v2_{self.voice_name}')
                
            if os.path.exists(fine_tuned_model_path):
                # Use fine-tuned model
                config_path = os.path.join(fine_tuned_model_path, "config.json")
                model_path = os.path.join(fine_tuned_model_path, "model.pth")
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at {config_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                    
                log.info(f"Loading fine-tuned model from: {fine_tuned_model_path}")
                self.tts = TTS(
                    model_path=fine_tuned_model_path,
                    config_path=config_path,
                    progress_bar=False,
                    gpu=True
                ).to(self.device)
                self.is_multi_speaker = False
                self.voice_reference_path = os.path.join(fine_tuned_model_path, "reference.wav")
                log.info(f"Loaded fine-tuned model for voice: {self.voice_name}")
                    
            else:
                # Use base model with reference voice
                log.info("Fine-tuned model found for c3po. Using c3po voice pack.")
                try:
                    # Explicitly catch torch CUDA errors here instead of relying on fallback
                    if self.device == "cuda":
                        try:
                            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                        except (RuntimeError, AssertionError) as e:
                            if "Torch not compiled with CUDA enabled" in str(e) or "CUDA" in str(e):
                                log.warning(f"CUDA error when loading TTS model: {e}")
                                log.info("Falling back to CPU for TTS model")
                                self.device = "cpu"
                                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                            else:
                                raise
                    else:
                        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                        
                    self.is_multi_speaker = True
                except Exception as e:
                    # If we can't initialize the model at all, raise immediately
                    log.error(f"Failed to initialize TTS model: {e}")
                    raise TTSInitializationError(f"Failed to initialize TTS model: {e}")
                    
                # Look for voice reference in voice_reference_pack
                voice_ref_dir = os.path.join(voice_ref_pack_dir, self.voice_name)
                os.makedirs(voice_ref_dir, exist_ok=True)
                self.voice_reference_path = os.path.join(voice_ref_dir, "clone_speech.wav")
                
                # Check if the requested voice is already in our discovered voice packs
                if self.voice_name in available_voice_packs:
                    voice_dir = available_voice_packs[self.voice_name]
                    log.info(f"Found verified voice pack for {self.voice_name}")
                    
                    # Copy or ensure the voice reference file exists
                    reference_wav = os.path.join(voice_dir, "reference.wav")
                    clone_speech_wav = os.path.join(voice_dir, "clone_speech.wav")
                    
                    if os.path.exists(clone_speech_wav):
                        self.voice_reference_path = clone_speech_wav
                        log.info(f"Using existing clone_speech.wav for {self.voice_name}")
                    elif os.path.exists(reference_wav):
                        # If reference.wav exists but clone_speech.wav doesn't, create a copy
                        if not os.path.exists(clone_speech_wav):
                            shutil.copy2(reference_wav, clone_speech_wav)
                            log.info(f"Copied reference.wav to clone_speech.wav for compatibility")
                        self.voice_reference_path = clone_speech_wav
                    else:
                        log.warning(f"Voice pack for {self.voice_name} exists but is missing reference audio files")
                
                # If we still don't have a valid voice reference file, attempt to download
                if not os.path.exists(self.voice_reference_path):
                    # No valid voice reference found - try to download from known sources
                    from oarc.utils.speech_utils import SpeechUtils
                    
                    # This will be used for automatic downloading when needed
                    voice_repo_urls = {
                        "c3po": "https://huggingface.co/Borcherding/XTTS-v2_C3PO/tree/main"
                    }
                    
                    if self.voice_name in voice_repo_urls:
                        log.info(f"Voice reference file not found. Attempting to download from repository...")
                        if self.download_voice_pack(voice_repo_urls[self.voice_name], self.voice_name):
                            # Check if download created the correct file
                            if os.path.exists(self.voice_reference_path):
                                log.info(f"Successfully downloaded and verified voice reference file: {self.voice_reference_path}")
                            else:
                                # Try alternative path
                                alt_ref_path = os.path.join(voice_ref_dir, "reference.wav")
                                if os.path.exists(alt_ref_path):
                                    # Copy to expected location
                                    import shutil
                                    shutil.copy2(alt_ref_path, self.voice_reference_path)
                                    log.info(f"Copied downloaded reference.wav to clone_speech.wav for compatibility")
                                else:
                                    error_message = (
                                        f"Voice reference file still not found at {self.voice_reference_path} after download\n"
                                        f"Please ensure voice reference exists at: {voice_ref_dir}\n"
                                        f"The file should be named 'clone_speech.wav'."
                                    )
                                    log.error(error_message)
                                    raise FileNotFoundError(error_message)
                        else:
                            error_message = (
                                f"Failed to download voice reference file from repository.\n"
                                f"Voice reference file not found at {self.voice_reference_path}\n"
                                f"Please ensure voice reference exists at: {voice_ref_dir}\n"
                                f"The file should be named 'clone_speech.wav'."
                            )
                            log.error(error_message)
                            raise FileNotFoundError(error_message)
                    else:
                        # No known repository for this voice
                        error_message = (
                            f"Voice reference file not found at {self.voice_reference_path}\n"
                            f"Please ensure voice reference exists at: {voice_ref_dir}\n"
                            f"The file should be named 'clone_speech.wav'.\n"
                            f"No known repository URL for voice '{self.voice_name}' for automatic download."
                        )
                        
                        # Check available voice reference packs
                        if os.path.exists(voice_ref_pack_dir):
                            voices = os.listdir(voice_ref_pack_dir)
                            if voices:
                                error_message += f"\nAvailable voices in reference pack: {voices}"
                        
                        log.error(error_message)
                        raise FileNotFoundError(error_message)
                    
            log.info(f"TTS Model initialized successfully on {self.device}")
            log.info(f"=========== TEXT-TO-SPEECH INITIALIZED ===========")
            return True
                
        except Exception as e:
            log.error(f"Error initializing TTS model: {str(e)}", exc_info=True)
            # Don't attempt CPU fallback here if there's a non-CUDA error, just raise
            raise TTSInitializationError(f"Error initializing TTS model: {str(e)}")

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
