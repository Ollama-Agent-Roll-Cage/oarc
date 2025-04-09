"""
OARC Speech Manager.

This module centralizes the management of speech-related functionality,
including Text-to-Speech (TTS) model initialization, voice reference handling,
and audio resource management. It ensures consistent configuration, robust
error handling, and efficient resource utilization across the application.
"""

import os
import shutil
import numpy as np
import torch
import json
from TTS.api import TTS

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.decorators.singleton import singleton
from oarc.utils.setup.cuda_utils import check_cuda_capable
from oarc.speech.speech_utils import SpeechUtils


@singleton
class SpeechManager:
    """
    Centralized manager for speech-related functionality.

    This class is responsible for managing Text-to-Speech (TTS) models, voice
    references, and audio device configurations. It ensures efficient resource
    utilization, consistent error handling, and seamless integration of speech
    processing capabilities across the application.

    By employing the singleton pattern, this class guarantees that only one
    instance exists throughout the application lifecycle, promoting resource
    efficiency and maintaining a consistent state.
    """
    
    def __init__(self, voice_name="c3po", voice_type="xtts_v2"):
        """
        Initialize the speech manager with the specified voice configuration.

        Args:
            voice_name (str, optional): The name of the voice to use. Defaults to "c3po".
            voice_type (str, optional): The type of voice technology to use. Defaults to "xtts_v2".

        This method sets up the initial configuration for the speech manager, including
        the voice name and type, and prepares the necessary resources for speech processing.
        """
        self.voice_name = voice_name
        self.voice_type = voice_type
        self.is_multi_speaker = None
        self.paths = Paths()  # Correctly gets the singleton instance via the decorator
        self.sample_rate = 22050
        
        # Configure device using more reliable CUDA detection
        is_cuda_available, cuda_version = check_cuda_capable()
        
        # Verify torch was actually compiled with CUDA if CUDA is detected
        if (is_cuda_available):
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
        self.voice_ref_path = None
        self.initialize_tts_model()
    
    def setup_paths(self):
        """
        Set up paths for speech-related resources.

        Ensures all necessary directories for TTS voices, models, and output files
        are properly configured and created if they do not exist.
        """
        # Get path dictionary from the Paths singleton instance
        self.developer_tools_dict = self.paths.get_tts_paths_dict()
        
        # Extract individual paths for easier access - updated names
        self.current_path = self.developer_tools_dict['current_path']
        self.parent_path = self.developer_tools_dict['parent_path']
        self.speech_dir = self.developer_tools_dict['speech_dir']
        self.recognize_speech_dir = self.developer_tools_dict['recognize_speech_dir']
        self.generate_speech_dir = self.developer_tools_dict['generate_speech_dir']
        self.tts_voice_ref_wav_pack_path = self.developer_tools_dict['tts_voice_ref_wav_pack_path_dir']
        
        # Create any missing directories
        for path in [self.speech_dir, self.recognize_speech_dir, self.generate_speech_dir]:
            os.makedirs(path, exist_ok=True)

    # Voice Pack Management methods
    def verify_voice_ref_pack(self, voice_dir):
        """
        Verify the integrity of a voice pack directory.

        This method checks whether the specified directory contains all the
        required files for a valid voice pack.

        Args:
            voice_dir (str): Path to the voice pack directory.

        Returns:
            bool: True if the voice pack is valid (contains all required files),
              False otherwise.
        """
        log.info(f"Verifying voice pack at {voice_dir}")
        
        # Check for reference audio file (either reference.wav or clone_speech.wav)
        has_ref = (
            os.path.isfile(os.path.join(voice_dir, "reference.wav")) or
            os.path.isfile(os.path.join(voice_dir, "clone_speech.wav"))
        )
        
        # Check for required model files
        has_model = os.path.isfile(os.path.join(voice_dir, "model.pth"))
        has_config = os.path.isfile(os.path.join(voice_dir, "config.json"))
        has_vocab = os.path.isfile(os.path.join(voice_dir, "vocab.json"))
        
        # Log the verification results
        if not has_ref:
            log.error(f"Voice pack missing reference audio file (reference.wav or clone_speech.wav)")
        if not has_model:
            log.error(f"Voice pack missing model.pth file")
        if not has_config:
            log.error(f"Voice pack missing config.json file")
        if not has_vocab:
            log.error(f"Voice pack missing vocab.json file")
            
        # Return overall verification result
        is_valid = has_ref and has_model and has_config and has_vocab
        log.info(f"Voice pack verification {'successful' if is_valid else 'failed'}")
        
        return is_valid
    
    def list_ref_voices(self):
        """
        List all available voice packs
        
        Returns:
            list: Names of available voice packs
        """
        try:
            voice_ref_dir = self.paths.get_voice_ref_path()
            voices = [d for d in os.listdir(voice_ref_dir) 
                     if os.path.isdir(os.path.join(voice_ref_dir, d))]
            
            # Filter to only include verified voice packs
            verified_voice_ref_packs = []
            for voice in voices:
                voice_dir = os.path.join(voice_ref_dir, voice)
                if self.verify_voice_ref_pack(voice_dir):
                    verified_voice_ref_packs.append(voice)
                    
            log.info(f"Found {len(verified_voice_ref_packs)} verified voice packs: {', '.join(verified_voice_ref_packs)}")
            return verified_voice_ref_packs
            
        except Exception as e:
            log.error(f"Error listing available voices: {e}")
            return []
            
    def get_voice_ref_pack_path(self, voice_name):
        """
        Get the full path to a voice pack
        
        Args:
            voice_name (str): Name of the voice pack
            
        Returns:
            str: Full path to the voice pack directory or None if not found
        """
        voice_ref_path = self.paths.get_voice_ref_path()
        voice_dir = os.path.join(voice_ref_path, voice_name)
        if os.path.isdir(voice_dir) and self.verify_voice_ref_pack(voice_dir):
            return voice_dir
        return None
    
    def delete_voice_ref_pack(self, voice_name):
        """
        Delete a voice pack
        
        Args:
            voice_name (str): Name of the voice pack to delete
            
        Returns:
            bool: Success status
        """
        voice_ref_dir = self.paths.get_voice_ref_path()
        voice_dir = os.path.join(voice_ref_dir, voice_name)
        
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
        voice_ref_pack = {}
        
        voice_ref_pack_dir = os.path.join(self.paths.get_coqui_path(), 'voice_reference_pack')
        if not os.path.exists(voice_ref_pack_dir):
            os.makedirs(voice_ref_pack_dir, exist_ok=True)
            log.info(f"Created voice reference pack directory at {voice_ref_pack_dir}")
            return voice_ref_pack
            
        # Check for subdirectories which might contain voice packs
        try:
            paths = [d for d in os.listdir(voice_ref_pack_dir) 
                          if os.path.isdir(os.path.join(voice_ref_pack_dir, d))]
            
            if not paths:
                log.info("No voice packs found in voice_reference_pack directory")
                return voice_ref_pack
                
            log.info(f"Found {len(paths)} potential voice packs: {', '.join(paths)}")
            
            # Verify each potential voice pack
            for name in paths:
                path = os.path.join(voice_ref_pack_dir, name)
                
                # Check if this is a valid voice pack
                if self.verify_voice_ref_pack(path):
                    voice_ref_pack[name] = path
                    log.info(f"Verified voice pack: {name}")
                else:
                    log.warning(f"Directory {name} is not a valid voice pack")
                    
            log.info(f"Found {len(voice_ref_pack)} verified voice packs: {', '.join(voice_ref_pack.keys())}")
            return voice_ref_pack
            
        except Exception as e:
            log.error(f"Error discovering voice packs: {e}", exc_info=True)
            return {}

    def initialize_tts_model(self):
        """
        Initialize the appropriate TTS model.
        
        Follows a sequential blocking pattern:
        1. Check if base Coqui XTTS v2 model exists, download if needed
        2. Check if default c3po voice model exists, download if needed
        3. Copy reference.wav from custom model to voice_ref_pack directory
        4. Initialize the appropriate TTS model
        """
        try:
            log.info(f"=========== INITIALIZING TEXT-TO-SPEECH ===========")
            
            # Get device configuration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Using device: {self.device}")
            
            # Get paths using the Paths singleton
            model_git_dir = self.paths.get_model_dir()
            coqui_dir = self.paths.get_coqui_path()
            voice_ref_dir = self.paths.get_voice_ref_path()
            custom_coqui_dir = self.paths.get_tts_paths_dict()['custom_coqui']
            
            log.info(f"Using model directory: {model_git_dir}")
            log.info(f"Using Coqui directory: {coqui_dir}")
            log.info(f"Using custom Coqui directory: {custom_coqui_dir}")
            log.info(f"Using voice reference directory: {voice_ref_dir}")
            
            # STEP 1: Ensure base Coqui XTTS v2 model exists
            base_xtts_dir = os.path.join(coqui_dir, "xtts")
            os.makedirs(base_xtts_dir, exist_ok=True)
            
            log.info("Checking for base Coqui XTTS v2 model")
            if not os.path.exists(os.path.join(base_xtts_dir, "model.pth")):
                log.info("Base Coqui XTTS v2 model not found, downloading...")
                from oarc.hf.hf_utils import HfUtils
                _, model_success = HfUtils.download_voice_ref_pack(
                    "coqui/XTTS-v2",  # Base XTTS v2 repository
                    "xtts",
                    target_type="base_model"  # Use special type for base model
                )
                if not model_success:
                    log.error("Failed to download base XTTS v2 model")
            else:
                log.info("Base Coqui XTTS v2 model already exists")
                
            # STEP 2: Ensure c3po voice model exists
            from oarc.speech.voice.voice_ref_pack import VoiceRefPackType
            from oarc.hf.hf_utils import HfUtils
            
            c3po_model_dir = os.path.join(custom_coqui_dir, "XTTS-v2_C3PO")
            if not os.path.exists(c3po_model_dir) or not os.path.exists(os.path.join(c3po_model_dir, "model.pth")):
                log.info("C3PO voice model not found, downloading...")
                c3po_repo = VoiceRefPackType.C3PO.value.repo_url
                _, c3po_success = HfUtils.download_voice_ref_pack(
                    c3po_repo,
                    "XTTS-v2_C3PO",
                    target_type="model"
                )
                if not c3po_success:
                    log.error("Failed to download C3PO voice model")
            else:
                log.info("C3PO voice model already exists")
            
            # STEP 3: Prepare voice reference directory and find reference file
            voice_dir = os.path.join(voice_ref_dir, self.voice_name)
            os.makedirs(voice_dir, exist_ok=True)
            
            # Look for reference files
            from oarc.speech.voice.voice_utils import VoiceUtils
            self.voice_ref_path = VoiceUtils.find_voice_ref_file(self.voice_name)
            
            # If no reference file found, ensure it exists
            if not self.voice_ref_path:
                log.info(f"No reference file found for {self.voice_name}, attempting to create one")
                from oarc.speech.speech_utils import SpeechUtils
                
                # Try to ensure the voice reference exists (downloads/copies if needed)
                if SpeechUtils.ensure_voice_reference_exists(self.voice_name):
                    # Get the path to the newly created reference file
                    self.voice_ref_path = VoiceUtils.find_voice_ref_file(self.voice_name)
                    
                    if self.voice_ref_path:
                        log.info(f"Successfully created voice reference: {self.voice_ref_path}")
                    else:
                        # This should not happen if ensure_voice_reference_exists returned True
                        log.error(f"Voice reference was created but could not be found")
                
                # If we still don't have a reference file, raise an error
                if not self.voice_ref_path:
                    available_refs = os.listdir(voice_ref_dir) if os.path.exists(voice_ref_dir) else 'None'
                    raise FileNotFoundError(
                        f"Voice reference file not found for {self.voice_name}. "
                        f"Please ensure voice references exist in: {voice_ref_dir}\n"
                        f"Available voices: {available_refs}"
                    )
            
            # STEP 4: Find the most appropriate model to initialize
            fine_tuned_model_path = self.find_fine_tuned_model(
                voice_name=self.voice_name,
                coqui_dir=coqui_dir,
                custom_coqui_dir=custom_coqui_dir,
                verbose=False
            )
            
            # STEP 5: Initialize TTS model
            if fine_tuned_model_path:
                # Initialize with fine-tuned model
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
                    progress_bar=False
                )
                self.tts.to(self.device)
                
                # TEMPORARY FIX: Force model to be treated as single-speaker to avoid 
                # the "Model is multi-speaker but no `speaker` is provided" error
                self.is_multi_speaker = False
                log.info("OVERRIDE: Treating model as single-speaker to avoid speaker errors")
                
                log.info(f"Loaded fine-tuned model for voice: {self.voice_name}")
            else:
                # Initialize with base model and reference voice
                log.info(f"Using base model with voice reference for {self.voice_name}")
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
                log.info(f"Initializing XTTS v2 model: {model_name}")
                self.tts = TTS(model_name=model_name)
                self.tts.to(self.device)
                self.is_multi_speaker = True
            
            log.info("=" * 50)
            log.info(f"TTS Model initialized successfully on {self.device}")
            log.info(f"Using voice reference: {self.voice_ref_path}")
            log.info(f"TEXT-TO-SPEECH INITIALIZED")
            log.info("=" * 50)
            return True
                
        except Exception as e:
            raise RuntimeError(f"Error initializing TTS model: {str(e)}")

    def _check_if_multi_speaker(self, tts_instance):
        """
        Check if a TTS model is multi-speaker by examining its config.
        
        Args:
            tts_instance: The TTS instance to check
            
        Returns:
            bool: True if multi-speaker, False if single-speaker
        """
        try:
            # Try to access the model config dictionary
            config = tts_instance.model_config
            
            # Check for XTTS multi-speaker indicators
            if hasattr(tts_instance, "is_multi_speaker"):
                return tts_instance.is_multi_speaker
                
            # Check for various config keys that indicate multi-speaker
            if "model_args" in config and "num_speakers" in config["model_args"]:
                return config["model_args"]["num_speakers"] > 1
                
            # Check for datasets with multiple speakers
            if "datasets" in config and len(config["datasets"]) > 0:
                # If any dataset has multiple speakers, it's a multi-speaker model
                for dataset in config["datasets"]:
                    if "meta_file_train" in dataset:
                        return True
                        
            # For XTTS specific checks
            if "model" in config and isinstance(config["model"], dict):
                # Some models use speaker_encoder which indicates multi-speaker
                if "use_speaker_encoder" in config["model"] and config["model"]["use_speaker_encoder"]:
                    return True
                
                # For XTTS v2
                if "model_type" in config["model"] and "xtts" in config["model"]["model_type"].lower():
                    # Most XTTS models are multi-speaker by default
                    return True
                    
            # For backward compatibility - if it has voice_file or voice_dir attributes
            if hasattr(tts_instance, "voice_file") or hasattr(tts_instance, "voice_dir"):
                return True
                
            # If we made it here, assume it's not multi-speaker
            return False
            
        except Exception as e:
            log.warning(f"Error checking if model is multi-speaker: {e}")
            # Default to True for safety - we'll pass speaker_wav which non-multi-speaker models will ignore
            return True

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
                log.error("TTS model not initialized")
                return np.array([], dtype=np.float32)
            
            # Clear VRAM cache
            torch.cuda.empty_cache()
            
            log.info(f"Generating speech with {'multi' if self.is_multi_speaker else 'single'}-speaker model")
            
            # Force the model to be treated as multi-speaker and ensure voice_ref_path is available
            if not self.voice_ref_path:
                # Try to find the reference file
                from oarc.speech.voice.voice_utils import VoiceUtils
                self.voice_ref_path = VoiceUtils.find_voice_ref_file(self.voice_name)
                
                if not self.voice_ref_path:
                    log.error("Voice reference file not found, using fallback")
                    # Use a fallback reference file - make sure this exists in your repo
                    fallback_path = os.path.join(self.paths.get_voice_ref_path(), "c3po", "reference.wav")
                    if os.path.exists(fallback_path):
                        self.voice_ref_path = fallback_path
                    else:
                        log.error("Fallback reference file not found")
                        return np.array([], dtype=np.float32)
            
            # Always provide speaker_wav for maximum compatibility
            try:
                audio = self.tts.tts(
                    text=text, 
                    speaker_wav=self.voice_ref_path,
                    language=language,
                    speed=speed
                )
            except Exception as e:
                log.warning(f"Error with speaker reference, trying without: {e}")
                # Try as single speaker mode
                audio = self.tts.tts(
                    text=text,
                    language=language,
                    speed=speed
                )
            
            # Convert to float32 numpy array
            audio_np = np.array(audio, dtype=np.float32)
            
            # Normalize audio
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.9
            
            return audio_np
            
        except Exception as e:
            log.error(f"Error generating speech: {str(e)}")
            return np.array([], dtype=np.float32)  # Return empty array on error

    def generate_speech_to_file(self, text, output_file, speed=1.0, language="en", force_fallback=False, overwrite=False):
        """
        Generate speech and save directly to a file, with automatic fallback to alternative models if needed.
        
        This method attempts to generate speech using the currently initialized TTS model,
        then falls back to alternative models if needed, writing the result directly to a file.
        
        Args:
            text (str): The text to convert to speech
            output_file (str): Path to save the output audio file
            speed (float): Speed factor for speech generation (default: 1.0)
            language (str): Language code for speech (default: "en")
            force_fallback (bool): Force using the fallback chain even if a model is initialized
            overwrite (bool): Whether to overwrite existing files (default: False)
            
        Returns:
            bool: True if speech was successfully generated and saved, False otherwise
        """
        # If we don't want to overwrite, generate a non-conflicting filename
        if not overwrite and os.path.exists(output_file):
            output_file = SpeechUtils.get_non_conflicting_filename(output_file)

        # If we have an initialized model and aren't forcing fallback, use it
        if self.tts is not None and not force_fallback:
            try:
                log.info(f"Generating speech using initialized TTS model")
                audio = self.generate_speech(text, speed, language)
                
                # Save the generated audio to a file
                if audio is not None and len(audio) > 0:
                    import soundfile as sf
                    sf.write(output_file, audio, self.sample_rate)
                    log.info(f"Speech generated and saved to {output_file}")
                    return True
                else:
                    log.warning("Generated audio was empty, falling back to alternative models")
            except Exception as e:
                log.warning(f"Error using initialized TTS model: {e}", exc_info=True)
                log.warning("Falling back to alternative models")
        else:
            log.info("No initialized TTS model, using fallback models")
        
        # Get paths for fallback logic
        paths = Paths()
        coqui_dir = paths.get_coqui_path()
        custom_coqui_dir = paths.get_tts_paths_dict()['custom_coqui']
        
        # Step 1: Try to find a fine-tuned model
        fine_tuned_model_path = self.find_fine_tuned_model(self.voice_name, coqui_dir, custom_coqui_dir, verbose=False)
        
        # Step 2: Get voice reference file using SpeechUtils
        if not self.voice_ref_path:
            self.voice_ref_path = SpeechUtils.find_voice_ref_pack_file(self.voice_name)
        
        # Determine device to use
        device = self.device  # Use already determined device
        
        try:
            # Step 3: Attempt to generate speech with the appropriate model
            if fine_tuned_model_path:
                # Use fine-tuned model
                log.info(f"Using fine-tuned model from: {fine_tuned_model_path}")
                config_path = os.path.join(fine_tuned_model_path, "config.json")
                
                # Load the model
                fallback_tts = TTS(
                    model_path=fine_tuned_model_path,
                    config_path=config_path,
                    progress_bar=False
                )
                fallback_tts.to(device)
                
                # Check if this model is multi-speaker
                is_fallback_multi_speaker = self._check_if_multi_speaker(fallback_tts)
                
                # Generate different arguments based on whether it's multi-speaker
                tts_args = {
                    "text": text,
                    "file_path": output_file,
                    "language": language,
                    "speed": speed
                }
                
                # Add speaker_wav parameter if this is a multi-speaker model
                if is_fallback_multi_speaker and self.voice_ref_path:
                    tts_args["speaker_wav"] = self.voice_ref_path
                    log.info(f"Using speaker_wav with fallback model: {self.voice_ref_path}")
                
                # Since TTS API doesn't have an overwrite parameter, we need to handle it specially
                if os.path.exists(output_file) and not overwrite:
                    temp_file = output_file + ".temp"
                    log.info(f"Using temporary file: {temp_file}")
                    
                    # Use a temporary file name for the tts_to_file call
                    tts_args["file_path"] = temp_file
                    
                    # Generate speech with model to temp file
                    fallback_tts.tts_to_file(**tts_args)
                    
                    # Copy the temp file to the non-conflicting filename
                    shutil.copy2(temp_file, output_file)
                    
                    # Remove the temp file
                    os.remove(temp_file)
                    log.info(f"Renamed temporary file to: {output_file}")
                else:
                    # Generate speech directly to output file
                    fallback_tts.tts_to_file(**tts_args)
            
            elif self.voice_ref_path:
                # Use XTTS v2 with voice reference
                log.info(f"Using XTTS v2 with voice reference: {self.voice_ref_path}")
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
                log.info(f"Initializing XTTS v2 model with reference voice: {model_name}")
                
                # Initialize TTS with XTTS v2
                fallback_tts = TTS(model_name=model_name)
                fallback_tts.to(device)
                
                # Handle overwrite parameter same as above
                if os.path.exists(output_file) and not overwrite:
                    temp_file = output_file + ".temp"
                    
                    # Generate speech with reference voice to temp file
                    fallback_tts.tts_to_file(
                        text=text,
                        file_path=temp_file,
                        speaker_wav=self.voice_ref_path,
                        language=language,
                        speed=speed
                    )
                    
                    # Copy the temp file to the non-conflicting filename
                    shutil.copy2(temp_file, output_file)
                    
                    # Remove the temp file
                    os.remove(temp_file)
                else:
                    # Generate speech with reference voice directly
                    fallback_tts.tts_to_file(
                        text=text,
                        file_path=output_file,
                        speaker_wav=self.voice_ref_path,
                        language=language,
                        speed=speed
                    )
            else:
                # Fall back to a simple model
                log.warning(f"No voice model or reference found for {self.voice_name}. Using default model instead.")
                model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                log.info(f"Initializing default TTS model: {model_name}")
                fallback_tts = TTS(model_name=model_name)
                
                # Handle overwrite parameter
                if os.path.exists(output_file) and not overwrite:
                    temp_file = output_file + ".temp"
                    
                    # Generate speech with default model to temp file
                    fallback_tts.tts_to_file(
                        text=text,
                        file_path=temp_file
                    )
                    
                    # Copy the temp file to the non-conflicting filename
                    shutil.copy2(temp_file, output_file)
                    
                    # Remove the temp file
                    os.remove(temp_file)
                else:
                    # Generate speech with default model directly
                    fallback_tts.tts_to_file(
                        text=text,
                        file_path=output_file
                    )
            
            log.info(f"Speech generated successfully and saved to {output_file}")
            return True
            
        except Exception as e:
            log.error(f"Error in fallback TTS generation: {e}", exc_info=True)
            return False

    def find_fine_tuned_model(self, voice_name, coqui_dir=None, custom_coqui_dir=None, verbose=True):
        """
        Find a fine-tuned model in available directories with configurable logging detail.
        
        This method searches for fine-tuned voice models in standard and custom
        directories, with thorough logging for debugging purposes. It checks various
        naming patterns and validates found models.
        
        Args:
            voice_name (str): Name of the voice to find model for
            coqui_dir (str, optional): Path to the main Coqui directory. 
                                       If None, uses the path from Paths singleton.
            custom_coqui_dir (str, optional): Path to custom Coqui directory.
                                             If None, uses the path from Paths singleton.
            verbose (bool, optional): Whether to log with INFO level (True) or DEBUG level (False).
                                     Defaults to True.
            
        Returns:
            str: Path to fine-tuned model or None if not found
        """
        # Get paths if not provided
        if coqui_dir is None or custom_coqui_dir is None:
            paths = Paths()
            if coqui_dir is None:
                coqui_dir = paths.get_coqui_path()
            if custom_coqui_dir is None:
                custom_coqui_dir = paths.get_tts_paths_dict()['custom_coqui']
        
        # Configure logging level based on verbose flag
        import logging
        original_level = log._base_logger.level
        if not verbose:
            log._base_logger.setLevel(logging.DEBUG)
        
        try:
            # Check if directories exist with appropriate logging
            log_fn = log.info if verbose else log.debug
            log_fn(f"Checking if coqui_dir exists: {os.path.exists(coqui_dir)}")
            log_fn(f"Checking if custom_coqui_dir exists: {os.path.exists(custom_coqui_dir)}")
            
            # Log directory contents for debugging
            if os.path.exists(custom_coqui_dir):
                log_fn(f"Contents of custom_coqui_dir: {os.listdir(custom_coqui_dir)}")
            
            # Check for Borcherding directory which might contain models
            borcherding_dir = os.path.join(os.path.dirname(custom_coqui_dir), "Borcherding")
            if os.path.exists(borcherding_dir):
                log_fn(f"Contents of Borcherding directory: {os.listdir(borcherding_dir)}")
            
            # Check various naming patterns in both directories
            possible_paths = [
                os.path.join(coqui_dir, f"XTTS-v2_{voice_name}"),
                os.path.join(coqui_dir, voice_name),
                os.path.join(custom_coqui_dir, f"XTTS-v2_{voice_name}"),
                os.path.join(custom_coqui_dir, voice_name),
                os.path.join(custom_coqui_dir, f"{voice_name}_xtts_v2"),
                
                # Check potential old locations for backward compatibility
                os.path.join(os.path.dirname(coqui_dir), "custom_xtts_v2", f"XTTS-v2_{voice_name}"),
                os.path.join(os.path.dirname(coqui_dir), "custom_xtts_v2", voice_name),
                
                # Add paths matching actual directory structures seen in the wild
                os.path.join(os.path.dirname(custom_coqui_dir), "Borcherding", f"XTTS-v2_{voice_name}"),
                os.path.join(os.path.dirname(custom_coqui_dir), "Borcherding-XTTS-v2_C3PO"),
                
                # Try parent directory
                os.path.join(os.path.dirname(custom_coqui_dir), f"XTTS-v2_{voice_name}"),
                
                # Try HuggingFace cache
                os.path.join(os.path.dirname(coqui_dir), "huggingface", "hub", f"models--Borcherding--XTTS-v2_{voice_name}")
            ]
            
            # Log all paths we're checking
            for path in possible_paths:
                log_fn(f"Checking path: {path}")
            
            # Check each path for model files
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = os.path.join(path, "config.json")
                    model_path = os.path.join(path, "model.pth")
                    
                    if os.path.exists(config_path) and os.path.exists(model_path):
                        log_fn(f"Found fine-tuned model at: {path}")
                        
                        # Extra validation - check that it's actually an XTTS model
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            if 'model' in config and 'model_type' in config['model'] and 'xtts' in config['model']['model_type'].lower():
                                return path
                            else:
                                # Use the diagnostic function for more detailed logging
                                from oarc.speech.speech_utils import SpeechUtils
                                diagnostics = SpeechUtils.diagnose_model_config(path)
                                log.warning(f"Found model at {path} but it's not an XTTS model")
                                log.debug(f"Model validation diagnostics: {diagnostics}")
                                
                                # Even if validation fails, let's use the model anyway if it has the basics
                                log.info(f"Attempting to use model despite validation failure: {path}")
                                return path
                        except Exception as e:
                            log.warning(f"Found model files at {path} but couldn't validate config: {e}", exc_info=True)
                            # Still return it and try using it anyway
                            return path
                    else:
                        log_fn(f"Found directory {path} but missing model files")
            
            log.warning(f"No fine-tuned model found for {voice_name}")
            return None
            
        finally:
            # Always restore original logging level if changed
            if not verbose:
                log._base_logger.setLevel(original_level)

    def cleanup(self):
        """
        Clean up resources used by the speech manager.
        
        Releases memory and GPU resources to prevent leaks.
        """
        log.info("Cleaning up speech manager resources")
        torch.cuda.empty_cache()
        self.tts = None
        
        # Reset the singleton instance
        self._reset_singleton()

# TODO this should be moved into its own script file
if __name__ == "__main__":
    """Command-line interface for the OARC Speech Manager."""

    import sys
    import argparse

    from oarc.utils.const import SUCCESS, FAILURE

    parser = argparse.ArgumentParser(description="OARC Speech Manager")
    parser.add_argument("--list-voices", action="store_true", help="List all available voice packs")
    args = parser.parse_args()
    
    # Initialize the SpeechManager singleton
    manager = SpeechManager()
    
    if args.list_voices:
        print("=" * 50)
        print("Available Voice Packs")
        voices = manager.list_ref_voices()
        if voices:
            for voice in voices:
                print(f"- {voice}")
                print("=" * 50)
        else: 
            # No voice packs found
            from oarc.utils.paths import Paths
            paths = Paths()
            print(f"- No voice packs found. Please check your setup.")
            print(f"- Voice reference pack path: {paths.get_voice_ref_path()}")
            print(f"- Expected path structure for voice pack: {paths.get_voice_ref_path()}/[voice_name]/clone_speech.wav\n")
            print("=" * 50)
            sys.exit(FAILURE)
        
    else:
        parser.print_help()
        
    sys.exit(SUCCESS)
