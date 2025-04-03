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
from TTS.api import TTS

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.decorators.singleton import singleton
from oarc.utils.setup.tts_utils import accept_coqui_license


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
        
        # Configure device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
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

    def initialize_tts_model(self):
        """
        Initialize the appropriate finetuned text to speech with Coqui TTS.
        
        This method configures the TTS engine, checking for fine-tuned models first,
        then falling back to base models with voice reference if needed. It includes
        error handling and device fallback mechanisms.
        
        Returns:
            bool: True if initialization was successful
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
            model_git_dir = self.paths.get_model_dir()  # Use the singleton instance
            log.info(f"Using model directory: {model_git_dir}")

            # Construct paths
            coqui_dir = os.path.join(model_git_dir, 'coqui')
            if not os.path.exists(coqui_dir):
                os.makedirs(coqui_dir, exist_ok=True)
                log.warning(f"Coqui directory not found, creating: {coqui_dir}")

            # List available voices
            available_voices = [d.replace('XTTS-v2_', '') for d in os.listdir(coqui_dir) 
                              if d.startswith('XTTS-v2_') and os.path.isdir(os.path.join(coqui_dir, d))]
            log.info(f"Available voices: {', '.join(available_voices) if available_voices else 'None'}")
            
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
                log.info(f"No fine-tuned model found for {self.voice_name}, using base model with voice reference")
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                self.is_multi_speaker = True
                    
                # Look for voice reference in voice_reference_pack
                voice_ref_dir = os.path.join(coqui_dir, 'voice_reference_pack', self.voice_name)
                os.makedirs(voice_ref_dir, exist_ok=True)
                self.voice_reference_path = os.path.join(voice_ref_dir, "clone_speech.wav")
                    
                if not os.path.exists(self.voice_reference_path):
                    raise FileNotFoundError(
                        f"Voice reference file not found at {self.voice_reference_path}\n"
                        f"Please ensure voice reference exists at: {voice_ref_dir}\n"
                        f"Available voices in reference pack: {os.listdir(os.path.join(coqui_dir, 'voice_reference_pack'))}"
                    )
                    
            log.info(f"TTS Model initialized successfully on {self.device}")
            log.info(f"=========== TEXT-TO-SPEECH INITIALIZED ===========")
            return True
                
        except Exception as e:
            raise RuntimeError(f"Error initializing TTS model: {str(e)}", exc_info=True)

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
