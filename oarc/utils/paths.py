"""
OARC Project Paths Utility

This module defines the Paths class, which provides methods to manage essential directories for the OARC project. 
It handles directories for models, HuggingFace caches, Ollama models, and spells by ensuring their existence and 
creating them if necessary. Logging is integrated to facilitate tracking of directory creation and verification.
"""

import os
import logging


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class Paths():
    

    def __init__(self):
        # Get and cache the project root path
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Initialize and create the model directory right away
        self.model_dir = self.get_model_dir()


    def get_model_dir(self):
        """Get and validate model directory with fallback"""
        model_dir = os.getenv('OARC_MODEL_GIT')
        if not model_dir:
            model_dir = os.path.join(self.project_root, 'models')
            os.makedirs(model_dir, exist_ok=True)
            log.warning(f"OARC_MODEL_GIT environment variable not set. Using default: {model_dir}")
        return model_dir
    
    def get_hf_cache_dir(self):
        """Get and validate HuggingFace cache directory with fallback"""
        hf_home = os.getenv('HF_HOME')
        if not hf_home:
            hf_home = os.path.join(self.get_model_dir(), 'huggingface')
            os.makedirs(hf_home, exist_ok=True)
            log.warning(f"HF_HOME environment variable not set. Using default: {hf_home}")
        return hf_home
    
    def get_ollama_models_dir(self):
        """Get and validate Ollama models directory with fallback"""
        ollama_models = os.getenv('OLLAMA_MODELS')
        if not ollama_models:
            ollama_models = os.path.join(self.get_model_dir(), 'ollama_models')
            os.makedirs(ollama_models, exist_ok=True)
            log.warning(f"OLLAMA_MODELS environment variable not set. Using default: {ollama_models}")
        return ollama_models
    
    def get_spell_path(self):
        """Get and validate spell directory with fallback"""
        spell_path = os.path.join(self.get_model_dir(), 'spells')
        os.makedirs(spell_path, exist_ok=True)
        return spell_path
    
    # New TTS-related path methods
    def get_coqui_dir(self):
        """Get Coqui TTS models directory"""
        coqui_dir = os.path.join(self.get_model_dir(), 'coqui')
        os.makedirs(coqui_dir, exist_ok=True)
        return coqui_dir
    
    def get_whisper_dir(self):
        """Get Whisper STT models directory"""
        whisper_dir = os.path.join(self.get_model_dir(), 'whisper')
        os.makedirs(whisper_dir, exist_ok=True)
        return whisper_dir
    
    def get_generated_dir(self):
        """Get generated audio output directory"""
        generated_dir = os.path.join(self.get_model_dir(), 'generated')
        os.makedirs(generated_dir, exist_ok=True)
        return generated_dir
    
    def get_voice_reference_dir(self):
        """Get voice reference samples directory"""
        voice_ref_dir = os.path.join(self.get_coqui_dir(), 'voice_reference_pack')
        os.makedirs(voice_ref_dir, exist_ok=True)
        return voice_ref_dir
    
    def get_tts_paths_dict(self):
        """Get dictionary of TTS-related paths
        
        Returns:
            dict: Dictionary containing all paths needed for TTS functionality
        """
        return {
            'current_dir': os.getcwd(),
            'parent_dir': os.path.dirname(os.getcwd()),
            'speech_dir': self.get_coqui_dir(),
            'recognize_speech_dir': self.get_whisper_dir(),
            'generate_speech_dir': self.get_generated_dir(),
            'tts_voice_ref_wav_pack_path_dir': self.get_voice_reference_dir()
        }
    
    def ensure_paths(self, path_dict):
        """Ensures all directories in a nested path dictionary exist
        
        Args:
            path_dict: Dictionary containing paths to ensure exist
            
        Returns:
            bool: True if all paths were created/verified successfully
        """
        try:
            log.debug("Ensuring all paths in dictionary exist")
            for key, path in path_dict.items():
                if isinstance(path, str) and (key.endswith('_dir') or key.endswith('_path')):
                    os.makedirs(path, exist_ok=True)
                    log.debug(f"Created/verified directory: {path}")
                
                # Handle nested dictionaries of paths
                elif isinstance(path, dict):
                    for subkey, subpath in path.items():
                        if isinstance(subpath, str):
                            os.makedirs(subpath, exist_ok=True)
                            log.debug(f"Created/verified nested directory: {subpath}")
            
            log.info("All paths created/verified successfully")
            return True
        except Exception as e:
            log.error(f"Error ensuring paths exist: {e}")
            return False