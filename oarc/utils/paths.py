"""
OARC Project Paths Utility

This module defines the Paths class as a singleton, which provides methods to manage essential directories 
for the OARC project. It handles directories for models, HuggingFace caches, Ollama models, and spells 
by ensuring their existence and creating them if necessary. The singleton pattern ensures consistency
across the entire application.
"""

import os
from oarc.utils.log import log

# Global paths dictionary to store all path values
_PATHS = {
    'base': {},
    'models': {},
    'tts': {},
    'env_vars': {}  # Track last seen environment variables
}

# Flag to track if the paths have been initialized
_INITIALIZED = False


class Paths:
    """
    Singleton class for managing paths in the OARC project.
    
    This class provides static methods to access paths, ensuring that only
    one instance of the paths configuration exists across the application.
    """
    
    
    @classmethod
    def initialize(cls):
        """Initialize the paths configuration if not already initialized"""
        global _INITIALIZED, _PATHS
        
        if _INITIALIZED:
            log.debug("Paths already initialized, skipping initialization")
            return
        
        # Get and cache the project root path
        # Fix: Go up one more directory level to get to the actual project root
        file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(os.path.dirname(file_path))  # oarc package dir
        project_root = os.path.dirname(package_dir)  # actual project root
        _PATHS['base']['project_root'] = project_root
        
        # Initialize paths
        cls.refresh_paths()
        
        _INITIALIZED = True
        log.info("Paths singleton initialized")
        

    @classmethod
    def refresh_paths(cls):
        """
        Refresh all paths by checking environment variables and updating paths if needed.
        This allows paths to be updated when environment variables change without restarting.
        """
        global _PATHS
        log.info("Refreshing paths configuration")
        
        # Store current environment variables
        env_vars = {
            'OARC_MODEL_GIT': os.getenv('OARC_MODEL_GIT'),
            'HF_HOME': os.getenv('HF_HOME'),
            'OLLAMA_MODELS': os.getenv('OLLAMA_MODELS')
        }
        
        # Check if environment variables have changed
        if _PATHS['env_vars'] != env_vars:
            log.info("Environment variables changed, updating paths")
            _PATHS['env_vars'] = env_vars
            
            # Update model directory
            model_dir = env_vars['OARC_MODEL_GIT']
            if not model_dir:
                model_dir = os.path.join(_PATHS['base']['project_root'], 'models')
                os.makedirs(model_dir, exist_ok=True)
                log.warning(f"OARC_MODEL_GIT environment variable not set. Using default: {model_dir}")
            _PATHS['base']['model_dir'] = model_dir
            
            # Update HF cache directory
            hf_home = env_vars['HF_HOME']
            if not hf_home:
                hf_home = os.path.join(_PATHS['base']['model_dir'], 'huggingface')
                os.makedirs(hf_home, exist_ok=True)
                log.warning(f"HF_HOME environment variable not set. Using default: {hf_home}")
            _PATHS['models']['hf_cache'] = hf_home
            
            # Update Ollama models directory
            ollama_models = env_vars['OLLAMA_MODELS']
            if not ollama_models:
                ollama_models = os.path.join(_PATHS['base']['model_dir'], 'ollama_models')
                os.makedirs(ollama_models, exist_ok=True)
                log.warning(f"OLLAMA_MODELS environment variable not set. Using default: {ollama_models}")
            _PATHS['models']['ollama_models'] = ollama_models
            
            # Update other paths that derive from the base model directory
            spells_path = os.path.join(_PATHS['base']['model_dir'], 'spells')
            os.makedirs(spells_path, exist_ok=True)
            _PATHS['models']['spells'] = spells_path
            
            # TTS-related paths
            coqui_dir = os.path.join(_PATHS['base']['model_dir'], 'coqui')
            os.makedirs(coqui_dir, exist_ok=True)
            _PATHS['tts']['coqui'] = coqui_dir
            
            whisper_dir = os.path.join(_PATHS['base']['model_dir'], 'whisper')
            os.makedirs(whisper_dir, exist_ok=True)
            _PATHS['tts']['whisper'] = whisper_dir
            
            generated_dir = os.path.join(_PATHS['base']['model_dir'], 'generated')
            os.makedirs(generated_dir, exist_ok=True)
            _PATHS['tts']['generated'] = generated_dir
            
            voice_ref_dir = os.path.join(coqui_dir, 'voice_reference_pack')
            os.makedirs(voice_ref_dir, exist_ok=True)
            _PATHS['tts']['voice_reference'] = voice_ref_dir
            
            # Update working directory values
            _PATHS['base']['current_dir'] = os.getcwd()
            _PATHS['base']['parent_dir'] = os.path.dirname(os.getcwd())
            
            log.info("Path refresh complete")
        else:
            log.debug("No environment variable changes detected")


    @classmethod
    def get_model_dir(cls):
        """Get and validate model directory with fallback"""
        return _PATHS['base']['model_dir']
    

    @classmethod
    def get_hf_cache_dir(cls):
        """Get and validate HuggingFace cache directory with fallback"""
        return _PATHS['models']['hf_cache']
    

    @classmethod
    def get_ollama_models_dir(cls):
        """Get and validate Ollama models directory with fallback"""
        return _PATHS['models']['ollama_models']
    

    @classmethod
    def get_spell_path(cls):
        """Get and validate spell directory with fallback"""
        return _PATHS['models']['spells']
    

    @classmethod
    def get_coqui_dir(cls):
        """Get Coqui TTS models directory"""
        return _PATHS['tts']['coqui']
    

    @classmethod
    def get_whisper_dir(cls):
        """Get Whisper STT models directory"""
        return _PATHS['tts']['whisper']
    

    @classmethod
    def get_generated_dir(cls):
        """Get generated audio output directory"""
        return _PATHS['tts']['generated']
    

    @classmethod
    def get_voice_reference_dir(cls):
        """Get voice reference samples directory"""
        return _PATHS['tts']['voice_reference']
    

    @classmethod
    def get_tts_paths_dict(cls):
        """Get dictionary of TTS-related paths
        
        Returns:
            dict: Dictionary containing all paths needed for TTS functionality
        """
        return {
            'current_dir': _PATHS['base']['current_dir'],
            'parent_dir': _PATHS['base']['parent_dir'],
            'speech_dir': _PATHS['tts']['coqui'],
            'recognize_speech_dir': _PATHS['tts']['whisper'],
            'generate_speech_dir': _PATHS['tts']['generated'],
            'tts_voice_ref_wav_pack_path_dir': _PATHS['tts']['voice_reference']
        }
    

    @classmethod
    def ensure_paths(cls, path_dict):
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
            

    @classmethod
    def dump_paths(cls):
        """Dump all configured paths for debugging
        
        Returns:
            dict: A copy of the current paths dictionary
        """
        log.debug("Dumping current paths configuration")
        # Return a copy to prevent modification of the original
        return {k: v.copy() for k, v in _PATHS.items() if k != 'env_vars'}


    @classmethod
    def log_paths(cls):
        """
        Log all currently configured paths to help with debugging and verification.
        This provides a clear overview of where the system is looking for various resources.
        """
        log.info("========== OARC PATH CONFIGURATION ==========")
        
        # Base paths
        log.info("--- Base Paths ---")
        log.info(f"Project Root: {_PATHS['base'].get('project_root', 'Not set')}")
        log.info(f"Model Directory: {_PATHS['base'].get('model_dir', 'Not set')}")
        log.info(f"Current Directory: {_PATHS['base'].get('current_dir', 'Not set')}")
        log.info(f"Parent Directory: {_PATHS['base'].get('parent_dir', 'Not set')}")
        
        # Model paths
        log.info("--- Model Paths ---")
        log.info(f"HuggingFace Cache: {_PATHS['models'].get('hf_cache', 'Not set')}")
        log.info(f"Ollama Models: {_PATHS['models'].get('ollama_models', 'Not set')}")
        log.info(f"Spells: {_PATHS['models'].get('spells', 'Not set')}")
        
        # TTS paths
        log.info("--- TTS Paths ---")
        log.info(f"Coqui: {_PATHS['tts'].get('coqui', 'Not set')}")
        log.info(f"Whisper: {_PATHS['tts'].get('whisper', 'Not set')}")
        log.info(f"Generated: {_PATHS['tts'].get('generated', 'Not set')}")
        log.info(f"Voice Reference: {_PATHS['tts'].get('voice_reference', 'Not set')}")
        
        # Environment variables status
        log.info("--- Environment Variables ---")
        for var_name, value in _PATHS['env_vars'].items():
            status = "Set" if value else "Not set (using default)"
            log.info(f"{var_name}: {status}")
        
        log.info("==========================================")
    

    # YOLO-related path methods
    @classmethod
    def get_yolo_models_dir(cls):
        """Get directory for YOLO models"""
        yolo_dir = os.path.join(cls.get_model_dir(), "huggingface", "yolo")
        os.makedirs(yolo_dir, exist_ok=True)
        return yolo_dir
    

    @classmethod
    def get_yolo_default_model_path(cls, model_name="yolov8n-obb.pt"):
        """Get path to a specific YOLO model file with default model name"""
        return os.path.join(cls.get_yolo_models_dir(), model_name)


# Initialize paths when the module is imported
Paths.initialize()