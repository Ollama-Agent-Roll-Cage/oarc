"""
OARC Project Paths Utility

This module defines the Paths class as a singleton, which provides methods to manage essential directories 
for the OARC project. It handles directories for models, HuggingFace caches, Ollama models, and spells 
by ensuring their existence and creating them if necessary. The singleton pattern ensures consistency
across the entire application.
"""

import os
from oarc.utils.log import log
from oarc.utils.decorators.singleton import singleton
from oarc.utils.const import (
    DEFAULT_MODELS_DIR, 
    HUGGINGFACE_DIR,
    OLLAMA_MODELS_DIR, 
    SPELLS_DIR,
    COQUI_DIR,
    CUSTOM_COQUI_DIR, 
    WHISPER_DIR, 
    GENERATED_DIR, 
    VOICE_REFERENCE_DIR,
    YOLO_DIR
)

# Define a constant for the ignored agents directory
IGNORED_AGENTS_DIR = "IgnoredAgents"

@singleton
class Paths:
    """
    Singleton class for managing paths in the OARC project.
    
    This class provides methods to access paths, ensuring that only
    one instance of the paths configuration exists across the application.
    """
    
    def __init__(self):
        """Initialize the paths configuration"""
        self._paths = {
            'base': {},
            'models': {},
            'tts': {},
            'env_vars': {}  # Track last seen environment variables
        }
        
        # Get and cache the project root path
        file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(os.path.dirname(file_path))  # oarc package dir
        project_root = os.path.dirname(package_dir)  # actual project root
        self._paths['base']['project_root'] = project_root
        
        # Initialize paths
        self.refresh_paths()
        
        log.info("Paths singleton initialized")

    def refresh_paths(self):
        """
        Refresh all paths by checking environment variables and updating paths if needed.
        This allows paths to be updated when environment variables change without restarting.
        """
        log.info("Refreshing paths configuration")
        
        # Store current environment variables
        env_vars = {
            'OARC_MODEL_GIT': os.getenv('OARC_MODEL_GIT'),
            'HF_HOME': os.getenv('HF_HOME'),
            'OLLAMA_MODELS': os.getenv('OLLAMA_MODELS')
        }
        
        # Check if environment variables have changed
        if self._paths['env_vars'] != env_vars:
            log.info("Environment variables changed, updating paths")
            self._paths['env_vars'] = env_vars
            
            # Update model directory
            model_dir = env_vars['OARC_MODEL_GIT']
            if not model_dir:
                model_dir = os.path.join(self._paths['base']['project_root'], DEFAULT_MODELS_DIR)
                os.makedirs(model_dir, exist_ok=True)
                log.warning(f"OARC_MODEL_GIT environment variable not set. Using default: {model_dir}")
            self._paths['base']['model_dir'] = model_dir
            
            # Update HF cache directory
            hf_home = env_vars['HF_HOME']
            if not hf_home:
                hf_home = os.path.join(self._paths['base']['model_dir'], HUGGINGFACE_DIR)
                os.makedirs(hf_home, exist_ok=True)
                log.warning(f"HF_HOME environment variable not set. Using default: {hf_home}")
            self._paths['models']['hf_cache'] = hf_home
            
            # Update Ollama models directory
            ollama_models = env_vars['OLLAMA_MODELS']
            if not ollama_models:
                ollama_models = os.path.join(self._paths['base']['model_dir'], OLLAMA_MODELS_DIR)
                os.makedirs(ollama_models, exist_ok=True)
                log.warning(f"OLLAMA_MODELS environment variable not set. Using default: {ollama_models}")
            self._paths['models']['ollama_models'] = ollama_models
            
            # Update other paths that derive from the base model directory
            spells_path = os.path.join(self._paths['base']['model_dir'], SPELLS_DIR)
            os.makedirs(spells_path, exist_ok=True)
            self._paths['models']['spells'] = spells_path
            
            # TTS-related paths
            coqui_dir = os.path.join(self._paths['base']['model_dir'], COQUI_DIR)
            os.makedirs(coqui_dir, exist_ok=True)
            self._paths['tts']['coqui'] = coqui_dir
            
            custom_coqui_dir = os.path.join(self._paths['base']['model_dir'], CUSTOM_COQUI_DIR)
            os.makedirs(custom_coqui_dir, exist_ok=True)
            self._paths['tts']['custom_coqui'] = custom_coqui_dir
            
            whisper_dir = os.path.join(self._paths['base']['model_dir'], WHISPER_DIR)
            os.makedirs(whisper_dir, exist_ok=True)
            self._paths['tts']['whisper'] = whisper_dir
            
            generated_dir = os.path.join(self._paths['base']['model_dir'], GENERATED_DIR)
            os.makedirs(generated_dir, exist_ok=True)
            self._paths['tts']['generated'] = generated_dir
            
            voice_ref_dir = os.path.join(coqui_dir, VOICE_REFERENCE_DIR)
            os.makedirs(voice_ref_dir, exist_ok=True)
            self._paths['tts']['voice_reference'] = voice_ref_dir
            
            # Update working directory values
            self._paths['base']['current_dir'] = os.getcwd()
            self._paths['base']['parent_dir'] = os.path.dirname(os.getcwd())
            
            log.info("Path refresh complete")
        else:
            log.debug("No environment variable changes detected")

    # Convert all class methods to instance methods
    def get_model_dir(self):
        """Get and validate model directory with fallback"""
        return self._paths['base']['model_dir']
    
    def get_hf_cache_dir(self):
        """Get and validate HuggingFace cache directory with fallback"""
        return self._paths['models']['hf_cache']
    
    def get_ollama_models_dir(self):
        """Get and validate Ollama models directory with fallback"""
        return self._paths['models']['ollama_models']
    
    def get_spell_path(self):
        """Get and validate spell directory with fallback"""
        return self._paths['models']['spells']
    
    def get_coqui_dir(self):
        """Get Coqui TTS models directory"""
        return self._paths['tts']['coqui']
    
    def get_whisper_dir(self):
        """Get Whisper STT models directory"""
        return self._paths['tts']['whisper']
    
    def get_generated_dir(self):
        """Get generated audio output directory"""
        return self._paths['tts']['generated']
    
    def get_voice_reference_dir(self):
        """Get voice reference samples directory"""
        return self._paths['tts']['voice_reference']
    
    def get_tts_paths_dict(self):
        """Get dictionary of TTS-related paths
        
        Returns:
            dict: Dictionary containing all paths needed for TTS functionality
        """
        return {
            'current_dir': self._paths['base']['current_dir'],
            'parent_dir': self._paths['base']['parent_dir'],
            'speech_dir': self._paths['tts']['coqui'],
            'recognize_speech_dir': self._paths['tts']['whisper'],
            'generate_speech_dir': self._paths['tts']['generated'],
            'tts_voice_ref_wav_pack_path_dir': self._paths['tts']['voice_reference'],
            'custom_coqui': self._paths['tts']['custom_coqui']
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
            
    def dump_paths(self):
        """Dump all configured paths for debugging
        
        Returns:
            dict: A copy of the current paths dictionary
        """
        log.debug("Dumping current paths configuration")
        # Return a copy to prevent modification of the original
        return {k: v.copy() for k, v in self._paths.items() if k != 'env_vars'}

    def log_paths(self):
        """
        Log all currently configured paths to help with debugging and verification.
        This provides a clear overview of where the system is looking for various resources.
        """
        log.info("========== OARC PATH CONFIGURATION ==========")
        
        # Base paths
        log.info("--- Base Paths ---")
        log.info(f"Project Root: {self._paths['base'].get('project_root', 'Not set')}")
        log.info(f"Model Directory: {self._paths['base'].get('model_dir', 'Not set')}")
        log.info(f"Current Directory: {self._paths['base'].get('current_dir', 'Not set')}")
        log.info(f"Parent Directory: {self._paths['base'].get('parent_dir', 'Not set')}")
        
        # Model paths
        log.info("--- Model Paths ---")
        log.info(f"HuggingFace Cache: {self._paths['models'].get('hf_cache', 'Not set')}")
        log.info(f"Ollama Models: {self._paths['models'].get('ollama_models', 'Not set')}")
        log.info(f"Spells: {self._paths['models'].get('spells', 'Not set')}")
        
        # TTS paths
        log.info("--- TTS Paths ---")
        log.info(f"Coqui: {self._paths['tts'].get('coqui', 'Not set')}")
        log.info(f"Whisper: {self._paths['tts'].get('whisper', 'Not set')}")
        log.info(f"Generated: {self._paths['tts'].get('generated', 'Not set')}")
        log.info(f"Voice Reference: {self._paths['tts'].get('voice_reference', 'Not set')}")
        
        # Environment variables status
        log.info("--- Environment Variables ---")
        for var_name, value in self._paths['env_vars'].items():
            status = "Set" if value else "Not set (using default)"
            log.info(f"{var_name}: {status}")
        
        log.info("==========================================")
    
    # YOLO-related path methods
    def get_yolo_models_dir(self):
        """Get directory for YOLO models"""
        yolo_dir = os.path.join(self.get_model_dir(), HUGGINGFACE_DIR, YOLO_DIR)
        os.makedirs(yolo_dir, exist_ok=True)
        return yolo_dir
    
    def get_yolo_default_model_path(self, model_name="yolov8n-obb.pt"):
        """Get path to a specific YOLO model file with default model name"""
        return os.path.join(self.get_yolo_models_dir(), model_name)
    
    def get_modelfile_ignored_agents_dir(self):
        """Get and ensure the ignored agents directory for ModelfileWriter
        
        Returns:
            str: Path to the ignored agents directory
        """
        # Create the path to the IgnoredAgents directory in the Ollama models directory
        ignored_agents_dir = os.path.join(self.get_ollama_models_dir(), IGNORED_AGENTS_DIR)
        os.makedirs(ignored_agents_dir, exist_ok=True)
        return ignored_agents_dir
    
    def get_modelfile_paths_dict(self):
        """Get dictionary of paths needed for ModelfileWriter and ConversionManager
        
        Returns:
            dict: Dictionary containing all paths needed for ModelfileWriter
        """
        # Get the ignored pipeline directory path
        ignored_pipeline_dir = os.path.join(self.get_ollama_models_dir(), 'agentFiles', 'ignoredPipeline')
        os.makedirs(ignored_pipeline_dir, exist_ok=True)

        return {
            'model_git_dir': self.get_model_dir(),
            'hf_cache_dir': self.get_hf_cache_dir(),
            'ollama_models_dir': self.get_ollama_models_dir(),
            'spells_path': self.get_spell_path(),
            'ignored_agents_dir': self.get_modelfile_ignored_agents_dir(),
            'current_dir': self._paths['base']['current_dir'],
            'parent_dir': self._paths['base']['parent_dir'],
            'ignored_pipeline_dir': ignored_pipeline_dir
        }