"""
OARC Project Paths Utility

This module provides the `Paths` singleton class, which centralizes the management of essential directories 
used throughout the OARC project. It ensures the existence of directories for models, HuggingFace caches, 
Ollama models, and other resources, creating them as needed. By using the singleton pattern, this utility 
ensures consistent path configurations across the application, while also supporting dynamic updates when 
environment variables change.
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
    YOLO_DIR,
    OUTPUT_DIR
)

# Define a constant for the ignored agents directory
IGNORED_AGENTS_DIR = "IgnoredAgents"

@singleton
class Paths:
    """
    Singleton class for managing and centralizing path configurations in the OARC project.

    This class ensures consistent access to essential directories, dynamically updates paths
    when environment variables change, and provides utility methods to validate and log paths.
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
        This ensures that any changes to environment variables are reflected in the path configurations
        without requiring a restart of the application.
        """
        log.info("Refreshing paths configuration")
        
        # Store current environment variables
        env_vars = {
            'OARC_MODEL_GIT': os.getenv('OARC_MODEL_GIT'),
            'HF_HOME': os.getenv('HF_HOME'),
            'OLLAMA_MODELS': os.getenv('OLLAMA_MODELS'),
            'OARC_OUTPUT_DIR': os.getenv('OARC_OUTPUT_DIR')  # Add environment variable for output directory
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
            self._paths['base']['model_dir'] = model_dir
            
            # Update HF cache directory
            hf_home = env_vars['HF_HOME']
            if not hf_home:
                hf_home = os.path.join(self._paths['base']['model_dir'], HUGGINGFACE_DIR)
                os.makedirs(hf_home, exist_ok=True)
            self._paths['models']['hf_cache'] = hf_home
            
            # Update Ollama models directory
            ollama_models = env_vars['OLLAMA_MODELS']
            if not ollama_models:
                ollama_models = os.path.join(self._paths['base']['model_dir'], OLLAMA_MODELS_DIR)
                os.makedirs(ollama_models, exist_ok=True)
            self._paths['models']['ollama_models'] = ollama_models
            
            # Update other paths that derive from the base model directory
            spells_path = os.path.join(self._paths['base']['model_dir'], SPELLS_DIR)
            os.makedirs(spells_path, exist_ok=True)
            self._paths['models']['spells'] = spells_path
            
            # TTS-related paths
            coqui_dir = os.path.join(self._paths['base']['model_dir'], COQUI_DIR)
            os.makedirs(coqui_dir, exist_ok=True)
            self._paths['tts']['coqui'] = coqui_dir
            
            # Set custom_coqui_dir using the constants properly
            custom_coqui_dir = os.path.join(coqui_dir, CUSTOM_COQUI_DIR)
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
            
            # Update working directory values - renamed for clarity
            self._paths['base']['current_path'] = os.getcwd()
            self._paths['base']['parent_path'] = os.path.dirname(os.getcwd())
            
            # Set up global output directory
            # Check for environment variable override
            if env_vars['OARC_OUTPUT_DIR']:
                output_dir = env_vars['OARC_OUTPUT_DIR']
                log.info(f"Using OARC_OUTPUT_DIR environment variable for output: {output_dir}")
            else:
                # Default: Use current working directory (pwd) instead of project root
                output_dir = os.path.join(self._paths['base']['current_path'], OUTPUT_DIR)
                log.info(f"Using current working directory for output: {output_dir}")
            
            os.makedirs(output_dir, exist_ok=True)
            self._paths['base']['output_dir'] = output_dir
            
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
    
    def get_coqui_path(self):
        """Get Coqui TTS models directory"""
        return self._paths['tts']['coqui']
    
    def get_whisper_dir(self):
        """Get Whisper STT models directory"""
        return self._paths['tts']['whisper']
    
    def get_generated_dir(self):
        """Get generated audio output directory"""
        return self._paths['tts']['generated']
    
    def get_voice_ref_path(self):
        """Get voice reference samples directory"""
        return self._paths['tts']['voice_reference']
    
    def get_tts_paths_dict(self):
        """Get dictionary of TTS-related paths
        
        Returns:
            dict: Dictionary containing all paths needed for TTS functionality
        """
        return {
            'current_path': self._paths['base']['current_path'],  # Renamed
            'parent_path': self._paths['base']['parent_path'],    # Renamed
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
                    for _, subpath in path.items():
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

    def get_output_dir(self):
        """
        Get the global output directory path.
        
        This directory is located in the current working directory (pwd) by default,
        but can be overridden by setting the OARC_OUTPUT_DIR environment variable.
        It's intended for all types of output files across the application,
        providing a centralized location for generated content.
        
        Returns:
            str: Full path to the output directory
        """
        return self._paths['base']['output_dir']
    
    def get_output_subdir(self, subdir_name):
        """
        Get a subdirectory within the global output directory, creating it if needed.
        
        Args:
            subdir_name (str): Name of the subdirectory to create/retrieve
            
        Returns:
            str: Full path to the output subdirectory
        """
        output_subdir = os.path.join(self.get_output_dir(), subdir_name)
        os.makedirs(output_subdir, exist_ok=True)
        return output_subdir
    
    def get_test_output_dir(self):
        """
        Get the directory for test outputs, creating it if needed.
        
        Returns:
            str: Full path to the test output directory
        """
        return self.get_output_subdir("tests")

    def log_paths(self):
        """
        Log all currently configured paths to help with debugging and verification.
        This provides a clear overview of where the system is looking for various resources.
        """
        log.info("=" * 50)
        log.info("OARC PATH CONFIGURATION")

        # Base paths
        log.info("----- Base Paths -----")
        log.info(f"Project Root: {self._paths['base'].get('project_root', 'Not set')}")
        log.info(f"Current: {self._paths['base'].get('current_path', 'Not set')}")  # Renamed
        log.info(f"Parent: {self._paths['base'].get('parent_path', 'Not set')}")    # Renamed
        log.info(f"Model: {self._paths['base'].get('model_dir', 'Not set')}")
        log.info(f"Output: {self._paths['base'].get('output_dir', 'Not set')}")
        
        # Model paths
        log.info("----- Model Paths -----")
        log.info(f"HF Cache: {self._paths['models'].get('hf_cache', 'Not set')}")
        log.info(f"Ollama Models: {self._paths['models'].get('ollama_models', 'Not set')}")
        log.info(f"Spells Path: {self._paths['models'].get('spells', 'Not set')}")
        
        # TTS paths
        log.info("----- TTS Paths -----")
        log.info(f"Coqui: {self._paths['tts'].get('coqui', 'Not set')}")
        log.info(f"Whisper: {self._paths['tts'].get('whisper', 'Not set')}")
        log.info(f"Generated: {self._paths['tts'].get('generated', 'Not set')}")
        log.info(f"Voice Reference: {self._paths['tts'].get('voice_reference', 'Not set')}")
        
        # Environment variables status
        log.info("----- Environment Variables -----")
        for var_name, value in self._paths['env_vars'].items():
            status = "Set" if value else "Not set (using default)"
            log.info(f"{var_name}: {status}")
        
        log.info("=" * 50)
    
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
            'current_path': self._paths['base']['current_path'],  # Renamed
            'parent_path': self._paths['base']['parent_path'],    # Renamed
            'ignored_pipeline_dir': ignored_pipeline_dir
        }
    
    def get_custom_coqui_dir(self):
        """
        Get the directory path for custom Coqui XTTS v2 models.
        
        This path points to the custom_xtts_v2 directory within the Coqui directory,
        where fine-tuned voice models are stored.
        
        Returns:
            str: Full path to the custom XTTS v2 models directory
        """
        return self._paths['tts']['custom_coqui']