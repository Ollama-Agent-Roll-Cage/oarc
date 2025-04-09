#!/usr/bin/env python3
"""
SpellLoader is responsible for loading and managing spells,
which are Ollama model configurations for specific use cases.
"""

import os
import logging
import json

from oarc.ollama.modelfile.conversion_manager import ConversionManager
from oarc.ollama import ModelfileWriter
from oarc.ollama.utils.ollama_commands import OllamaCommands
from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name=s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class SpellLoader:
    """
    SpellLoader class manages loading and initializing Ollama model configurations.
    It provides methods to access these configurations for use in the OARC system.
    """
    
    def __init__(self):
        """Initialize SpellLoader with path configuration."""
        log.info("Initializing SpellLoader")
        
        # Use the constructor to get the singleton instance
        self.paths = Paths()
        
        # Get required directories
        self.model_git_dir = self.paths.get_model_dir()
        self.hf_cache_dir = self.paths.get_hf_cache_dir()
        self.ollama_models_dir = self.paths.get_ollama_models_dir()
        self.spells_path = self.paths.get_spell_path()
        
        log.info(f"Using model directory: {self.model_git_dir}")
        log.info(f"Using HuggingFace cache directory: {self.hf_cache_dir}")
        log.info(f"Using Ollama models directory: {self.ollama_models_dir}")
        log.info(f"Using spell directory: {self.spells_path}")
        
        # Get comprehensive paths dictionary from Paths singleton
        # This ensures all needed keys are present for both ModelfileWriter and ConversionManager
        self.pathLibrary = self.paths.get_modelfile_paths_dict()
        
        # Add current_dir and parent_dir which ConversionManager needs
        self.pathLibrary['current_dir'] = os.getcwd()  
        self.pathLibrary['parent_dir'] = os.path.dirname(os.getcwd())
        
        # Ensure directories exist
        success = self.paths.ensure_paths(self.pathLibrary)
        if success:
            log.info("Base paths initialized successfully.")
        else:
            log.error("Failed to initialize paths.")
        
        # Initialize spells
        self.initializeSpells()
    
    def initializeSpells(self):
        """Initialize spell configurations and setup Ollama modelfile writer."""
        log.info("Spells initializing...")
        
        # Create ModelfileWriter instance with paths from Paths singleton
        self.model_write_class = ModelfileWriter(self.pathLibrary)
        
        self.ollama_commands = OllamaCommands() # initialize ollama commands
        
        # Get or update paths for conversion manager
        if 'ignored_pipeline_dir' not in self.pathLibrary:
            self.pathLibrary['ignored_pipeline_dir'] = os.path.join(self.pathLibrary['ollama_models_dir'], 'agentFiles', 'ignoredPipeline')
            os.makedirs(self.pathLibrary['ignored_pipeline_dir'], exist_ok=True)
        
        self.create_convert_manager = ConversionManager(self.pathLibrary)  # Create model manager
        self.tts_processor = None # TTS processor (initialize as None, will be created when needed)
            
        log.info("Spells initialized successfully.")
        return True
