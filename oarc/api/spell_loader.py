#!/usr/bin/env python3
"""
This module defines the SpellLoader class which orchestrates the initialization of the chatbot wizard's environment.
It configures the directory structure for storing models, agent configurations, pipelines, and speech processing data,
and integrates necessary components such as the Ollama commands, model writer, and conversion manager.
"""

import os
import logging

from oarc.ollamaUtils.modelfileFactory.conversion_manager import ConversionManager
from oarc.ollamaUtils import ModelfileWriter
from oarc.ollamaUtils.ollama_commands import OllamaCommands
from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class SpellLoader():
    """
    The SpellLoader class is responsible for setting up and managing the environment 
    required for the ollamaAgentRollCage chatbot wizard. It initializes the directory 
    structure for models, agents, pipelines, and speech processing, ensuring all necessary 
    paths exist. Additionally, it integrates key components such as Ollama commands, 
    model writing utilities, and a conversion manager to facilitate seamless operation 
    of the chatbot wizard's functionalities.
    """


    def __init__(self):
        """
        Initialize the SpellLoader instance and set up the environment.
        This includes configuring paths, initializing base directories, 
        and preparing the necessary components for the chatbot wizard's operation.
        """
        self.paths = Paths()
        self.initializeBasePaths()
        self.initializeSpells()
        

    def initializeBasePaths(self):
        """
        Set up the foundational directory structure required for storing model data, 
        cache files, and other resources. This ensures that all necessary paths 
        are created and ready for use by the chatbot wizard.
        """
        # Get model directory using Paths utility
        self.model_git_path = self.paths.get_model_dir()
        log.info(f"Using model directory: {self.model_git_path}")
        
        # Get HuggingFace cache directory
        self.hf_cache_path = self.paths.get_hf_cache_dir()
        log.info(f"Using HuggingFace cache directory: {self.hf_cache_path}")
        
        # Get Ollama models directory
        self.ollama_models_path = self.paths.get_ollama_models_dir()
        log.info(f"Using Ollama models directory: {self.ollama_models_path}")
        
        # Get spell directory
        self.spell_path = self.paths.get_spell_path()
        log.info(f"Using spell directory: {self.spell_path}")

        # Get base directories
        self.current_path = os.getcwd()
        self.parent_path = os.path.abspath(os.path.join(self.current_path, os.pardir))
            
        # Initialize base path structure
        #TODO UPDATE TO STORE ALL NEW MODEL PATHS CORRECTLY and CREATE CENTRAL MODEL MANAGER FOR OLLAMA AND HUGGING FACE HUB
            
        self.pathLibrary = {
            # Main directories
            'current_dir': self.current_path,
            'parent_dir': self.parent_path,
            'model_git_dir': self.model_git_path,
            'ollama_models_dir': self.ollama_models_path,
                
            # Model directories 
            'huggingface_models': {
                'base_dir': os.path.join(self.model_git_path, 'huggingface'),
                'whisper': os.path.join(self.model_git_path, 'huggingface', 'whisper'),
                'xtts': os.path.join(self.model_git_path, 'huggingface', 'xtts'),
                'yolo': os.path.join(self.model_git_path, 'huggingface', 'yolo'),
                'llm': os.path.join(self.model_git_path, 'huggingface', 'llm')
            },
                
            # Agent directories
            'ignored_agents_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredAgents'),
            'agent_files_dir': os.path.join(self.model_git_path, 'agentFiles', 'publicAgents'),
            'ignored_agentfiles': os.path.join(self.model_git_path, 'agentFiles', 'ignoredAgentfiles'),
            'public_agentfiles': os.path.join(self.model_git_path, 'agentFiles', 'publicAgentfiles'),
                
            # Pipeline directories
            'ignored_pipeline_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline'),
            'llava_library_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'llavaLibrary'),
            'conversation_library_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'conversationLibrary'),
                
            # Data constructor directories
            'image_set_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'dataConstructor', 'imageSet'),
            'video_set_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'dataConstructor', 'videoSet'),
                
            # Speech directories
            'speech_library_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'speechLibrary'),
            'recognize_speech_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'recognizeSpeech'),
            'generate_speech_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'generateSpeech'),
            'tts_voice_ref_wav_pack_dir': os.path.join(self.model_git_path, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'publicVoiceReferencePack'),
        }

        # Ensure all directories in pathLibrary exist
        self.paths.ensure_paths(self.pathLibrary)

        log.info("Base paths initialized successfully.")


    def initializeSpells(self):
        """
        Initialize all spell-related components required for the chatbot wizard.
        This includes setting up commands, model writing utilities, 
        conversion managers, and other necessary tools for seamless operation.
        """
        log.info("Spells initializing...")
        self.ollama_commands = OllamaCommands() # initialize ollama commands
        self.model_write_class = ModelfileWriter(self.pathLibrary) # Write model files
        self.create_convert_manager = ConversionManager(self.pathLibrary)  # Create model manager
        self.tts_processor = None # TTS processor (initialize as None, will be created when needed)
            
        log.info("Spells initialized successfully.")
        return True
