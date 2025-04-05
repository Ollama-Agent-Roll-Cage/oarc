"""
ModelfileWriter is responsible for constructing Ollama modelfiles.
It handles the generation of modelfiles with appropriate parameters and configurations.
"""

import os
import json
from oarc.utils.paths import Paths
from oarc.utils.log import log
from oarc.utils.decorators.singleton import singleton

@singleton
class ModelfileWriter:
    """
    ModelfileWriter handles the creation of Ollama modelfiles based on 
    various templates and configurations.
    """
    
    def __init__(self, pathLibrary=None):
        """
        Initialize ModelfileWriter with path configuration.
        
        Args:
            pathLibrary (dict, optional): Dictionary of paths. If None, uses Paths singleton.
        """
        # Use the constructor to get the singleton instance
        self.paths = Paths()
        
        # Set path from provided library or from Paths
        if pathLibrary:
            self.spells_path = pathLibrary.get('spells_path')
            self.model_path = pathLibrary.get('ollama_models_dir')
            self.ignored_agents_dir = pathLibrary.get('ignored_agents_dir')
            # Add model_git_dir for compatibility
            self.model_git_dir = pathLibrary.get('model_git_dir')
        else:
            # Get all paths from the Paths singleton
            self.spells_path = self.paths.get_spell_path()
            self.model_path = self.paths.get_ollama_models_dir()
            self.ignored_agents_dir = self.paths.get_modelfile_ignored_agents_dir()
            self.model_git_dir = self.paths.get_model_dir()
        
        log.info(f"ModelfileWriter initialized with spells path: {self.spells_path}")
        log.info(f"ModelfileWriter using models path: {self.model_path}")
        
        # Create directories if they don't exist
        os.makedirs(self.spells_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.ignored_agents_dir, exist_ok=True)
        
        # Define ANSI color codes for terminal output
        self.colors = {
            'OKBLUE': '\033[94m',
            'WARNING': '\033[93m',
            'WHITE': '\033[97m',
            'RED': '\033[91m',
            'LIGHT_CYAN': '\033[96m',
            'BRIGHT_YELLOW': '\033[33;1m'
        }
        
        log.info(f"ModelfileWriter initialized with ignored_agents_dir: {self.ignored_agents_dir}")
        self.template = ""  # Initialize template property
    
    def write_model_file(self):
        """
        Writes a model file using user provided inputs.

        This method prompts the user for model details and writes the resulting
        configuration to a model file. It serves as a template for model file generation,
        ensuring that the necessary parameters are collected and formatted correctly.

        Args:
            None

        Returns:
            None
        """
        #TODO ADD WRITE MODEL FILE CLASS
        # collect agent data with text input
        self.user_create_agent_name = input(self.colors['WARNING'] + "<<< PROVIDE SAFETENSOR OR GGUF NAME (WITH .gguf or .safetensors) >>> " + self.colors['OKBLUE'])
        user_input_temperature = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + self.colors['OKBLUE'])

        model_create_dir = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            log.info(f"Created model directory: {model_create_dir}")
            
            # Get current model template data
            self.ollama_show_template()
            
            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_create_agent_name}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"TEMPLATE \"\"\"\n{self.template}\n\"\"\"\n")
            
            log.info(f"Created modelfile at: {model_create_file}")
            return
        except Exception as e:
            log.error(f"Error creating directory or text file: {str(e)}")
            return f"Error creating directory or text file: {str(e)}"
    

    def write_model_file_and_run_agent_create_ollama(self):
        """
        Executes the agent creation process by writing a model file and 
        executing the associated command. 
        
        This method collects the necessary user inputs, generates a model file 
        with the provided parameters, and triggers the agent creation command 
        for Ollama automation.
        
        Returns:
            None
        """
        # collect agent data with text input
        self.user_create_agent_name = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + self.colors['OKBLUE'])
        user_input_temperature = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + self.colors['OKBLUE'])
        system_prompt = input(self.colors['WHITE'] + "<<< PROVIDE SYSTEM PROMPT >>> " + self.colors['OKBLUE'])

        model_create_dir = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            log.info(f"Created model directory: {model_create_dir}")
            
            # Get current model template data
            self.ollama_show_template()
            
            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_input_model_select}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
                f.write(f"TEMPLATE \"\"\"\n{self.template}\n\"\"\"\n")
            
            log.info(f"Created modelfile at: {model_create_file}")

            # Execute create_agent_cmd - no longer needs to specify batch file
            success = self.create_agent_cmd(self.user_create_agent_name, '')
            return success
        except Exception as e:
            log.error(f"Error creating directory or text file: {str(e)}")
            return f"Error creating directory or text file: {str(e)}"


    def write_model_file_and_run_agent_create_gguf(self, model_git):
        """
        Automatically generate a new agent using predefined command-line instructions.

        This method sets up the configurations and executes the necessary command
        to create a new agent.
        
        Args:
            model_git: Path to the model git repository

        Returns:
            bool: True if successful, error message if failed
        """
        self.model_git = model_git

        # collect agent data with text input
        self.converted_gguf_model_name = input(self.colors['WARNING'] + "<<< PROVIDE SAFETENSOR OR CONVERTED GGUF NAME (with EXTENTION .gguf or .safetensors) >>> " + self.colors['OKBLUE'])
        self.user_create_agent_name = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + self.colors['OKBLUE'])
        user_input_temperature = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + self.colors['OKBLUE'])
        system_prompt = input(self.colors['WHITE'] + "<<< PROVIDE SYSTEM PROMPT >>> " + self.colors['OKBLUE'])

        model_create_dir = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}\\modelfile")
        self.gguf_path_part = os.path.join(self.model_git, "converted")
        self.gguf_path = os.path.join(self.gguf_path_part, f"{self.converted_gguf_model_name}.gguf")
        log.info(f"model_git: {model_git}")
        
        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            log.info(f"Created model directory: {model_create_dir}")

            # Copy gguf to IgnoredAgents dir
            self.copy_gguf_to_ignored_agents()
            log.info(f"Copied GGUF file to model directory")

            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM ./{self.converted_gguf_model_name}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
            
            log.info(f"Created modelfile at: {model_create_file}")
            
            # Execute create_agent_cmd - no longer needs to specify batch file
            success = self.create_agent_cmd(self.user_create_agent_name, '')
            return success
        except Exception as e:
            log.error(f"Error creating directory or text file: {str(e)}")
            return f"Error creating directory or text file: {str(e)}"