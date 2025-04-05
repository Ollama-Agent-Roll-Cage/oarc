"""
The model_write_class provides methods for writing custom ollama modelfile's through automation.
This process allows for on the spot model creation with defined parameters for any given model or agent
build/workflow. 
"""

import os
from oarc.utils.log import log
from oarc.utils.paths import Paths

class ModelfileWriter:

    def __init__(self, pathLibrary=None):
        """
        Initialize the ModelfileWriter class.
        
        Args:
            pathLibrary (dict, optional): Legacy parameter for backward compatibility.
                                         If provided, specific paths can be overridden.
        """
        # Get paths from the singleton
        self.paths = Paths.get_instance()
        
        # Set up paths using the singleton
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.dirname(self.current_dir)
        self.ignored_agents_dir = os.path.join(self.paths.get_model_dir(), "AgentFiles", "Ignored_Agents")
        
        # Override with provided paths if any (for backward compatibility)
        if pathLibrary:
            log.info("Using provided pathLibrary for compatibility")
            if 'current_dir' in pathLibrary:
                self.current_dir = pathLibrary['current_dir']
            if 'parent_dir' in pathLibrary:
                self.parent_dir = pathLibrary['parent_dir']
            if 'ignored_agents_dir' in pathLibrary:
                self.ignored_agents_dir = pathLibrary['ignored_agents_dir']
        
        # Ensure directories exist
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