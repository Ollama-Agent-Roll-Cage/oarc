"""
Module used to use commands to convert between various data structurs
"""

import os
import shutil
import json

from oarc.utils.log import log
from oarc.ollama.modelfile.create_ollama_model import create_ollama_model
from oarc.ollama.modelfile.safetensors_to_gguf import convert_safetensors_to_gguf

 
class ConversionManager:


    def __init__(self, pathLibrary):
        """Initialize the ConversionManager instance with the specified library paths.
        
        Args:
            pathLibrary (dict): A dictionary containing configuration keys such as 
                    'current_dir', 'parent_dir', and 'ignored_pipeline_dir'.
        """
        self.pathLibrary = pathLibrary
        self.current_dir = self.pathLibrary['current_dir']
        self.parent_dir = self.pathLibrary['parent_dir']
        self.ignored_pipeline_dir = self.pathLibrary['ignored_pipeline_dir']


    def safe_tensor_gguf_convert(self, safe_tensor_input_name):
        """Convert a safetensors model to GGUF format.

        Args:
            safe_tensor_input_name (str): The name of the input safetensors model to convert.

        Returns:
            bool: True if conversion was successful, False otherwise
        """
        log.info(f"Converting {safe_tensor_input_name} from safetensors to GGUF")
        success, output_path = convert_safetensors_to_gguf(self.model_git, safe_tensor_input_name)
        
        if success:
            log.info(f"CONVERTED: {safe_tensor_input_name}")
            log.info(f"<<< USER >>> ")
            return True
        return False


    def create_agent_cmd(self, user_create_agent_name, cmd_file_name):
        """Execute the agent creation process to create an agent.

        Args:
            user_create_agent_name (str): The name to assign to the new agent.
            cmd_file_name (str): Deprecated parameter, kept for backward compatibility
        
        Returns:
            bool: True if agent creation was successful, False otherwise
        """
        try:
            log.info(f"Creating agent: {user_create_agent_name}")
            # Use the new Python function instead of the batch file
            return create_ollama_model(user_create_agent_name)
        except Exception as e:
            log.error(f"Error executing create_agent_cmd: {str(e)}")
            return False


    def copy_gguf_to_ignored_agents(self):
        """
        Prepares the GGUF file for conversion to Ollama format by ensuring the file is correctly positioned and contains all necessary metadata.
        """
        self.create_ollama_model_dir = os.path.join(self.ignored_agents, self.user_create_agent_name)
        log.info(f"Copying GGUF file to: {self.create_ollama_model_dir}")
        log.debug(f"Source path: {self.gguf_path}")
        log.debug(f"Destination directory: {self.create_ollama_model_dir}")
        
        # Copy the file from self.gguf_path to create_ollama_model_dir
        shutil.copy(self.gguf_path, self.create_ollama_model_dir)
        return
    

    def write_dict_to_json(self, dictionary, file_path):
        """
        Serialize the provided dictionary to a JSON file at the specified file path.

        Args:
            dictionary (dict): The dictionary to write to JSON.
            file_path (str): The path where the JSON file will be created.
        """
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)

        # write_dict_to_json(general_navigator_agent, 'general_navigator_agent.json')


    def read_json_to_dict(file_path):
        """
        Deserialize a JSON file to a dictionary."
        """
        # # Example usage
        # general_navigator_agent = read_json_to_dict('general_navigator_agent.json')
        # print(general_navigator_agent)

        with open(file_path, 'r') as json_file:
            dictionary = json.load(json_file)
        return dictionary