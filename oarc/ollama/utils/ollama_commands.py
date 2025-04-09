"""
This module contains the class ollama_commands which contains methods for interacting with the ollama library.
"""

import ollama
import sys
import logging

class OllamaCommands:

    def __init__(self):
        self.name = "ollamaCommands"
    
    def quit(self):
        sys.exit()

    async def ollama_show_modelfile(self, user_input_model_select):
        return ollama.show(f"{user_input_model_select}")
    
    async def ollama_show_template(self, user_input_model_select):
        modelfile_data = ollama.show(f"{user_input_model_select}")
        return modelfile_data.get('template', '')
    
    async def ollama_show_license(self, user_input_model_select):
        modelfile_data = ollama.show(f"{user_input_model_select}")
        return modelfile_data.get('license', '')

    async def ollama_show_loaded_models(self):
        ollama_loaded_models = ollama.ps()
        return ollama_loaded_models
    
    async def ollama_list(self):
        """Get list of available models"""
        try:
            logging.info("Calling ollama.list() to get available models")
            result = ollama.list()
            logging.info(f"ollama.list() result: {result}")
            
            # Check if it's the new ListResponse format (ollama._types.ListResponse)
            if hasattr(result, 'models'):
                # Extract model names from the models list attribute
                models = [model.model for model in result.models if hasattr(model, 'model')]
                logging.info(f"Extracted models from ListResponse format: {models}")
                return models
                
            # Check if it's the list of Model objects format
            elif isinstance(result, list):
                # Extract model names from the list of Model objects
                models = [model.model for model in result if hasattr(model, 'model')]
                logging.info(f"Extracted models from list format: {models}")
                return models
                
            # Check if it's the old format (dict with 'models' key)
            elif isinstance(result, dict) and 'models' in result:
                # Extract model names from dictionary format
                models = [model['name'] for model in result['models'] if 'name' in model]
                logging.info(f"Extracted models from dict format: {models}")
                return models
                
            else:
                logging.warning(f"Unexpected response format from ollama.list(): {type(result)}")
                return []
                
        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")
            return []
