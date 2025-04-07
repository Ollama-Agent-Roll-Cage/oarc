"""
This module contains the class OllamaCommands which contains methods for interacting with the Ollama library.
It provides a comprehensive set of methods for managing models, generating responses, and retrieving information.
"""

import asyncio
import json
import logging
import sys
from typing import List, Dict, Union, Optional, Any, AsyncGenerator

try:
    import ollama
    from ollama import AsyncClient, Client, ResponseError
except ImportError:
    logging.error("Ollama library not installed. Please install with 'pip install ollama'")
    raise ImportError("Ollama library not installed. Please install with 'pip install ollama'")


class OllamaCommands:
    """
    A class for interacting with the Ollama library, providing both synchronous and asynchronous
    methods for model management, response generation, and information retrieval.
    """

    def __init__(self, host: str = "http://localhost:11434", async_mode: bool = True):
        """
        Initialize the OllamaCommands class.

        Args:
            host: The URL of the Ollama server (default: "http://localhost:11434")
            async_mode: Whether to use the async client by default (default: True)
        """
        self.name = "ollamaCommands"
        self.host = host
        self.async_mode = async_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.sync_client = None
        self.async_client = None
        
        # Create the sync client
        try:
            self.sync_client = Client(host=self.host)
            self.logger.info(f"Initialized Ollama sync client with host: {self.host}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama sync client: {str(e)}")
        
        # Create the async client if in async mode
        if self.async_mode:
            try:
                self.async_client = AsyncClient(host=self.host)
                self.logger.info(f"Initialized Ollama async client with host: {self.host}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Ollama async client: {str(e)}")
    
    def quit(self) -> None:
        """Exit the application."""
        sys.exit()

    # Model Information Methods
    
    async def ollama_show_modelfile(self, user_input_model_select: str) -> Dict[str, Any]:
        """
        Get the full model information for a specified model.
        
        Args:
            user_input_model_select: The name of the model to show
            
        Returns:
            Dict containing all model information
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            if self.async_client:
                return await self.async_client.show(f"{user_input_model_select}")
            return ollama.show(f"{user_input_model_select}")
        except ResponseError as e:
            self.logger.error(f"Error showing model {user_input_model_select}: {str(e)}")
            raise
    
    async def ollama_show_template(self, user_input_model_select: str) -> str:
        """
        Get just the template for a specified model.
        
        Args:
            user_input_model_select: The name of the model to get the template from
            
        Returns:
            The model's prompt template as a string
        """
        try:
            modelfile_data = await self.ollama_show_modelfile(user_input_model_select)
            return modelfile_data.get('template', '')
        except Exception as e:
            self.logger.error(f"Error retrieving template for {user_input_model_select}: {str(e)}")
            return ''
    
    async def ollama_show_license(self, user_input_model_select: str) -> str:
        """
        Get just the license information for a specified model.
        
        Args:
            user_input_model_select: The name of the model to get the license from
            
        Returns:
            The model's license information as a string
        """
        try:
            modelfile_data = await self.ollama_show_modelfile(user_input_model_select)
            return modelfile_data.get('license', '')
        except Exception as e:
            self.logger.error(f"Error retrieving license for {user_input_model_select}: {str(e)}")
            return ''

    async def ollama_show_loaded_models(self) -> List[Dict[str, Any]]:
        """
        Get information about currently running models.
        
        Returns:
            List of dictionaries containing information about running models
        """
        try:
            if self.async_client:
                return await self.async_client.ps()
            return ollama.ps()
        except Exception as e:
            self.logger.error(f"Error listing loaded models: {str(e)}")
            return []
    
    async def ollama_list(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names
        """
        try:
            self.logger.info("Calling ollama.list() to get available models")
            
            if self.async_client:
                result = await self.async_client.list()
            else:
                result = ollama.list()
                
            self.logger.info(f"ollama.list() result: {result}")
            if not isinstance(result, dict) or 'models' not in result:
                self.logger.warning("Unexpected response format from ollama.list()")
                return []
                
            # Extract and format model names
            models = [model['name'] for model in result['models'] if 'name' in model]
            self.logger.info(f"Extracted models: {models}")
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []

    # Model Management Methods
    
    async def ollama_pull(self, model_name: str) -> Dict[str, Any]:
        """
        Pull a model from the Ollama library.
        
        Args:
            model_name: The name of the model to pull
            
        Returns:
            Dictionary with pull status information
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            self.logger.info(f"Pulling model: {model_name}")
            
            if self.async_client:
                return await self.async_client.pull(model_name)
            return ollama.pull(model_name)
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {str(e)}")
            raise
    
    async def ollama_delete(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a model from local storage.
        
        Args:
            model_name: The name of the model to delete
            
        Returns:
            Dictionary with deletion status information
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            self.logger.info(f"Deleting model: {model_name}")
            
            if self.async_client:
                return await self.async_client.delete(model_name)
            return ollama.delete(model_name)
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {str(e)}")
            raise
    
    async def ollama_copy(self, source_model: str, target_model: str) -> Dict[str, Any]:
        """
        Copy a model to a new name.
        
        Args:
            source_model: The name of the source model
            target_model: The name for the new copy
            
        Returns:
            Dictionary with copy status information
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            self.logger.info(f"Copying model {source_model} to {target_model}")
            
            if self.async_client:
                return await self.async_client.copy(source_model, target_model)
            return ollama.copy(source_model, target_model)
        except Exception as e:
            self.logger.error(f"Error copying model {source_model} to {target_model}: {str(e)}")
            raise
    
    async def ollama_create(self, 
                          model_name: str, 
                          from_model: str, 
                          system_prompt: str = None,
                          modelfile: str = None) -> Dict[str, Any]:
        """
        Create a new model based on an existing one.
        
        Args:
            model_name: The name for the new model
            from_model: The base model to use
            system_prompt: Optional system prompt to use
            modelfile: Optional modelfile content
            
        Returns:
            Dictionary with creation status information
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            self.logger.info(f"Creating model {model_name} from {from_model}")
            
            kwargs = {"model": model_name, "from_": from_model}
            if system_prompt:
                kwargs["system"] = system_prompt
            if modelfile:
                kwargs["modelfile"] = modelfile
                
            if self.async_client:
                return await self.async_client.create(**kwargs)
            return ollama.create(**kwargs)
        except Exception as e:
            self.logger.error(f"Error creating model {model_name}: {str(e)}")
            raise

    # Generation Methods
    
    async def ollama_chat(self, 
                         model_name: str, 
                         messages: List[Dict[str, str]], 
                         stream: bool = False,
                         options: Dict[str, Any] = None) -> Union[Dict[str, Any], AsyncGenerator]:
        """
        Chat with a model.
        
        Args:
            model_name: The name of the model to use
            messages: A list of message dictionaries with 'role' and 'content' keys
            stream: Whether to stream the response
            options: Additional options for the chat
            
        Returns:
            If stream=False, a dictionary with the response
            If stream=True, an async generator yielding response chunks
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            kwargs = {"model": model_name, "messages": messages, "stream": stream}
            if options:
                kwargs.update(options)
                
            if self.async_mode:
                if not self.async_client:
                    self.logger.error("Async client not initialized but async_mode is True")
                    raise RuntimeError("Async client not initialized")
                
                return await self.async_client.chat(**kwargs)
            
            return ollama.chat(**kwargs)
        except Exception as e:
            self.logger.error(f"Error in chat with model {model_name}: {str(e)}")
            raise
    
    async def ollama_generate(self, 
                            model_name: str, 
                            prompt: str, 
                            stream: bool = False,
                            options: Dict[str, Any] = None) -> Union[Dict[str, Any], AsyncGenerator]:
        """
        Generate a response from a model.
        
        Args:
            model_name: The name of the model to use
            prompt: The prompt to send to the model
            stream: Whether to stream the response
            options: Additional options for generation
            
        Returns:
            If stream=False, a dictionary with the response
            If stream=True, an async generator yielding response chunks
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            kwargs = {"model": model_name, "prompt": prompt, "stream": stream}
            if options:
                kwargs.update(options)
                
            if self.async_mode:
                if not self.async_client:
                    self.logger.error("Async client not initialized but async_mode is True")
                    raise RuntimeError("Async client not initialized")
                
                return await self.async_client.generate(**kwargs)
            
            return ollama.generate(**kwargs)
        except Exception as e:
            self.logger.error(f"Error in generate with model {model_name}: {str(e)}")
            raise
    
    async def ollama_embed(self, 
                         model_name: str, 
                         input_text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get embeddings for text.
        
        Args:
            model_name: The name of the model to use
            input_text: The text to embed or a list of texts for batch embedding
            
        Returns:
            Dictionary with the embeddings
            
        Raises:
            ResponseError: If there's an error from the Ollama API
        """
        try:
            kwargs = {"model": model_name, "input": input_text}
                
            if self.async_mode:
                if not self.async_client:
                    self.logger.error("Async client not initialized but async_mode is True")
                    raise RuntimeError("Async client not initialized")
                
                return await self.async_client.embed(**kwargs)
            
            return ollama.embed(**kwargs)
        except Exception as e:
            self.logger.error(f"Error in embed with model {model_name}: {str(e)}")
            raise

    # Utility Methods
    
    async def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive details about a model including parameters, size, and more.
        
        Args:
            model_name: The name of the model
            
        Returns:
            Dictionary with detailed model information
        """
        try:
            model_data = await self.ollama_show_modelfile(model_name)
            
            # Get running instance details if model is loaded
            loaded_models = await self.ollama_show_loaded_models()
            running_instance = None
            
            for model in loaded_models:
                if model.get('name') == model_name:
                    running_instance = model
                    break
            
            # Combine the information
            details = {
                "name": model_name,
                "parameters": model_data.get('parameters', {}),
                "template": model_data.get('template', ''),
                "license": model_data.get('license', ''),
                "modelfile": model_data.get('modelfile', ''),
                "running": running_instance is not None
            }
            
            if running_instance:
                details["instance"] = running_instance
            
            return details
        except Exception as e:
            self.logger.error(f"Error getting details for model {model_name}: {str(e)}")
            return {"name": model_name, "error": str(e)}
    
    async def get_models_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available and running models.
        
        Returns:
            Dictionary with model summary information
        """
        try:
            available_models = await self.ollama_list()
            running_models = await self.ollama_show_loaded_models()
            
            return {
                "available_count": len(available_models),
                "available_models": available_models,
                "running_count": len(running_models),
                "running_models": running_models
            }
        except Exception as e:
            self.logger.error(f"Error getting models summary: {str(e)}")
            return {"error": str(e)}
    
    def is_model_running(self, model_name: str, running_models: List[Dict[str, Any]] = None) -> bool:
        """
        Check if a specific model is currently running.
        
        Args:
            model_name: The name of the model to check
            running_models: Optional list of running models (to avoid extra API call)
            
        Returns:
            True if the model is running, False otherwise
        """
        try:
            if running_models is None:
                # Get running models if not provided
                if self.async_mode:
                    # Create a new event loop for the synchronous call
                    loop = asyncio.new_event_loop()
                    running_models = loop.run_until_complete(self.ollama_show_loaded_models())
                    loop.close()
                else:
                    running_models = ollama.ps()
            
            for model in running_models:
                if model.get('name') == model_name:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking if model {model_name} is running: {str(e)}")
            return False
    
    async def ensure_model_available(self, model_name: str, auto_pull: bool = False) -> bool:
        """
        Ensure a model is available locally, optionally pulling it if not.
        
        Args:
            model_name: The name of the model to check
            auto_pull: Whether to automatically pull the model if not available
            
        Returns:
            True if the model is available, False otherwise
        """
        try:
            available_models = await self.ollama_list()
            
            if model_name in available_models:
                return True
                
            if auto_pull:
                self.logger.info(f"Model {model_name} not available locally, pulling...")
                await self.ollama_pull(model_name)
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error ensuring model {model_name} is available: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Simple synchronous example
    def run_sync_example():
        commands = OllamaCommands(async_mode=False)
        
        # Get available models
        loop = asyncio.new_event_loop()
        models = loop.run_until_complete(commands.ollama_list())
        loop.close()
        
        print("Available models:", models)
        
        if models:
            # Use the first available model
            model = models[0]
            
            # Generate a response
            response = ollama.generate(model=model, prompt="Hello, how are you?")
            print(f"Response from {model}:", response.get('response', ''))
    
    # Asynchronous example
    async def run_async_example():
        commands = OllamaCommands()
        
        # Get summary of models
        summary = await commands.get_models_summary()
        print("Models summary:", json.dumps(summary, indent=2))
        
        if summary.get('available_models'):
            # Use the first available model
            model = summary['available_models'][0]
            
            # Ensure the model is available
            available = await commands.ensure_model_available(model, auto_pull=True)
            
            if available:
                # Get model details
                details = await commands.get_model_details(model)
                print(f"Model details for {model}:", json.dumps(details, indent=2))
                
                # Chat with the model
                response = await commands.ollama_chat(
                    model_name=model,
                    messages=[{"role": "user", "content": "What is the meaning of life?"}]
                )
                print("Chat response:", response.get('message', {}).get('content', ''))
    
    # Run examples
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(run_async_example())
    else:
        run_sync_example()