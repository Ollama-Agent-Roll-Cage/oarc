"""
This module provides a FastAPI-based API for managing agent storage. It defines the AgentStorageAPI class, which 
sets up various endpoints to create, list, retrieve, load, delete, and reset agents. Additionally, the API allows for 
updating agent models and flags to meet different configuration needs. Each endpoint interacts with an underlying 
AgentStorage system to ensure that agents are managed consistently, with robust error handling via HTTPExceptions.
"""

import logging

from fastapi import APIRouter, HTTPException
from typing import Dict, Optional

from oarc.database import AgentStorage

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class AgentStorageAPI:
    """
    A FastAPI-based API for managing agent storage operations.

    This class provides a comprehensive set of endpoints to perform various operations on agents, such as creating, 
    listing, retrieving, loading, deleting, and resetting agents. It also supports updating agent configurations, 
    including models and flags, to accommodate diverse use cases and requirements. 

    Each endpoint is designed to interact with the underlying AgentStorage system, ensuring consistent and reliable 
    management of agents. Robust error handling is implemented to provide meaningful HTTP responses in case of 
    failures, making the API resilient and user-friendly.

    The API is structured to facilitate easy integration into larger systems, enabling seamless management of agents 
    in dynamic environments.
    """
    
    def __init__(self):
        """Initialize the AgentStorageAPI with a FastAPI router.

        This constructor sets up the API router and initializes the routes for managing agent storage operations. 
        It ensures that the API is ready to handle requests for creating, listing, retrieving, loading, deleting, 
        resetting agents, and updating their configurations.
        """
        self.router = APIRouter()
        self.setup_routes()
    

    def setup_routes(self):
        """
        This module defines the `setup_routes` method, which configures API endpoints 
        for managing agent storage operations. The routes provide functionality for 
        creating, listing, retrieving, updating, and deleting agents, as well as 
        managing agent configurations, models, and commands. Additionally, it includes 
        endpoints for resetting agents to default templates and retrieving available 
        models and commands.
        Routes:
        - POST /api/agent/create: Create a new agent from a template.
        - GET /api/agent/list: Retrieve a list of available agents.
        - GET /api/agent/{agent_id}: Retrieve the configuration of a specific agent.
        - POST /api/agent/load: Load an existing agent.
        - DELETE /api/agent/{agent_id}: Delete an existing agent.
        - POST /api/agent/reset: Reset all agents to their default templates.
        - GET /api/agent/models: Retrieve a list of available models.
        - GET /api/agent/commands: Retrieve the available command library.
        - PUT /api/agent/{agent_id}/flags: Update the flags of a specific agent.
        - PUT /api/agent/{agent_id}/models: Update the models of a specific agent.
        Each route is implemented as an asynchronous function and includes error 
        handling to return appropriate HTTP status codes and error messages in case 
        of failures.
        """

        @self.router.post("/api/agent/create")
        async def create_agent(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
            """
            Create a new agent from a specified template.

            This endpoint allows users to create a new agent by providing the name of an existing template, 
            a unique agent ID, and an optional custom configuration. The custom configuration can be used 
            to override default settings in the template.

            Args:
            template_name (str): The name of the template to use for creating the agent.
            agent_id (str): A unique identifier for the new agent.
            custom_config (Optional[Dict]): An optional dictionary containing custom configuration 
                            parameters to override the template defaults.

            Returns:
            dict: A dictionary containing the status of the operation and the created agent's details.

            Raises:
            HTTPException: If an error occurs during agent creation, an HTTP 500 error is returned 
                       with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent = agent_storage.create_agent_from_template(template_name, agent_id, custom_config)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/list")
        async def list_agents(self):
            """
            Retrieve a list of all available agents.

            This endpoint provides a list of agents currently stored in the system. 
            It interacts with the underlying AgentStorage to fetch the agent details.

            Returns:
            dict: A dictionary containing a list of available agents.

            Raises:
            HTTPException: If an error occurs while retrieving the agents, an HTTP 500 error 
                   is returned with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agents = await agent_storage.list_available_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/{agent_id}")
        async def get_agent(self, agent_id: str):
            """
            Retrieve the configuration of a specific agent.

            This endpoint fetches the configuration details of an agent identified by its unique ID. 
            If the agent is not found, a 404 HTTP error is returned.

            Args:
            agent_id (str): The unique identifier of the agent whose configuration is to be retrieved.

            Returns:
            dict: A dictionary containing the agent's configuration details.

            Raises:
            HTTPException: If the agent is not found (404) or if an error occurs during retrieval (500).
            """
            try:
                agent_storage = AgentStorage()
                agent_config = agent_storage.get_agent_config(agent_id)
                if not agent_config:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                return agent_config
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.post("/api/agent/load")
        async def load_agent(self, agent_id: str):
            """
            Load an existing agent by its unique identifier.

            This endpoint allows users to load an agent into memory for further operations. 
            It interacts with the underlying AgentStorage system to retrieve and initialize 
            the agent's state.

            Args:
            agent_id (str): The unique identifier of the agent to be loaded.

            Returns:
            dict: A dictionary containing the status of the operation and the loaded agent's details.

            Raises:
            HTTPException: If an error occurs during the loading process, an HTTP 500 error is returned 
                   with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent = agent_storage.load_agent(agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.delete("/api/agent/{agent_id}")
        async def delete_agent(self, agent_id: str):
            """
            Deletes an existing agent from the storage.

            This endpoint removes the agent identified by the provided `agent_id` 
            from the system. If the operation is successful, a success message is 
            returned. In case of an error during the deletion process, an HTTP 500 
            error is raised with the error details.

            Args:
                agent_id (str): The unique identifier of the agent to be deleted.

            Returns:
                dict: A response containing the status and a message indicating 
                the result of the deletion operation.

            Raises:
                HTTPException: If an error occurs during the deletion process, an 
                    HTTP 500 error is raised with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.purge_agent(agent_id)
                return {"status": "success", "message": f"Agent {agent_id} deleted"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.post("/api/agent/reset")
        async def reset_agents(self):
            """
            Endpoint to reset all agents to their default templates.

            This method handles a POST request to reset all agents stored in the system.
            It performs the following actions:
            - Purges all existing agents.
            - Sets up default agents.
            - Reloads agent templates.

            Returns:
                dict: A JSON response containing the status and a success message.

            Raises:
                HTTPException: If an error occurs during the reset process, 
                an HTTP 500 error is returned with the exception details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.purge_agents()
                agent_storage.setup_default_agents()
                agent_storage.reload_templates()
                return {"status": "success", "message": "All agents reset to defaults"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/models")
        async def get_available_models(self):
            """
            Retrieve a list of available models.

            This endpoint provides a list of models that can be used for agent configuration. 
            It interacts with the underlying AgentStorage system to fetch the available models.

            Returns:
            dict: A dictionary containing a list of available models.

            Raises:
            HTTPException: If an error occurs while retrieving the models, an HTTP 500 error 
               is returned with the error details.
            """
            try:
                agent_storage = AgentStorage()
                models = await agent_storage.get_available_models()
                return {"models": models}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/commands")
        async def get_command_library(self):
            """
            Retrieve the available command library.

            This endpoint provides a list of commands that can be used by agents. 
            It interacts with the underlying AgentStorage system to fetch the 
            command library, which includes all supported commands for agent operations.

            Returns:
            dict: A dictionary containing a list of available commands.

            Raises:
            HTTPException: If an error occurs while retrieving the commands, an HTTP 500 
                   error is returned with the error details.
            """
            try:
                agent_storage = AgentStorage()
                commands = await agent_storage.get_command_library()
                return {"commands": commands}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.put("/api/agent/{agent_id}/flags")
        async def update_agent_flags(self, agent_id: str, flags: Dict[str, bool]):
            """
            Update the flags of a specific agent.

            This endpoint allows users to modify the boolean flags associated with an agent. 
            Flags can be used to enable or disable specific features or behaviors of the agent.

            Args:
            agent_id (str): The unique identifier of the agent whose flags are to be updated.
            flags (Dict[str, bool]): A dictionary containing flag names as keys and their 
                         corresponding boolean values as values.

            Returns:
            dict: A dictionary containing the status of the operation and a success message.

            Raises:
            HTTPException: If an error occurs during the update process, an HTTP 500 error is 
                   returned with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.initialize_agent_storage(agent_id)
                
                # Update flags
                for flag_name, flag_value in flags.items():
                    if hasattr(agent_storage, flag_name):
                        setattr(agent_storage, flag_name, flag_value)
                
                agent_storage.save_agent_state()
                return {"status": "success", "message": f"Agent {agent_id} flags updated"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        
        @self.router.put("/api/agent/{agent_id}/models")
        async def update_agent_models(self, agent_id: str, models: Dict[str, str]):
            """
            Update the models of a specific agent.

            This endpoint allows users to update the models associated with an agent. 
            Models can include various components such as language models, embedding models, 
            vision models, and others. The provided model configuration is applied to the 
            specified agent, and the updated state is saved.

            Args:
            agent_id (str): The unique identifier of the agent whose models are to be updated.
            models (Dict[str, str]): A dictionary containing model names as keys and their 
                 corresponding configurations or identifiers as values.

            Returns:
            dict: A dictionary containing the status of the operation and a success message.

            Raises:
            HTTPException: If an error occurs during the update process, an HTTP 500 error is 
               returned with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.initialize_agent_storage(agent_id)
                
                # Update models
                if "largeLanguageModel" in models:
                    agent_storage.large_language_model = models["largeLanguageModel"]
                if "embeddingModel" in models:
                    agent_storage.embedding_model = models["embeddingModel"]
                if "visionModel" in models:
                    agent_storage.language_and_vision_model = models["visionModel"]
                if "yoloModel" in models:
                    agent_storage.yolo_model = models["yoloModel"]
                if "whisperModel" in models:
                    agent_storage.whisper_model = models["whisperModel"]
                if "voiceName" in models:
                    agent_storage.voice_name = models["voiceName"]
                
                agent_storage.save_agent_state()
                return {"status": "success", "message": f"Agent {agent_id} models updated"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))