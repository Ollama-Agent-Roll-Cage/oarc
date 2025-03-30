"""
Agent API Module
----------------
This module defines the `AgentAPI` class, which provides a FastAPI-based interface 
for managing agents. It includes endpoints for creating, retrieving, and loading 
agent instances, enabling seamless interaction with agent-related data.

Endpoints:
    - POST /create: Create a new agent using a specified template and unique identifier.
    - GET /state/{agent_id}: Retrieve the current state of an agent by its unique identifier.
    - POST /load: Load an existing agent instance by its unique identifier.

Dependencies:
    - FastAPI: For API creation and request handling.
    - AgentStorage: For managing agent data persistence and operations.

Usage:
    Import and instantiate the `AgentAPI` class within your FastAPI application to 
    integrate agent management functionality into your service.
"""

from fastapi import HTTPException

from oarc.base_api.base_tool_api import BaseToolAPI
from oarc.database.agent_storage import AgentStorage


class AgentAPI(BaseToolAPI):
    """
    FastAPI wrapper for agent management functionality.

    This class provides endpoints for creating, retrieving, and loading agents.
    It serves as a bridge between the FastAPI framework and the underlying 
    agent storage system, enabling seamless interaction with agent-related data.
    """

    def __init__(self):
        """
        Initialize the AgentAPI instance.

        This constructor sets up the API with a predefined prefix and tags for 
        categorizing agent management endpoints. It also initializes the 
        AgentStorage instance for handling agent-related data operations.
        """
        super().__init__(prefix="/api/agent", tags=["agent-management"])
        self.agent_storage = AgentStorage()
    

    def setup_routes(self):
        """
        Set up the API routes for managing agents.
        
        This method is responsible for configuring the endpoints that allow 
        interaction with agent-related functionality, such as creating, 
        retrieving, and loading agents.
        """


        @self.router.post("/create")
        async def create_agent(self, template: str, agent_id: str):
            """
            Asynchronously create a new agent instance from a specified template.

            Args:
            template (str): The template used to create the agent.
            agent_id (str): The unique identifier for the new agent.

            Returns:
            dict: A dictionary containing the status of the operation and the created agent data.

            Raises:
            HTTPException: If an error occurs during the creation process, 
                       an HTTP 500 error is raised with the error details.
            """
            try:
                agent = await self.agent_storage.create_agent_from_template(template, agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


        @self.router.get("/state/{agent_id}")
        async def get_agent_state(self, agent_id: str):
            """
            Retrieve the current state of an agent instance asynchronously.

            Args:
                agent_id (str): The unique identifier of the agent whose state is to be retrieved.

            Returns:
                The current state of the agent instance.

            Raises:
                HTTPException: If an error occurs while retrieving the agent state, 
                       an HTTP 500 error is raised with the error details.
            """
            try:
                state = await self.agent_storage.get_agent_state(agent_id)
                return state
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


        @self.router.post("/load")
        async def load_agent(self, agent_id: str):
            """
            Asynchronously load an existing agent instance using the provided agent ID.
            This method retrieves the agent's data from persistent storage and ensures 
            it is properly initialized and ready for interaction. If an error occurs 
            during the loading process, an HTTPException with a 500 status code is raised.
            Args:
                agent_id (str): The unique identifier of the agent to be loaded.
            Returns:
                dict: A dictionary containing the status of the operation and the loaded agent data.
            Raises:
                HTTPException: If an error occurs during the loading process, with details of the error.
            """
            try:
                agent = await self.agent_storage.load_agent(agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))