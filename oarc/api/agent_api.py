"""
Agent API Module
----------------
This module defines the AgentAPI class which provides endpoints for managing agents.
Endpoints:
    - POST /create: Create a new agent from a given template.
    - GET /state/{agent_id}: Retrieve the current state of an agent.
    - POST /load: Load an existing agent instance.

Dependencies:
    - FastAPI for API creation and request handling.
    - AgentStorage for interacting with agent data persistence.

Usage:
    Import and instantiate AgentAPI within your FastAPI application to include agent management functionality.
"""

from fastapi import HTTPException

from oarc.base_api.BaseToolAPI import BaseToolAPI
from oarc.database.agentStorage import AgentStorage

class AgentAPI(BaseToolAPI):
    def __init__(self):
        super().__init__(prefix="/api/agent", tags=["agent-management"])
        self.agent_storage = AgentStorage()
    
    def setup_routes(self):
        @self.router.post("/create")
        async def create_agent(self, template: str, agent_id: str):
            try:
                agent = await self.agent_storage.create_agent_from_template(template, agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/state/{agent_id}")
        async def get_agent_state(self, agent_id: str):
            try:
                state = await self.agent_storage.get_agent_state(agent_id)
                return state
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.router.post("/load")
        async def load_agent(self, agent_id: str):
            try:
                agent = await self.agent_storage.load_agent(agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))