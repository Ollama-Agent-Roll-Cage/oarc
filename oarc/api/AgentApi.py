from typing import Optional
from fastapi import APIRouter
from base_api.BaseToolAPI import BaseToolAPI
from pandasDB.agentStorage import AgentStorage

#TODO decide wether to create custom apis by importing oarc utils, or to create apis 
# in each util and import then in the oarc_api
class AgentAPI(BaseToolAPI):
    def __init__(self):
        super().__init__(prefix="/api/agent", tags=["agent-management"])
    
    def setup_routes(self):
        @self.router.post("/create")
        async def create_agent(self, template: str, agent_id: str):
            storage = AgentStorage()
            agent = await storage.create_agent_from_template(template, agent_id)
            return {"status": "success", "agent": agent}

        @self.router.get("/state/{agent_id}")
        async def get_agent_state(self, agent_id: str):
            storage = AgentStorage()
            state = await storage.get_agent_state(agent_id)
            return state