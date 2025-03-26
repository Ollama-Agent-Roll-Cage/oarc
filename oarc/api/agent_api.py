from typing import Optional, Dict
from fastapi import APIRouter, HTTPException
from base_api.BaseToolAPI import BaseToolAPI
from pandasDB.agentStorage import AgentStorage

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