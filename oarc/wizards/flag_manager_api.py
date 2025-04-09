from fastapi import APIRouter

class FlagManagerAPI:
    def __init__(self):
        self.router = APIRouter()
        self.setup_routes()
    
    def setup_routes(self):
        @self.router.post("/api/flag-manager")
        async def get_agent_state():
            # TODO this needs to call this method in FlagManager.py
            pass