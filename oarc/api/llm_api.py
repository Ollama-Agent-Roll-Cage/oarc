from typing import Optional
from fastapi import APIRouter, HTTPException
from base_api.BaseToolAPI import BaseToolAPI
from promptModel.multiModalPrompting import multiModalPrompting

#TODO decide wether to create custom apis by importing oarc utils, or to create apis 
# in each util and import then in the oarc_api

class LLMPromptAPI(BaseToolAPI):
    def __init__(self):
        super().__init__(prefix="/api/llm", tags=["language-model"])
    
    def setup_routes(self):
        @self.router.post("/complete")
        async def complete(self, prompt: str, agent_id: Optional[str] = None):
            try:
                prompt_handler = multiModalPrompting()
                if agent_id:
                    # Load agent configuration
                    prompt_handler.loaded_agent = await prompt_handler.load_agent(agent_id)
                response = await prompt_handler.send_prompt(prompt)
                return {"response": response}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))