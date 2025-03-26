from fastapi import APIRouter, HTTPException
import os

class BaseToolAPI:
    def __init__(self, prefix: str, tags: list[str]):
        self.router = APIRouter(prefix=prefix, tags=tags)
        self.model_git_dir = self.get_model_dir()
        self.setup_routes()
    
    def get_model_dir(self):
        """Get and validate model directory"""
        model_dir = os.getenv('OARC_MODEL_GIT')
        if not model_dir:
            raise EnvironmentError("OARC_MODEL_GIT environment variable not set")
        return model_dir
    
    def setup_routes(self):
        """Each tool implements its own routes"""
        raise NotImplementedError