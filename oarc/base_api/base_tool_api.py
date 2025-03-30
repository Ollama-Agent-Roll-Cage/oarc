"""
Base Tool API Module

This module establishes the BaseToolAPI class, which provides a foundational structure 
for building tool-specific API endpoints using FastAPI. It handles:
- Initialization of API routes with custom prefix and tags
- Path management for model directories using the centralized Paths utility
- Standardized method interfaces requiring subclasses to implement specific API routes

The design ensures consistent environment variable handling and directory structure
across all tool APIs in the OARC system.
"""

import logging

from fastapi import APIRouter
from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class BaseToolAPI:
    """Base class for all tool-specific APIs.
    
    Provides common functionality for API route setup and path management.
    Requires subclasses to implement setup_routes() method.
    """

    def __init__(self, prefix="/", tags=None):
        """Initialize the BaseToolAPI with routes configuration and path management.
        
        Args:
            prefix: URL prefix for all routes in this API
            tags: OpenAPI tags for route documentation
        """
        # Initialize router with prefix and tags
        self.router = APIRouter(prefix=prefix, tags=tags or [])
        
        # Initialize paths utility for consistent directory management
        self.paths = Paths()
        self.model_dir = self.paths.get_model_dir()
        
        # Setup routes (implemented by subclasses)
        self.setup_routes()
        
        log.debug(f"BaseToolAPI initialized with prefix '{prefix}' and model directory: {self.model_dir}")
    

    def setup_routes(self):
        """Define and set up API routes for the tool.
        
        This abstract method must be implemented by each subclass to specify
        the API endpoints and their corresponding request handlers. It ensures
        that each tool provides its own unique functionality while adhering to
        the standardized structure of the BaseToolAPI.
        """
        raise NotImplementedError("Subclasses must implement setup_routes()")