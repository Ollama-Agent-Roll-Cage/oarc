"""
Abstract base class for all server implementations in OARC.

This module defines the core server interface that all server implementations must follow.
It establishes a consistent API for server lifecycle management including initialization,
starting, stopping, and status checking capabilities.
"""

import abc
import threading
import uvicorn
from fastapi import FastAPI, Request

from oarc.utils.log import log


class Server(abc.ABC):
    """
    Abstract base class defining the interface for all server implementations.
    
    This class establishes the required methods that concrete server implementations
    must provide, ensuring a consistent interface for managing server lifecycle.
    """
    
    # Singleton pattern support
    _instances = {}
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """Get or create the singleton instance of this server"""
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]
    
    def __init__(self, server_name="GenericServer", host="localhost", port=8000):
        """Initialize the server with basic configuration."""
        self.server_name = server_name
        self.host = host
        self.port = port
        self.is_running = False
        self.server_thread = None
        
        # Create a FastAPI application for this server
        self.app = FastAPI(title=server_name)
        
        log.info(f"Initializing {self.server_name} at {self.host}:{self.port}")
    
    def initialize(self):
        """Initialize server resources and prepare for startup."""
        log.info(f"Initializing {self.server_name}")
        
        # Add default root endpoint for health checks
        self.app.add_api_route("/", self.root_handler, methods=["GET"])
        
        return True
    
    async def root_handler(self, request: Request):
        """Default handler for root endpoint providing health check"""
        return {"status": "active", "server": self.server_name}
    
    def start(self):
        """Start the server."""
        log.info(f"Starting {self.server_name} on {self.host}:{self.port}")
        
        # Create uvicorn configuration
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run server in a separate thread to avoid blocking
        self.server_thread = threading.Thread(target=server.run, daemon=True)
        self.server_thread.start()
        
        self.is_running = True
        log.info(f"{self.server_name} running at {self.get_url()}")
        return True
    
    def stop(self):
        """Stop the server and release resources."""
        log.info(f"Stopping {self.server_name}")
        self.is_running = False
        # Note: There's no clean way to stop a uvicorn server from code
        # In a real production environment, use a process manager
        return True
    
    def status(self):
        """Get the current status of the server."""
        return {
            "name": self.server_name,
            "running": self.is_running,
            "url": self.get_url() if self.is_running else None
        }
    
    def get_url(self):
        """Get the server's URL."""
        protocol = "https" if self.is_secure() else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    def is_secure(self):
        """Check if the server uses secure connections."""
        return False
