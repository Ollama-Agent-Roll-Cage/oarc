"""
Gradio-based server API implementation for OARC.

This module extends the ServerAPI class to provide specific API functionality
for Gradio servers, including integration between REST API endpoints and
Gradio components, managing asynchronous interactions, and handling
file uploads and multimedia content.
"""

import asyncio
import inspect
from typing import Dict, List, Any, Callable, Optional, Union
import json

import gradio as gr
from fastapi import FastAPI, Request, Response, HTTPException

from oarc.server.server_api import ServerAPI
from oarc.utils.log import log


class GradioServerAPI(ServerAPI):
    """
    API implementation specifically for Gradio servers.
    
    This class provides the bridge between FastAPI routes and Gradio components,
    enabling REST API access to Gradio functions and interfaces.
    """
    
    def __init__(self, 
                 server_name="GradioServerAPI", 
                 host="localhost", 
                 port=7861,
                 gradio_server=None):
        """
        Initialize a Gradio server API.
        
        Args:
            server_name: Name of the API server
            host: Host address to bind to
            port: Port number to listen on
            gradio_server: Optional reference to an existing GradioServer instance
        """
        super().__init__(server_name, host, port)
        self.app = FastAPI(title=server_name)
        self.gradio_server = gradio_server
        self.api_routes = {}
        log.info(f"Initializing GradioServerAPI on port {port}")
    
    def initialize(self):
        """Initialize the Gradio API server."""
        log.info(f"Initializing {self.server_name}")
        super().initialize()
        
        # Set up default error handlers
        self._setup_error_handlers()
        
        return True
    
    def _setup_error_handlers(self):
        """Set up global error handlers for the API."""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            log.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
            return Response(
                status_code=exc.status_code,
                content=json.dumps({"error": exc.detail}),
                media_type="application/json"
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            log.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            return Response(
                status_code=500,
                content=json.dumps({"error": "Internal server error", "detail": str(exc)}),
                media_type="application/json"
            )
    
    def setup_routes(self):
        """Set up API routes for the Gradio server."""
        log.info("Setting up Gradio API routes")
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "server": self.server_name}
        
        # Status endpoint
        @self.app.get("/status")
        async def server_status():
            status = self.status()
            if self.gradio_server:
                status["gradio"] = self.gradio_server.status()
            return status
        
        # If there's a connected Gradio server, set up proxy routes
        if self.gradio_server:
            self._setup_gradio_proxy_routes()
    
    def _setup_gradio_proxy_routes(self):
        """Set up routes that proxy to Gradio server components."""
        if not self.gradio_server:
            log.warning("No Gradio server connected, skipping proxy routes")
            return
        
        log.info("Setting up Gradio proxy routes")
        
        # Register API endpoints for each Gradio component function
        for name, handler in self._get_gradio_functions().items():
            route_path = f"/api/{name}"
            
            # Create a closure to capture the handler
            async def create_endpoint(handler=handler):
                async def endpoint(request: Request):
                    try:
                        # Parse request body as JSON
                        data = await request.json()
                        
                        # Execute the Gradio function
                        result = await asyncio.to_thread(handler, **data)
                        return {"result": result}
                    except Exception as e:
                        log.error(f"Error in endpoint {route_path}: {str(e)}", exc_info=True)
                        raise HTTPException(status_code=500, detail=str(e))
                
                return endpoint
            
            # Register the endpoint
            self.app.post(route_path)(create_endpoint(handler))
            log.info(f"Registered API endpoint: POST {route_path}")
    
    def _get_gradio_functions(self) -> Dict[str, Callable]:
        """Extract callable functions from Gradio event handlers."""
        if not self.gradio_server:
            return {}
        
        functions = {}
        for handler in self.gradio_server.event_handlers:
            fn = handler["function"]
            name = fn.__name__
            functions[name] = fn
        
        return functions
    
    def connect_gradio_server(self, gradio_server):
        """Connect this API to a Gradio server instance."""
        log.info(f"Connecting to Gradio server: {gradio_server.server_name}")
        self.gradio_server = gradio_server
        # Re-setup routes to include Gradio proxy routes
        self.setup_routes()
    
    def start(self):
        """Start the Gradio API server."""
        log.info(f"=========== STARTING {self.server_name.upper()} ===========")
        log.info(f"Starting {self.server_name} on {self.host}:{self.port}")
        
        # Start the FastAPI server
        import uvicorn
        try:
            # Start the server in a non-blocking way
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Run the server in a separate thread
            import threading
            self.server_thread = threading.Thread(target=server.run, daemon=True)
            self.server_thread.start()
            
            self.is_running = True
            log.info(f"{self.server_name} running at {self.get_url()}")
            log.info(f"=========== {self.server_name.upper()} STARTED ===========")
            return True
        except Exception as e:
            log.error(f"Failed to start Gradio API server: {e}", exc_info=True)
            self.is_running = False
            return False
    
    def stop(self):
        """Stop the Gradio API server."""
        log.info(f"Stopping {self.server_name}")
        # There's no direct way to stop a uvicorn server gracefully from within
        # the application. In a production environment, you would use a process
        # manager or container orchestrator to handle this.
        self.is_running = False
        log.info(f"{self.server_name} stopping requested")
        return True
