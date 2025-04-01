"""
Server API handler for OARC server implementations.

This module extends the base Server class to provide API routing capabilities,
handling requests and directing them to the appropriate handlers based on 
endpoint patterns and HTTP methods. It also supports WebSocket routes for
real-time communication.
"""

import inspect
from typing import Dict, Callable, Any, List, Optional

from oarc.server.server import Server
from oarc.utils.log import log


class ServerAPI(Server):
    """
    Server API implementation that extends the base Server class.
    
    This class provides API routing capabilities, managing endpoints
    and handlers, and processing incoming requests for both HTTP and
    WebSocket protocols.
    """
    
    # Singleton pattern support
    _instances = {}
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """Get or create the singleton instance of this API server"""
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]
    
    def __init__(self, server_name="GenericServerAPI", host="localhost", port=8000):
        """Initialize the server API with routing tables."""
        super().__init__(server_name, host, port)
        self.routes = {}
        self.websocket_routes = {}  # New dictionary for WebSocket routes
        self.middleware = []
        log.info(f"Initializing ServerAPI with name: {server_name}")
    
    def initialize(self):
        """Initialize API server resources."""
        log.info(f"Initializing {self.server_name} API")
        self.setup_routes()
        return True
    
    def setup_routes(self):
        """
        Set up API routes by registering handlers.
        
        This method should be overridden by subclasses to define
        specific API endpoints and their corresponding handlers.
        """
        log.info("Setting up API routes")
        # Concrete implementations should override this method
    
    def add_route(self, path: str, handler: Callable, methods: List[str] = ["GET"]):
        """Register a route with the API."""
        if path not in self.routes:
            self.routes[path] = {}
        
        for method in methods:
            self.routes[path][method.upper()] = handler
            log.info(f"Added route: {method} {path}")
    
    def add_websocket_route(self, path: str, handler: Callable):
        """
        Register a WebSocket route with the API.
        
        Args:
            path: The endpoint path
            handler: The async function to handle WebSocket connections
        """
        self.websocket_routes[path] = handler
        log.info(f"Added WebSocket route: {path}")
    
    def add_middleware(self, middleware_func: Callable):
        """Add middleware to the API request processing pipeline."""
        self.middleware.append(middleware_func)
        log.info(f"Added middleware: {middleware_func.__name__}")
    
    def handle_request(self, path: str, method: str, **kwargs):
        """
        Process an incoming API request.
        
        Args:
            path: The endpoint path
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional parameters for the request
            
        Returns:
            Response data from the handler or error information
        """
        log.info(f"Handling request: {method} {path}")
        
        # Apply middleware
        for middleware in self.middleware:
            continue_processing = middleware(path, method, **kwargs)
            if not continue_processing:
                log.warning(f"Request blocked by middleware: {middleware.__name__}")
                return {"error": "Request blocked by middleware"}
        
        # Find and execute handler
        if path in self.routes and method.upper() in self.routes[path]:
            handler = self.routes[path][method.upper()]
            try:
                log.debug(f"Executing handler for {method} {path}")
                return handler(**kwargs)
            except Exception as e:
                log.error(f"Error in handler for {method} {path}: {str(e)}", exc_info=True)
                return {"error": str(e)}
        else:
            log.warning(f"No handler found for {method} {path}")
            return {"error": f"No handler for {method} {path}"}
    
    def apply_routes_to_app(self, app):
        """
        Apply all routes to a FastAPI application.
        
        Args:
            app: FastAPI application instance
            
        Returns:
            bool: True if routes were applied successfully
        """
        try:
            log.info(f"Applying routes to FastAPI application")
            
            # Apply regular HTTP routes
            for path, methods in self.routes.items():
                for method, handler in methods.items():
                    app.add_api_route(path, handler, methods=[method])
                    log.debug(f"Applied {method} {path} to FastAPI app")
            
            # Apply WebSocket routes
            for path, handler in self.websocket_routes.items():
                app.add_api_websocket_route(path, handler)
                log.debug(f"Applied WebSocket route {path} to FastAPI app")
                
            return True
        except Exception as e:
            log.error(f"Error applying routes to app: {e}", exc_info=True)
            return False
    
    @property
    def router(self):
        """
        Provide a FastAPI compatible router for including the API routes.
        
        This property converts the internal routes and websocket_routes dictionaries
        to a FastAPI APIRouter that can be included in a FastAPI application.
        
        Returns:
            APIRouter: A FastAPI router containing all API routes
        """
        from fastapi import APIRouter
        router = APIRouter()
        
        # Add standard HTTP routes
        if hasattr(self, 'routes') and self.routes:
            for path, methods in self.routes.items():
                for method_name, handler in methods.items():
                    method = getattr(router, method_name.lower(), None)
                    if method:
                        method(path)(handler)
        
        # Add WebSocket routes
        if hasattr(self, 'websocket_routes') and self.websocket_routes:
            for path, handler in self.websocket_routes.items():
                router.websocket(path)(handler)
                
        return router
    
    def start(self):
        """Start the API server."""
        log.info(f"Starting {self.server_name} API server at {self.host}:{self.port}")
        self.is_running = True
    
    def stop(self):
        """Stop the API server."""
        log.info(f"Stopping {self.server_name} API server")
        self.is_running = False
    
    def status(self):
        """Get the current status of the API server."""
        status_info = {
            "name": self.server_name,
            "running": self.is_running,
            "endpoint": f"{self.host}:{self.port}",
            "routes": {path: list(methods.keys()) for path, methods in self.routes.items()},
            "websocket_routes": list(self.websocket_routes.keys()),
            "middleware_count": len(self.middleware)
        }
        log.debug(f"Server status: {status_info}")
        return status_info
