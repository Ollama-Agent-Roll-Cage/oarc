"""
Gradio-based server implementation for OARC.

This module provides a concrete implementation of the Server abstract base class
using Gradio to create interactive web interfaces for AI models and components.
"""

import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable

import gradio as gr

from oarc.server.server import Server
from oarc.utils.log import log


class GradioServer(Server):
    """
    Gradio-specific server implementation of the abstract Server class.
    
    This class encapsulates Gradio's web interface capabilities, providing
    methods to create, configure, and manage Gradio applications.
    """
    
    def __init__(self, 
                 server_name="GradioServer", 
                 host="localhost", 
                 port=7860,
                 theme="default",
                 auth=None):
        """
        Initialize a Gradio server instance.
        
        Args:
            server_name: Name of the server
            host: Host address to bind to
            port: Port number to listen on
            theme: Gradio theme to apply
            auth: Authentication credentials if required
        """
        super().__init__(server_name, host, port)
        self.theme = theme
        self.auth = auth
        self.demo = None
        self.components = {}
        self.event_handlers = []
        log.info(f"Initializing Gradio server with theme: {theme}")
    
    def initialize(self):
        """Initialize Gradio server resources."""
        log.info(f"Initializing {self.server_name}")
        try:
            self._initialize_gradio_app()
            return True
        except Exception as e:
            log.error(f"Failed to initialize Gradio app: {e}", exc_info=True)
            return False
    
    def _initialize_gradio_app(self):
        """Create the Gradio app instance with appropriate configuration."""
        log.debug("Creating Gradio Blocks app")
        self.demo = gr.Blocks(title=self.server_name, theme=self.theme, analytics_enabled=False)
        log.debug("Gradio app instance created")
    
    def add_component(self, name: str, component_type: str, **kwargs):
        """
        Add a Gradio component to the server.
        
        Args:
            name: Unique name for the component
            component_type: Type of Gradio component (e.g., "Textbox", "Image")
            **kwargs: Additional parameters for the component
            
        Returns:
            The created Gradio component
        """
        log.info(f"Adding {component_type} component: {name}")
        if not hasattr(gr, component_type):
            log.error(f"Invalid Gradio component type: {component_type}")
            raise ValueError(f"Invalid Gradio component type: {component_type}")
        
        try:
            component_class = getattr(gr, component_type)
            component = component_class(**kwargs)
            self.components[name] = component
            return component
        except Exception as e:
            log.error(f"Error creating component {name}: {e}", exc_info=True)
            raise
    
    def add_event_handler(self, trigger_component, fn, inputs=None, outputs=None):
        """
        Register an event handler for a component.
        
        Args:
            trigger_component: Component that triggers the event
            fn: Function to call when event is triggered
            inputs: List of input components
            outputs: List of output components
        """
        if not self.demo:
            log.error("Cannot add event handler before initializing app")
            raise RuntimeError("Initialize app before adding event handlers")
        
        log.info(f"Adding event handler for {trigger_component}")
        handler = {
            "trigger": trigger_component,
            "function": fn,
            "inputs": inputs or [],
            "outputs": outputs or []
        }
        self.event_handlers.append(handler)
    
    def _setup_layout(self):
        """Set up the Gradio interface layout."""
        log.info("Setting up Gradio interface layout")
        # This should be overridden in subclasses
    
    def _register_event_handlers(self):
        """Register all event handlers with the Gradio app."""
        log.info(f"Registering {len(self.event_handlers)} event handlers")
        with self.demo:
            for handler in self.event_handlers:
                if hasattr(handler["trigger"], "click"):
                    handler["trigger"].click(
                        fn=handler["function"],
                        inputs=handler["inputs"],
                        outputs=handler["outputs"]
                    )
    
    def start(self):
        """Start the Gradio server."""
        log.info(f"=========== STARTING {self.server_name.upper()} ===========")
        log.info(f"Starting {self.server_name} on {self.host}:{self.port}")
        if not self.demo:
            log.error("Cannot start server: Gradio app not initialized")
            return False
        
        try:
            self._setup_layout()
            self._register_event_handlers()
            
            # Launch the Gradio server
            self.demo.launch(
                server_name=self.host,
                server_port=self.port,
                auth=self.auth,
                share=False,
                quiet=True
            )
            
            self.is_running = True
            log.info(f"{self.server_name} running at {self.get_url()}")
            log.info(f"=========== {self.server_name.upper()} STARTED ===========")
            return True
        except Exception as e:
            log.error(f"Failed to start Gradio server: {e}", exc_info=True)
            self.is_running = False
            return False
    
    def stop(self):
        """Stop the Gradio server."""
        log.info(f"Stopping {self.server_name}")
        if self.is_running and self.demo:
            try:
                self.demo.close()
                self.is_running = False
                log.info(f"{self.server_name} stopped")
                return True
            except Exception as e:
                log.error(f"Error stopping server: {e}", exc_info=True)
                return False
        return True
    
    def status(self):
        """Get the current status of the Gradio server."""
        status_info = {
            "name": self.server_name,
            "running": self.is_running,
            "url": self.get_url() if self.is_running else None,
            "component_count": len(self.components),
            "event_handler_count": len(self.event_handlers)
        }
        log.debug(f"Gradio server status: {status_info}")
        return status_info
