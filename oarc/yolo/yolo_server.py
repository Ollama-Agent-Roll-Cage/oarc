"""
YOLO Server implementation that extends the Server base class.
Provides a server for running YOLO object detection services.
"""

from oarc.server.server import Server
from oarc.utils.log import log
from oarc.yolo.server_api import YoloServerAPI
from oarc.yolo.processor import YoloProcessor

class YoloServer(Server):
    """
    Server implementation for YOLO object detection.
    This class extends the Server base class for consistency with the server architecture.
    
    This server provides a web interface and API for performing object detection using
    YOLO models. It handles both HTTP requests and WebSocket connections for real-time
    detection streaming.
    """
    

    def __init__(self, server_name="YoloServer", host="localhost", port=8000):
        """Initialize the YoloServer with configuration and processor"""
        super().__init__(server_name, host, port)
        
        # Get the processor and API singletons
        self.processor = YoloProcessor.get_instance()
        self.api = YoloServerAPI.get_instance()
        
        log.info(f"YoloServer initialized on {host}:{port}")
    

    def initialize(self):
        """Initialize server resources and apply API routes"""
        log.info(f"Initializing {self.server_name}")
        
        # Initialize the parent server first (adds default root endpoint)
        super().initialize()
        
        # Make sure the API routes are set up
        self.api.setup_routes()
        
        # Apply the API routes to our FastAPI app
        self.api.apply_routes_to_app(self.app)
        
        log.info(f"{self.server_name} initialization complete")
        return True
    
    
    def status(self):
        """Get the enhanced server status including YOLO component status"""
        # Get base status from parent class
        status_info = super().status()
        
        # Add YOLO-specific status information
        status_info.update({
            "processor": {
                "initialized": hasattr(self.processor, "model") and self.processor.model is not None,
                "model": self.processor.model_path if hasattr(self.processor, "model_path") else None,
                "device": self.processor.device if hasattr(self.processor, "device") else None
            },
            "api": {
                "initialized": self.api.initialized if hasattr(self.api, "initialized") else False,
                "routes": len(self.api.routes) if hasattr(self.api, "routes") else 0,
                "websocket_routes": len(self.api.websocket_routes) if hasattr(self.api, "websocket_routes") else 0
            }
        })
        
        return status_info
