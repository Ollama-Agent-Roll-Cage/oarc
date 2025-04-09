"""
YOLO API implementation that extends the ServerAPI base class.
Provides HTTP and WebSocket endpoints for YOLO object detection.
"""

from fastapi import HTTPException, UploadFile, WebSocket

from oarc.server.server_api import ServerAPI
from oarc.utils.log import log
from oarc.yolo.processor import YoloProcessor

class YoloServerAPI(ServerAPI):
    """
    API for YOLO object detection with endpoints for image detection and WebSocket streaming.
    This class extends ServerAPI for consistency with the server architecture.
    """
    
    # Singleton instance
    _instance = None
    

    @classmethod
    def get_instance(cls):
        """
        Get or create the singleton instance of YoloAPI.
        Ensures that only one instance of the API is created and reused.
        """
        if cls._instance is None:
            cls._instance = cls(server_name="YoloAPI", host="localhost", port=8000)
        return cls._instance
    

    def __init__(self, server_name="YoloAPI", host="localhost", port=8000):
        """
        Initialize the YoloAPI with server configuration and the YOLO processor.
        
        Args:
            server_name (str): Name of the server.
            host (str): Host address for the server.
            port (int): Port number for the server.
        """
        super().__init__(server_name, host, port)
        
        # Ensure initialization happens only once when used as a singleton
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        # Use the YoloProcessor singleton for object detection
        self.yolo_processor = YoloProcessor.get_instance()
        self.initialized = True
        log.info("YoloAPI singleton initialized")
    

    def setup_routes(self):
        """
        Set up API routes for YOLO object detection.
        Registers HTTP and WebSocket endpoints for detection and streaming.
        """
        super().setup_routes()
        
        # Register HTTP endpoint for object detection
        self.add_route("/detect", self.detect_objects, methods=["POST"])
        
        # Register WebSocket endpoint for real-time streaming
        self.add_websocket_route("/stream", self.vision_stream)
        
        log.info("YOLO API routes configured")
        

    async def detect_objects(self, image: UploadFile):
        """
        Process an uploaded image with YOLO and return detected objects.
        
        Args:
            image (UploadFile): The uploaded image file to process.
        
        Returns:
            dict: A dictionary containing the detected objects.
        
        Raises:
            HTTPException: If an error occurs during image processing.
        """
        log.info(f"Detection request received: {image.filename}")
        try:
            # Read the contents of the uploaded image file
            contents = await image.read()
            
            # Perform object detection using YoloProcessor
            _, detections = await self.yolo_processor.detect_api(contents)
            
            return {"detections": detections}
            
        except Exception as e:
            log.error(f"Error processing image: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
    async def vision_stream(self, websocket: WebSocket):
        """
        Stream YOLO detections over a WebSocket connection.
        
        Args:
            websocket (WebSocket): The WebSocket connection to stream detections to.
        
        Raises:
            Exception: If an error occurs during WebSocket communication.
        """
        log.info("WebSocket connection requested")
        try:
            # Delegate WebSocket handling to YoloProcessor
            await self.yolo_processor.stream_api(websocket)
        except Exception as e:
            log.error(f"WebSocket error: {e}", exc_info=True)