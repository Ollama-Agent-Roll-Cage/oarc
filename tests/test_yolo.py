"""
YOLO Testing Module

This module provides testing functionality for the YOLO object detection components
of the OARC package. It demonstrates how to initialize and use the YoloProcessor
and YoloServer components.

Functions:
    test_yolo_processor():
        Tests the YoloProcessor component by detecting objects in an image
        
    test_yolo_server():
        Tests the YoloServer component by starting the server and API
"""

import logging
import os
import cv2
import numpy as np

from oarc.yolo import YoloProcessor, YoloServer, YoloServerAPI
from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def test_yolo_processor():
    """Test YoloProcessor functionality.
    
    Initializes the YoloProcessor component with a default model path from Paths utility
    and tests object detection on a sample image.
    """
    log.info("Starting YoloProcessor test")
    
    # Get model path using Paths utility
    model_path = Paths.get_yolo_default_model_path()
    
    if not os.path.exists(model_path):
        log.warning(f"Default YOLO model not found at {model_path}")
        log.info("Initializing YoloProcessor without model")
        processor = YoloProcessor.get_instance()
    else:
        log.info(f"Initializing YoloProcessor with model from {model_path}")
        processor = YoloProcessor.get_instance(model_path=model_path)
    
    # Create a sample image for testing
    sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(sample_img, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(sample_img, "Test Object", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Process the sample image
    log.info("Processing test image")
    processed_img, detections = processor.process_frame(sample_img, return_detections=True)
    
    log.info(f"Image processed with {len(detections) if detections else 0} detections")
    log.info("YoloProcessor test completed")
    
    return processor

def test_yolo_server():
    """Test YoloServer functionality.
    
    Initializes and starts the YoloServer and YoloServerAPI components.
    """
    log.info("Starting YoloServer test")
    
    # Get the processor singleton
    processor = YoloProcessor.get_instance()
    
    # Initialize server components
    server = YoloServer.get_instance()
    api = YoloServerAPI.get_instance()
    
    log.info("Initializing YoloServer")
    server.initialize()
    
    log.info("Initializing YoloServerAPI")
    api.setup_routes()
    
    # Don't actually start the server in testing to avoid port conflicts
    log.info("YoloServer components initialized successfully")
    log.info("YoloServer test completed")
    
    return server, api

if __name__ == "__main__":
    try:
        processor = test_yolo_processor()
        server, api = test_yolo_server()
    except KeyboardInterrupt:
        log.info("Tests interrupted by user")
    except Exception as e:
        log.error(f"Error in tests: {str(e)}", exc_info=True)
