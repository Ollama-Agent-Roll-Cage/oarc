"""
YoloProcessor module for handling object detection tasks with YOLO models.
This module provides a singleton class for detecting objects in images and video.
"""

import os
import json
from time import time
from typing import Optional, Dict, Any, Tuple, List
from fastapi import WebSocket

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import win32gui
import win32con
import win32ui
import win32api
import websockets
import tempfile

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False

from oarc.yolo.detected_object import DetectedObject
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.decorators.singleton import singleton

@singleton
class YoloProcessor:
    """
    YoloProcessor is responsible for handling object detection tasks using YOLO models.
    
    This class implements the Singleton pattern to ensure only one YOLO model is loaded
    and used across the application, which helps with memory efficiency.
    """
    
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initialize the YOLO processor with the specified model.
        
        Args:
            model_name (str): Name of the YOLO model to load
        """
        self.model_name = model_name
        self.model = None
        
        # Fix: Use the constructor pattern for the Paths singleton
        self.paths = Paths()
        
        # Get the YOLO models directory
        self.models_dir = self.paths.get_yolo_models_dir()
        
        log.info(f"YoloProcessor initialized with models directory: {self.models_dir}")
        log.info(f"Using model: {model_name}")
        
        # Initialize tracking-related attributes
        self.tracked_objects = []
        self.conf_threshold = 0.25
        self.debounce_time = 0.5  # seconds
        
        # Color mapping for visualization
        self.neon_colors = [
            (0, 255, 127),   # Spring Green
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 255, 0),   # Yellow
            (0, 127, 255),   # Deep Sky Blue
            (255, 0, 127),   # Deep Pink
            (127, 0, 255),   # Purple
            (255, 127, 0),   # Orange
        ]
        self.color_map = {}
        self.color_index = 0
        
        # Lazy loading - model will be loaded on first use
        if os.path.exists(os.path.join(self.models_dir, model_name)):
            self.load_model(os.path.join(self.models_dir, model_name))
        else:
            log.warning(f"Model file not found at {os.path.join(self.models_dir, model_name)}")
            log.info("YoloProcessor initialized without a model. Use load_model() to load one.")

    def load_model(self, model_path: str) -> bool:
        """Load YOLO model from specified path"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            self.model = YOLO(model_path)
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Assign a unique neon color for each class"""
        if class_name not in self.color_map:
            color = self.neon_colors[self.color_index % len(self.neon_colors)]
            self.color_map[class_name] = color
            self.color_index += 1
        return self.color_map[class_name]

    def find_matching_object(self, points: np.ndarray, class_name: str) -> Optional[DetectedObject]:
        """Find the closest matching object of the same class"""
        min_distance = float('inf')
        matching_object = None
        
        center = np.mean(points, axis=0)
        
        for obj in self.tracked_objects:
            if obj.class_name == class_name:
                obj_center = np.mean(obj.points, axis=0)
                distance = np.linalg.norm(center - obj_center)
                
                if distance < min_distance:
                    min_distance = distance
                    matching_object = obj
                    
        if min_distance > 50:  # Adjust threshold as needed
            matching_object = None
            
        return matching_object

    def draw_oriented_bbox(self, img: np.ndarray, points: np.ndarray, 
                         color: Tuple[int, int, int], label: Optional[str] = None) -> None:
        """Draw oriented bounding box using polygon points"""
        points = points.astype(np.int32)
        cv2.polylines(img, [points], True, color, 2)
        
        if label:
            x_min = min(points[:, 0])
            y_min = min(points[:, 1])
            cv2.putText(img, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_frame(self, frame: np.ndarray, 
                     return_detections: bool = False) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Process frame with YOLO OBB detection
        
        Args:
            frame: Input image/frame
            return_detections: Whether to return detection data
            
        Returns:
            Tuple of processed frame and optional list of detections
        """
        if self.model is None:
            return frame, None if return_detections else frame

        results = self.model(frame, verbose=False)
        current_detections = set()
        detections_data = [] if return_detections else None
        
        frame_out = frame.copy()
        
        for r in results:
            if hasattr(r, 'obb') and r.obb is not None:
                boxes = r.obb.xyxyxyxy
                confs = r.obb.conf
                classes = r.obb.cls
                
                for box, conf, cls_idx in zip(boxes, confs, classes):
                    if conf >= self.conf_threshold:
                        points = box.cpu().numpy().reshape((-1, 2))
                        cls_idx = int(cls_idx.item())
                        name = self.model.names[cls_idx]
                        
                        tracked_obj = self.find_matching_object(points, name)
                        if tracked_obj is None:
                            tracked_obj = DetectedObject(points, name, conf)
                            self.tracked_objects.append(tracked_obj)
                        else:
                            tracked_obj.update(points, conf)
                            
                        current_detections.add(tracked_obj)
                        
                        if return_detections:
                            detections_data.append({
                                "points": points.tolist(),
                                "class_name": name,
                                "confidence": float(conf),
                                "is_visible": True
                            })

        # Update visibility and clean up
        for obj in self.tracked_objects:
            if obj not in current_detections:
                obj.check_visibility(self.debounce_time)
                
            if obj.is_visible:
                color = self.get_color_for_class(obj.class_name)
                label = f'{obj.class_name}: {obj.confidence:.2f}'
                self.draw_oriented_bbox(frame_out, obj.points, color, label)

        self.tracked_objects = [obj for obj in self.tracked_objects 
                              if time() - obj.last_seen < self.debounce_time * 5]

        return (frame_out, detections_data) if return_detections else frame_out

    def capture_screen(self) -> np.ndarray:
        """Capture screen content and return as numpy array"""
        hwin = win32gui.GetDesktopWindow()
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)
        
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def virtual_webcam_stream(self, camera_id: int = 0) -> None:
        """Stream YOLO detection to virtual webcam"""
        if not VIRTUAL_CAM_AVAILABLE:
            raise ImportError("pyvirtualcam not installed. Install with: pip install pyvirtualcam")
            
        cap = cv2.VideoCapture(camera_id)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=PixelFormat.BGR, backend='obs') as cam:
            print(f'Using virtual camera: {cam.device}')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                cam.send(processed_frame)
                cam.sleep_until_next_frame()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        cap.release()

    async def send_yolo_response_to_frontend(self, response: Dict[str, Any]):
        """Send YOLO detection response to frontend"""
        async with websockets.connect('ws://localhost:2020/yolo_stream') as websocket:
            await websocket.send(json.dumps(response))

    async def process_video(self, video_file):
        """
        Process a video file with YOLO object detection
        
        Args:
            video_file: Video file object (from FastAPI UploadFile)
            
        Returns:
            List of detection results for each frame
        """
        try:
            # Create a temporary file to save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
                temp_path = temp.name
                # Write the uploaded video to the temporary file
                content = await video_file.read()
                temp.write(content)
            
            # Process the video with OpenCV
            cap = cv2.VideoCapture(temp_path)
            all_detections = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                _, frame_detections = self.process_frame(frame, return_detections=True)
                all_detections.append(frame_detections)
                
            cap.release()
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return all_detections
            
        except Exception as e:
            log.error(f"Error processing video: {e}", exc_info=True)
            raise

    async def detect_api(self, image_data: bytes):
        """Process image bytes with YOLO and return detections
        
        Args:
            image_data: Raw image bytes to process
            
        Returns:
            Tuple of (processed image, detections)
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image data")
            
            # Process image with YOLO
            processed_img, detections = self.process_frame(img, return_detections=True)
            log.info(f"Processed image with {len(detections) if detections else 0} detections")
            
            return processed_img, detections
            
        except Exception as e:
            log.error(f"Error in detect_api: {str(e)}", exc_info=True)
            raise

    async def stream_api(self, websocket: WebSocket):
        """Handle WebSocket streaming of YOLO detections
        
        Args:
            websocket: FastAPI WebSocket connection
        """
        log.info("Starting WebSocket stream handler")
        try:
            await websocket.accept()
            log.info("WebSocket connection accepted")
            
            # Keep the WebSocket connection alive
            while True:
                # Receive image from client
                data = await websocket.receive_bytes()
                log.info(f"Received {len(data)} bytes of image data")
                
                try:
                    # Process with detect_api
                    processed_img, detections = await self.detect_api(data)
                    
                    # Encode processed image to JPEG
                    success, encoded_img = cv2.imencode('.jpg', processed_img)
                    
                    if not success:
                        log.error("Failed to encode processed image")
                        await websocket.send_json({"error": "Failed to encode image"})
                        continue
                    
                    # Send both processed image and detections
                    await websocket.send_bytes(encoded_img.tobytes())
                    await websocket.send_json({"detections": detections})
                    log.info("Sent processed image and detections")
                    
                except Exception as e:
                    log.error(f"Error processing stream frame: {str(e)}", exc_info=True)
                    await websocket.send_json({"error": str(e)})
                
        except Exception as e:
            log.error(f"WebSocket error: {str(e)}", exc_info=True)

    @classmethod
    def get_instance(cls, model_path=None):
        """
        Get or create the YoloProcessor singleton instance.
        This provides backwards compatibility with code that uses this method.
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            YoloProcessor: Singleton instance
        """
        instance = cls()  # The @singleton decorator ensures this returns the existing instance
        
        # If model_path provided, load the model
        if model_path and instance.model is None:
            log.info(f"Loading model from {model_path}")
            instance.load_model(model_path)
            
        return instance


def initialize_yolo(model_path=None, port=8000):
    """Initialize the YOLO processor and API
    
    Args:
        model_path (str, optional): Path to a YOLO model file
        port (int, optional): Port for the API server if starting it
        
    Returns:
        tuple: (YoloProcessor instance, YoloAPI instance)
    """
    log.info("Initializing YOLO processor and API")
    
    # Initialize YoloProcessor with default model if available
    if not model_path:
        # Use Paths utility to get the proper model paths
        paths = Paths()  # Use constructor instead of class method
        default_model_path = paths.get_yolo_default_model_path()
            
        if os.path.exists(default_model_path):
            model_path = default_model_path
            log.info(f"Using default model from {default_model_path}")
        else:
            log.warning(f"Default model not found at {default_model_path}, initializing without model")
            
    # Get or create the YoloProcessor singleton instance
    yolo_processor = YoloProcessor()
    if model_path:
        yolo_processor.load_model(model_path)
    
    # Get or create the YoloAPI singleton instance
    from oarc.yolo.yolo_server_api import YoloServerAPI
    yolo_api = YoloServerAPI()
    
    log.info("YOLO initialization complete")
    return yolo_processor, yolo_api
