"""
File: yoloProcessor.py
Description:
    Implements a YOLO-based object detection processor with oriented bounding boxes.
    Provides functionality for:
      - Loading and using a YOLO model for detection.
      - Tracking objects with debouncing and drawing bounding boxes.
      - Capturing screen content.
      - Streaming processed frames to a virtual webcam.
      - Exposing a FastAPI router for detection and streaming endpoints.
      
Usage:
    Run this script directly to perform detections on a sample image, capture the screen,
    or stream via a virtual webcam.
    
Dependencies:
    - OpenCV (cv2)
    - NumPy
    - torch
    - ultralytics
    - pyvirtualcam (optional, for virtual webcam streaming)
    - win32gui, win32ui, win32con, win32api (for screen capture on Windows)
    - websockets, fastapi (for API integration)
"""

import os
import json
from time import time
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import win32gui
import win32con
import win32ui
import win32api
import websockets
from fastapi import APIRouter, UploadFile, WebSocket

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False

class DetectedObject:
    def __init__(self, points: np.ndarray, class_name: str, confidence: float):
        self.points = points
        self.class_name = class_name
        self.confidence = confidence
        self.last_seen = time()
        self.is_visible = True

    def update(self, points: np.ndarray, confidence: float):
        self.points = points
        self.confidence = confidence
        self.last_seen = time()
        self.is_visible = True

    def check_visibility(self, debounce_time: float) -> bool:
        if self.is_visible and time() - self.last_seen > debounce_time:
            self.is_visible = False
        return self.is_visible

class YoloProcessor:
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 conf_threshold: float = 0.5,
                 debounce_ms: float = 200):
        """
        Initialize YoloProcessor with configurable settings
        
        Args:
            model_path: Path to YOLO model file (.pt)
            conf_threshold: Confidence threshold for detections (0-1)
            debounce_ms: Debounce time in milliseconds for object tracking
        """
        self.model = None
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.debounce_time = debounce_ms / 1000.0
        self.tracked_objects: List[DetectedObject] = []
        
        # Color settings for visualization
        self.color_map = {}
        self.color_index = 0
        self.neon_colors = [
            (0, 255, 255),    # Neon Yellow
            (255, 0, 255),    # Neon Pink
            (0, 255, 0),      # Neon Green
            (255, 128, 0),    # Neon Blue
            (0, 128, 255),    # Neon Orange
            (255, 0, 128),    # Neon Purple
            (0, 255, 128),    # Neon Turquoise
            (128, 255, 0),    # Neon Lime
        ]

        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        if model_path:
            self.load_model(model_path)

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


class YoloAPI:
    def __init__(self):
        self.router = APIRouter()
        self.setup_routes()
    
    def setup_routes(self):
        @self.router.post("/detect")
        async def detect_objects(self, image: UploadFile):
            pass
            
        @self.router.websocket("/stream")
        async def vision_stream(self, websocket: WebSocket):
            pass

if __name__ == "__main__":
    # Basic usage with oriented bounding boxes
    detector = YoloProcessor(model_path="path/to/model.pt", conf_threshold=0.7, debounce_ms=200)

    # Process a single frame
    frame = cv2.imread("image.jpg")
    processed_frame = detector.process_frame(frame)

    # Get frame with detections data
    processed_frame, detections = detector.process_frame(frame, return_detections=True)

    # Use virtual webcam feature
    detector.virtual_webcam_stream()

    # Process screen capture
    screen = detector.capture_screen()
    processed_screen = detector.process_frame(screen)