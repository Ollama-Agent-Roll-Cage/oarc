"""
This module defines the DetectedObject class, which represents objects detected
in a YOLO object detection system. It encapsulates properties such as bounding
points, class name, confidence score, and methods for updating and checking
visibility status.
"""

from time import time
import numpy as np


class DetectedObject:
    """
    Represents a detected object with its bounding points, class name, confidence score,
    and visibility status. Tracks the last time the object was seen for debouncing purposes.
    """

    def __init__(self, points: np.ndarray, class_name: str, confidence: float):
        """
        Initializes a DetectedObject instance.

        Args:
            points (np.ndarray): The bounding points of the detected object.
            class_name (str): The class name of the detected object.
            confidence (float): The confidence score of the detection.
        """
        self.points = points
        self.class_name = class_name
        self.confidence = confidence
        self.last_seen = time()  # Timestamp of when the object was last seen.
        self.is_visible = True  # Visibility status of the object.

    def update(self, points: np.ndarray, confidence: float):
        """
        Updates the detected object's bounding points, confidence score, and visibility status.

        Args:
            points (np.ndarray): The updated bounding points of the object.
            confidence (float): The updated confidence score of the detection.
        """
        self.points = points
        self.confidence = confidence
        self.last_seen = time()  # Update the timestamp to the current time.
        self.is_visible = True  # Mark the object as visible.

    def check_visibility(self, debounce_time: float) -> bool:
        """
        Checks if the object is still visible based on the debounce time.

        Args:
            debounce_time (float): The time threshold (in seconds) to determine visibility.

        Returns:
            bool: True if the object is visible, False otherwise.
        """
        if self.is_visible and time() - self.last_seen > debounce_time:
            self.is_visible = False  # Mark the object as not visible if debounce time has passed.
        return self.is_visible