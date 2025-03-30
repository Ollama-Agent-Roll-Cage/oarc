""" This module contains the YoloProcessor and YoloAPI classes, which are used for"""

from .yoloProcessor import YoloAPI
from .yoloProcessor import YoloProcessor
from .yoloProcessor import DetectedObject

__all__ = [
    'YoloProcessor', 
    'YoloAPI',
    'DetectedObject'
]