"""Utility packages for OARC.

This package contains various utility modules and subpackages 
used throughout the OARC project.
"""

from .log import get_logger, log
from .paths import Paths
from .speech_utils import SpeechUtils
from oarc.utils import setup
from oarc.utils.const import (
    SUCCESS,
    FAILURE,
)

__all__ = [
    # Functions
    'get_logger',
    'log',
    'Paths',
    'SpeechUtils',
    'setup',

    # Constants
    'SUCCESS',
    'FAILURE',
]
