"""
OARC package initialization.

This module initializes the OARC package and imports necessary components.
"""

__version__ = "0.1.0"
__author__ = "OARC Team"

from oarc import app
from oarc import cli
from oarc import main
from oarc import server
from oarc import utils
from oarc import yolo

__all__ = [
    'app',
    'cli', 
    'main', 
    'setup', 
    'utils',
    'server',
    'yolo',
]

def version():
    """Return the current version of OARC."""
    return __version__
