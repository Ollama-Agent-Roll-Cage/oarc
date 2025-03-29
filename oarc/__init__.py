"""
OARC package initialization.

This module initializes the OARC package and imports necessary components.
"""

__version__ = "0.1.0"
__author__ = "OARC Team"

from oarc import cli
from oarc import main
from oarc import setup
from oarc.utils.setup import setup_utils, tts_utils, cuda_utils, pyaudio_utils

__all__ = ['cli', 'main', 'setup', 'utils']

def version():
    """Return the current version of OARC."""
    return __version__
