"""OARC package initialization."""

__version__ = "0.1.0"
__author__ = "OARC Team"

from oarc import cli

__all__ = ['cli']

def version():
    """Return the current version of OARC."""
    return __version__
