"""OARC package initialization."""

__version__ = "0.1.1"
__author__ = "OARC Team"

# Remove the circular import
# from oarc import cli  <- This line causes the circular import

__all__ = []  # Empty for now, we don't need to expose cli here

def version():
    """Return the current version of OARC."""
    return __version__
