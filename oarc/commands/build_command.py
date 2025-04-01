"""
Build command module for OARC.

This module handles the 'build' command which builds the package.
"""

from oarc.utils.log import log
# Import build utilities directly from the module, not through __init__
from oarc.utils.setup.build_utils import build_package

def execute(**kwargs):
    """Execute the build command."""
    log.info("Building OARC package")
    
    # Call the build function
    build_package()
    
    log.info("Build completed successfully")
    return 0
