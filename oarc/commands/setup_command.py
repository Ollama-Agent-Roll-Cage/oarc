"""
Setup command module for OARC.

This module handles the 'setup' command which installs all dependencies.
"""

from oarc.utils.log import log
# Import the setup function directly from setup.py instead of through __init__
from oarc.utils.setup.setup import main as setup_main

def execute(**kwargs):
    """Execute the setup command."""
    log.info("Setting up OARC dependencies")
    
    # Call the setup main function
    setup_main()
    
    log.info("Setup completed successfully")
    return 0
