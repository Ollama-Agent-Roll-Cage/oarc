"""
Setup command module for OARC.

This module handles the 'setup' command which installs all dependencies.
"""

from oarc.utils.log import log
# Import the setup function directly from setup.py instead of through __init__
from oarc.utils.setup.setup import main as setup_main

def execute(**kwargs):
    """Execute the setup command.
    
    Args:
        **kwargs: Command arguments
        
    Returns:
        int: Command exit code
    """
    log.info("Setting up OARC dependencies")
    
    # Pass force parameter from kwargs if present
    force = kwargs.get('force', False)
    if force:
        log.info("Force flag enabled - will reinstall dependencies even if already present")
        
    success = setup_main(force=force)
    
    if success:
        log.info("Setup completed successfully")
        return 0
    else:
        log.error("Setup failed")
        return 1
