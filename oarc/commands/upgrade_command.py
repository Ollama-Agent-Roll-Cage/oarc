"""
Upgrade command module for OARC.

This module handles the 'upgrade' command which upgrades all dependencies.
"""

from oarc.utils.log import log
# Import the setup function directly from setup.py instead of through __init__
from oarc.utils.setup.upgrade import main as upgrade_main

def execute(**kwargs):
    """Execute the upgrade command."""
    log.info("Ugrade up OARC dependencies")
    
    # Call the setup main function
    upgrade_main()
    
    log.info("Upgrade completed successfully")
    return 0
