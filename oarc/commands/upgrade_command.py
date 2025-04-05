"""
OARC Upgrade Command

This module implements the upgrade command for the OARC CLI,
which upgrades and maintains project dependencies.
"""

from oarc.utils.log import log
from oarc.utils.setup.upgrade import main as upgrade_main

def execute(args=None, debug=False, config=None, **kwargs):
    """
    Execute the upgrade command.
    
    Args:
        args: Command line arguments
        debug: Whether to enable debug logging
        config: Configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        int: Exit status code (0 for success, non-zero for failure)
    """
    log.info("Upgrade OARC dependencies")
    
    success = upgrade_main()
    
    if success:
        log.info("Upgrade completed successfully")
        return 0
    else:
        log.error("Upgrade failed - see error messages above for details")
        return 1
