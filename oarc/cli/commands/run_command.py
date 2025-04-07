"""Run command implementation for OARC CLI.

This module provides the implementation of the run command for the OARC CLI.
"""

from oarc.app import app


def execute(**kwargs):
    """Execute the run command.
    
    This is the default command that runs the OARC application.
    
    Args:
        **kwargs: Command-specific arguments including:
            debug (bool): Whether to enable debug mode
            config (str): Path to configuration file
        
    Returns:
        dict: Results from the OARC application
    """
    result = app.main(**kwargs)
    print(f"OARC execution completed: {result}")
    return result
