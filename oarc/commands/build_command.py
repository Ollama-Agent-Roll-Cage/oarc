"""Build command implementation for OARC CLI.

This module provides the implementation of the build command for the OARC CLI.
"""

from oarc.utils.setup import build_utils


def execute(**kwargs):
    """Execute the build command.
    
    This command builds the OARC package wheel.
    
    Args:
        **kwargs: Command-specific arguments
        
    Returns:
        Any: Result of the build operation
    """
    print("Building OARC package...")
    result = build_utils.build_package()
    print("Build completed successfully!")
    return result
