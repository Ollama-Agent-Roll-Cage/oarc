"""Command type enum for OARC CLI.

This module defines the different command types available in the OARC CLI.
"""

from enum import Enum, auto


class CommandType(Enum):
    """Enumeration of command types supported by the OARC CLI.
    
    This enum defines the various commands that can be executed in the OARC CLI.
    Each command corresponds to a specific functionality within the CLI.
    """
    
    SETUP = "setup"
    BUILD = "build"
    RUN = "run"  # Default command when none is specified
    HELP = "help"


def get_command_type(command_name: str) -> CommandType:
    """Get the command type from the command name string.
    
    Args:
        command_name: The name of the command as a string
        
    Returns:
        CommandType: The corresponding command type enum value
        
    Raises:
        ValueError: If the command name is invalid
    """
    if command_name is None:
        return CommandType.HELP  # Changed default from RUN to HELP
    elif command_name == "-h" or command_name == "--help":
        return CommandType.HELP
    
    try:
        return CommandType(command_name)
    except ValueError:
        raise ValueError(f"Unknown command: {command_name}")
