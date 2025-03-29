"""Command-line interface for OARC."""
import argparse
from oarc.commands import (
    build_command,
    run_command,
    setup_command
)
from oarc.commands.command_type import CommandType, get_command_type


def cli(**kwargs):
    """Command line interface for OARC."""
    parser = argparse.ArgumentParser(description="OARC command line tool")
    
    # Add global arguments that apply to the default run command
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Path to configuration file')
    
    # Create subparsers for specific commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    subparsers.add_parser('setup', help='Setup dependencies')
    subparsers.add_parser('build', help='Build the OARC package wheel')
    
    args = parser.parse_args(kwargs.get('args', None))
    
    # Convert args to dictionary to pass to command functions
    config = vars(args)
    
    # Get the command type from the command name
    try:
        command_type = get_command_type(args.command)
    except ValueError as e:
        print(str(e))
        return 1
    
    # Remove command from config as it's not needed by the command functions
    if 'command' in config:
        del config['command']
    
    match command_type:
        case CommandType.SETUP:
            return setup_command.execute(**config)
        case CommandType.BUILD:
            return build_command.execute(**config)
        case CommandType.RUN:
            return run_command.execute(**config)
