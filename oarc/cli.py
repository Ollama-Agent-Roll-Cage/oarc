"""Command-line interface for OARC."""
import argparse
import logging
from oarc.commands import (
    build_command,
    run_command,
    setup_command
)
from oarc.commands.command_type import CommandType, get_command_type


# Create a proper logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def cli(**kwargs):
    """Command line interface for OARC."""
    log.info("Starting OARC CLI")
    parser = argparse.ArgumentParser(description="OARC command line tool")
    
    # Add global arguments that apply to the default run command
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Path to configuration file')
    
    # Create subparsers for specific commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    subparsers.add_parser('setup', help='Setup dependencies')
    subparsers.add_parser('build', help='Build the OARC package wheel')
    
    args = parser.parse_args(kwargs.get('args', None))
    log.info(f"Parsed arguments: {args}")
    
    # Convert args to dictionary to pass to command functions
    config = vars(args)
    
    # Get the command type from the command name
    try:
        command_type = get_command_type(args.command)
        log.info(f"Executing command type: {command_type}")
    except ValueError as e:
        log.error(f"Invalid command: {e}")
        return 1
    
    if 'command' in config:
        del config['command']
    
    match command_type:
        case CommandType.SETUP:
            log.info("Running setup command")
            return setup_command.execute(**config)
        case CommandType.BUILD:
            log.info("Running build command") 
            return build_command.execute(**config)
        case CommandType.RUN:
            log.info("Running default command")
            return run_command.execute(**config)
