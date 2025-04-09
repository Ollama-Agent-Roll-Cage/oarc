"""Command-line interface router for OARC."""

from oarc.cli.parser import parse_cli_args
from oarc.cli.commands.command_type import CommandType, get_command_type
from oarc.cli.help import MAIN_HELP

def handle(**kwargs):
    """Command line interface for OARC."""
    args = parse_cli_args(kwargs.get('args', None))
    config = vars(args)
    command_type = get_command_type(args.command)

    # Handle help command explicitly
    if command_type == CommandType.HELP:
        print(MAIN_HELP)
        return 0

    # Skip logging for help command
    if command_type != CommandType.HELP:
        from oarc.utils.log import log
        log.info("Starting OARC CLI")
        log.info(f"Parsed arguments: {args}")
        log.info(f"Executing command type: {command_type}")

        from oarc.cli.commands import build_command, run_command, setup_command

        if 'command' in config:
            del config['command']

        match command_type:
            case CommandType.SETUP:
                return setup_command.execute(**config) 
            case CommandType.BUILD:
                return build_command.execute(**config)
            case CommandType.RUN:
                return run_command.execute(**config)

    return 0
