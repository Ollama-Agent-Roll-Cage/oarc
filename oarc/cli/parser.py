import argparse
from oarc.cli.help import MAIN_HELP

def parse_cli_args(args=None):
    parser = argparse.ArgumentParser(
        description="OARC command line tool",
        epilog=MAIN_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Global options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Path to configuration file')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup dependencies')
    setup_parser.add_argument('--force', action='store_true', help='Force reinstallation of dependencies')
    
    # Build command
    subparsers.add_parser('build', help='Build the OARC package wheel')
    
    # Publish command
    publish_parser = subparsers.add_parser('publish', help='Publish package to PyPI')
    publish_parser.add_argument('--repository', default='pypi', help='Repository to publish to')
    publish_parser.add_argument('--username', help='PyPI username')
    publish_parser.add_argument('--password', help='PyPI password')
    publish_parser.add_argument('--dist-dir', default='dist', help='Directory containing distribution files')
    publish_parser.add_argument('--skip-build', action='store_true', help='Skip building the package before publishing')
    
    # Help command
    subparsers.add_parser('help', help='Show this help message')
    
    return parser.parse_args(args)
