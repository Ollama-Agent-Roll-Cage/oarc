"""
OARC Publish Command Module
This module implements the `publish` command for the OARC CLI.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Import the publish functionality from the publish script
# Assuming this file will be placed in the oarc/cli/commands directory
from oarc.utils.publish_utils import check_twine_installed, find_wheel_file, publish_to_pypi

logger = logging.getLogger("oarc.cli.commands.publish_command")

def add_publish_parser(subparsers):
    """Add the publish command parser to the given subparsers."""
    parser = subparsers.add_parser(
        "publish", 
        help="Publish OARC package to PyPI"
    )
    parser.add_argument(
        "--repository", 
        "-r", 
        default="pypi",
        help="Repository to publish to (default: pypi)"
    )
    parser.add_argument(
        "--username", 
        "-u", 
        help="PyPI username (will use PYPI_USERNAME env var if not provided)"
    )
    parser.add_argument(
        "--password", 
        "-p", 
        help="PyPI password (will use PYPI_PASSWORD env var if not provided or prompt if neither is available)"
    )
    parser.add_argument(
        "--dist-dir", 
        default="dist",
        help="Directory containing distribution files (default: dist)"
    )
    parser.add_argument(
        "--skip-build", 
        action="store_true",
        help="Skip building the package before publishing"
    )
    return parser

def execute_publish_command(args):
    """Execute the publish command with the given arguments."""
    logger.info("Publishing OARC package")
    
    # Check if Twine is installed
    if not check_twine_installed():
        logger.error("Twine is required for publishing. Please install it with 'pip install twine'.")
        return False
    
    # Build the package first if not skipped
    if not args.skip_build:
        logger.info("Building package before publishing...")
        # Import here to avoid circular imports
        from oarc.cli.commands.build_command import execute_build_command
        build_success = execute_build_command(args)
        if not build_success:
            logger.error("Build failed. Aborting publish.")
            return False
    
    # Find the wheel file
    wheel_file = find_wheel_file(args.dist_dir)
    if not wheel_file:
        logger.error(f"No wheel file found in {args.dist_dir}. Build may have failed.")
        return False
    
    # Get credentials
    username = args.username or os.environ.get("PYPI_USERNAME")
    password = args.password or os.environ.get("PYPI_PASSWORD")
    
    if not username:
        username = input("PyPI Username: ")
    
    if not password:
        import getpass
        password = getpass.getpass("PyPI Password: ")
    
    # Publish the package
    success = publish_to_pypi(wheel_file, args.repository, username, password)
    
    if success:
        logger.info(f"Package successfully published to {args.repository}.")
        return True
    else:
        logger.error("Failed to publish package.")
        return False
