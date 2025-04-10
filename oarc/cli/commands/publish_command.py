"""
Publish command module for OARC.

This module handles the 'publish' command which publishes the package to PyPI.
"""

import os
import getpass
from oarc.utils.log import log
from oarc.utils.setup.publish_utils import check_twine_installed, find_wheel_file, publish_to_pypi

def execute(**kwargs):
    """Execute the publish command."""
    log.info("Publishing OARC package")
    
    # Check if Twine is installed
    if not check_twine_installed():
        log.error("Twine is required for publishing. Please install it with 'pip install twine'.")
        return 1
    
    # Extract arguments from kwargs
    repository = kwargs.get('repository', 'pypi')
    username = kwargs.get('username')
    password = kwargs.get('password')
    dist_dir = kwargs.get('dist_dir', 'dist')
    skip_build = kwargs.get('skip_build', False)
    
    # Build the package first if not skipped
    if not skip_build:
        log.info("Building package before publishing...")
        from oarc.cli.commands.build_command import execute as build_execute
        build_result = build_execute(**kwargs)
        if build_result != 0:
            log.error("Build failed. Aborting publish.")
            return 1
    
    # Find the wheel file
    wheel_file = find_wheel_file(dist_dir)
    if not wheel_file:
        log.error(f"No wheel file found in {dist_dir}. Build may have failed.")
        return 1
    
    # Let twine handle authentication from .pypirc or environment variables
    # Only prompt if explicitly requested or if required arguments are provided
    if username or password:
        # Get credentials if explicitly provided
        username = username or os.environ.get("PYPI_USERNAME")
        password = password or os.environ.get("PYPI_PASSWORD")
        
        if not username and kwargs.get('username') is not None:
            username = input("PyPI Username: ")
        
        if not password and kwargs.get('password') is not None:
            password = getpass.getpass("PyPI Password: ")
    else:
        # Just use Twine's default authentication mechanism
        # It will use .pypirc or API tokens automatically
        log.info("Using .pypirc or environment variables for authentication")
    
    # Publish the package
    success = publish_to_pypi(wheel_file, repository, username, password)
    
    if success:
        log.info(f"Package successfully published to {repository}.")
        return 0
    else:
        log.error("Failed to publish package.")
        return 1