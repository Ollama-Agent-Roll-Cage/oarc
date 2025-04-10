#!/usr/bin/env python3
"""
Publish utilities for OARC package.
"""

import os
import subprocess
from pathlib import Path

from oarc.utils.log import log

def check_twine_installed():
    """Check if Twine is installed."""
    try:
        subprocess.run(["twine", "--version"], capture_output=True, check=True)
        log.info("Twine is installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.error("Twine is not installed. Please install it with 'pip install twine'.")
        return False

def find_wheel_file(dist_dir="dist"):
    """Find the wheel file in the dist directory."""
    # Get the project root directory (assuming this file is in oarc/utils/setup/)
    project_dir = Path(__file__).resolve().parents[3]
    dist_path = project_dir / dist_dir
    
    if not dist_path.exists():
        log.error(f"Distribution directory {dist_dir} does not exist.")
        return None
    
    wheel_files = list(dist_path.glob("*.whl"))
    if not wheel_files:
        log.error(f"No wheel files found in {dist_dir} directory.")
        return None
    
    # Return the most recently created wheel file
    latest_wheel = max(wheel_files, key=os.path.getctime)
    log.info(f"Found wheel file: {latest_wheel}")
    return latest_wheel

def publish_to_pypi(wheel_file, repository="pypi", username=None, password=None):
    """Publish the wheel file to PyPI using Twine."""
    log.info(f"Publishing {wheel_file} to {repository}...")
    
    cmd = ["twine", "upload"]
    
    # Add repository option if not uploading to default PyPI
    if repository.lower() != "pypi":
        cmd.extend(["--repository", repository])
    
    # Add credentials if provided
    if username:
        cmd.extend(["--username", username])
    if password:
        cmd.extend(["--password", password])
    
    # Add the wheel file
    cmd.append(str(wheel_file))
    
    # Create a safe version of the command for logging (mask password)
    safe_cmd = []
    for i, arg in enumerate(cmd):
        if i > 0 and cmd[i-1] == "--password":
            safe_cmd.append("******")
        else:
            safe_cmd.append(arg)
    
    log.info(f"Running command: {' '.join(safe_cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            log.info("Package successfully published to PyPI.")
            log.info(result.stdout)
            return True
        else:
            log.error(f"Failed to publish package. Error: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"An error occurred during publishing: {str(e)}")
        return False
