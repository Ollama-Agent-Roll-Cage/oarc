#!/usr/bin/env python3
"""
General setup utilities for OARC package.
"""

import sys
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def update_pip(venv_python=None):
    """Update pip to the latest version.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
    
    Returns:
        bool: True if update was successful
    
    Raises:
        subprocess.CalledProcessError: If pip update fails
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    log.info(f"Updating pip using Python from: {venv_python}")
    
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True
    )
    log.info("Pip has been successfully updated to the latest version.")
    return True


def ensure_pip_subprocess(venv_python=None):
    """Ensure pip subprocess is available.
    
    This checks if pip can be run as a subprocess, which is required for
    installing packages programmatically.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                     If None, use the current Python interpreter.
    
    Returns:
        bool: True if pip subprocess is available
    
    Raises:
        subprocess.CalledProcessError: If pip command fails
        FileNotFoundError: If pip executable is not found
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    log.info("Checking pip subprocess functionality...")
    
    result = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    log.info(f"Pip is working correctly: {result.stdout.strip()}")
    return True


def ensure_pip(venv_python=None):
    """Update pip and verify it works as a subprocess.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
    
    Returns:
        bool: True if setup was successful
    
    Raises:
        subprocess.CalledProcessError: If pip commands fail
        FileNotFoundError: If pip executable is not found
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
        
    log.info("Setting up pip...")
    update_pip(venv_python)
    ensure_pip_subprocess(venv_python)
    
    log.info("Pip is set up correctly and ready to use.")
    return True
