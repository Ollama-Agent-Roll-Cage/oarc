#!/usr/bin/env python3
"""
General setup utilities for OARC package.
"""

import subprocess
import sys
from pathlib import Path

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
    
    print(f"Updating pip using Python from: {venv_python}")
    
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True
    )
    print("Pip has been successfully updated to the latest version.")
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
    
    print("Checking pip subprocess functionality...")
    
    result = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"Pip is working correctly: {result.stdout.strip()}")
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
        
    print("Setting up pip...")
    update_pip(venv_python)
    ensure_pip_subprocess(venv_python)
    
    print("Pip is set up correctly and ready to use.")
    return True

def install_self(venv_python=None, editable=True):
    """Install the package in development mode.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
        editable: Whether to install in editable mode (-e flag)
    
    Returns:
        bool: True if installation was successful
    
    Raises:
        subprocess.CalledProcessError: If installation fails
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    print(f"Installing package using Python from: {venv_python}")
    
    cmd = [str(venv_python), "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(".")
    
    subprocess.run(cmd, check=True)
    print("Package has been successfully installed in development mode.")
    return True

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    ensure_pip()
