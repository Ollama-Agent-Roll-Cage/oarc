#!/usr/bin/env python3
"""
General setup utilities for OARC package.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util
import platform

from oarc.utils.log import log


def install_uv(venv_python=None):
    """Install UV package manager if not already installed.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
    
    Returns:
        bool: True if installation was successful or UV is already installed
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    # Check if UV is already installed
    if importlib.util.find_spec("uv") is not None:
        log.info("UV is already installed.")
        return True
    
    try:
        # First ensure pip is up to date
        log.info("Updating pip before installing UV...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True  # Use text mode for consistent string handling
        )
        
        # Install UV using pip
        log.info("Installing UV package manager...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "uv"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True  # Use text mode for consistent string handling
        )
        
        log.info("UV has been successfully installed.")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install UV: {e}")
        log.debug(f"Stderr: {e.stderr if e.stderr else 'No error output'}")
        return False


def update_pip(venv_python=None):
    """Update pip to the latest version.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
    
    Returns:
        bool: True if update was successful
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    log.info(f"Updating pip using Python from: {venv_python}")
    
    try:
        # Try to use UV first if it's installed
        if importlib.util.find_spec("uv") is not None:
            subprocess.run(
                [str(venv_python), "-m", "uv", "pip", "install", "--upgrade", "pip"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True  # Use text mode for consistent string handling
            )
        else:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True  # Use text mode for consistent string handling
            )
        log.info("Pip has been successfully updated to the latest version.")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to update pip: {e}")
        log.debug(f"Stderr: {e.stderr if e.stderr else 'No error output'}")
        return False


def ensure_pip_subprocess(venv_python=None):
    """Ensure pip subprocess is available.
    
    This checks if pip can be run as a subprocess, which is required for
    installing packages programmatically.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                     If None, use the current Python interpreter.
    
    Returns:
        bool: True if pip subprocess is available
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    log.info("Checking pip subprocess functionality...")
    
    try:
        # Check UV first if it's installed
        if importlib.util.find_spec("uv") is not None:
            result = subprocess.run(
                [str(venv_python), "-m", "uv", "pip", "--version"],
                capture_output=True, text=True, check=True
            )
            log.info(f"UV pip is working correctly: {result.stdout.strip()}")
        else:
            result = subprocess.run(
                [str(venv_python), "-m", "pip", "--version"],
                capture_output=True, text=True, check=True
            )
            log.info(f"Pip is working correctly: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to verify pip functionality: {e}")
        # Remove .decode() since we're using text=True
        log.debug(f"Stderr: {e.stderr if e.stderr else 'No error output'}")
        return False


def install_package(package_name, venv_python=None, options=None):
    """Install a package using UV if available, falling back to pip.
    
    Args:
        package_name: Name of the package to install
        venv_python: Path to Python executable in virtual environment
        options: Additional options to pass to the installer
        
    Returns:
        bool: True if installation was successful
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    if options is None:
        options = []
    
    log.info(f"Installing {package_name}...")
    
    try:
        # Try to use UV first if it's installed
        if importlib.util.find_spec("uv") is not None:
            cmd = [str(venv_python), "-m", "uv", "pip", "install", package_name] + options
            log.info(f"Using UV to install {package_name}")
        else:
            cmd = [str(venv_python), "-m", "pip", "install", package_name] + options
            log.info(f"Using pip to install {package_name}")
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install {package_name}: {e}")
        log.debug(f"Stderr: {e.stderr if e.stderr else 'No error output'}")
        return False


def install_numpy(venv_python=None, version="2.1.1"):
    """Install a specific version of numpy.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
        version: Version of numpy to install (defaults to 2.1.1).
    
    Returns:
        bool: True if installation was successful
    """
    package_spec = f"numpy=={version}"
    return install_package(package_spec, venv_python, ["--force-reinstall"])


def ensure_pip(venv_python=None):
    """Update pip and verify it works as a subprocess.
    Also installs UV package manager if not already installed.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
    
    Returns:
        bool: True if setup was successful
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
        
    log.info("Setting up package managers...")
    
    # Add fallback mechanism when pip has issues
    try:
        # First check if direct execution works
        try:
            # Direct test with simple command
            subprocess.run(
                [str(venv_python), "-c", "import sys; print(sys.executable)"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except subprocess.CalledProcessError:
            log.error("Python interpreter cannot be executed. Check your virtual environment.")
            return False
            
        # Try to use ensurepip as a fallback if pip is broken
        try:
            subprocess.run(
                [str(venv_python), "-m", "ensurepip", "--upgrade"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            log.info("Used ensurepip to bootstrap pip installation")
        except subprocess.CalledProcessError:
            log.warning("ensurepip failed, continuing anyway")
            
        # First install UV if not already available
        uv_success = install_uv(venv_python)
        
        # Then update pip (using UV if available)
        pip_success = update_pip(venv_python)
        
        # Verify everything works
        verify_success = ensure_pip_subprocess(venv_python)
        
        if uv_success and pip_success and verify_success:
            log.info("Package managers are set up correctly and ready to use.")
            return True
        else:
            log.warning("Package managers setup had some issues. Will try to continue anyway.")
            return True  # Return True to continue installation despite issues
    except Exception as e:
        log.error(f"Unexpected error during package manager setup: {e}")
        log.warning("Attempting to continue with installation despite package manager issues")
        return True  # Return True to continue installation anyway

