#!/usr/bin/env python3
"""
Virtual environment utilities for OARC setup process.
"""

import os
import sys
import subprocess
from pathlib import Path

from .setup_utils import PROJECT_ROOT, VENV_DIR

def check_python_version(python_path, required_version=(3, 8)):
    """Check if the Python interpreter meets the version requirements."""
    try:
        # Use a single subprocess call to get the Python version
        version_output = subprocess.check_output(
            [str(python_path), "--version"],
            universal_newlines=True,
            stderr=subprocess.STDOUT
        ).strip()
        
        # Parse version with regex
        import re
        version_match = re.search(r"Python (\d+)\.(\d+)\.(\d+)", version_output)
        
        if not version_match:
            print(f"Warning: Could not parse Python version from: {version_output}")
            return False
        
        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        patch = int(version_match.group(3))
        
        print(f"Using Python version: {major}.{minor}.{patch} from interpreter: {python_path}")
        print(f"Python interpreter: {python_path}")

        # Compare version components
        current_version = (major, minor)
        return current_version >= required_version
        
    except Exception as e:
        print(f"Error checking Python version for {python_path}: {e}")
        return False

def detect_existing_venv():
    """Detect existing virtual environments in the project directory."""
    venv_dirs = [VENV_DIR]
    for venv_dir in venv_dirs:
        if not venv_dir.exists():
            continue
        if os.name == 'nt':
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
        if python_exe.exists() and check_python_version(python_exe):
            return venv_dir
    return None

def get_venv_python():
    """Get or create a Python executable in a virtual environment."""
    # Check if already in a virtual environment
    if sys.prefix != sys.base_prefix:
        print(f"Already in virtual environment: {sys.prefix}")
        return Path(sys.executable)
    
    # Check for existing virtual environment
    existing_venv = detect_existing_venv()
    if existing_venv:
        print(f"Using existing virtual environment at {existing_venv}")
        if os.name == 'nt':
            venv_python = existing_venv / "Scripts" / "python.exe"
        else:
            venv_python = existing_venv / "bin" / "python"
        return venv_python
    
    # Create a new virtual environment
    if os.name == 'nt':
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"
    
    # Use current Python to create venv
    print(f"Creating virtual environment at {VENV_DIR}...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    print("Virtual environment created successfully.")
    
    return venv_python

def ensure_pip(venv_python):
    """Ensure pip is installed before upgrading."""
    print("Ensuring pip is installed...")
    try:
        subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: ensurepip returned an error: {e}")
        print("Continuing with existing pip...")
