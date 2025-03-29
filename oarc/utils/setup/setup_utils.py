#!/usr/bin/env python3
"""
Utility functions for OARC setup and installation process.
"""

import sys
import subprocess
import shutil
import configparser
from pathlib import Path
import os

# Define project constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
CONFIG_DIR = PROJECT_ROOT / "oarc" / "config_files"
LOG_DIR = PROJECT_ROOT / "logs"
TTS_REPO_DIR = PROJECT_ROOT / "coqui"

def read_build_config():
    """Read build configuration from .ini file."""
    config = configparser.ConfigParser()
    build_config_path = CONFIG_DIR / "build.ini"
    
    if build_config_path.exists():
        config.read(build_config_path)
        return config
    else:
        raise ValueError(f"Build configuration file not found: {build_config_path}")

def clean_egg_info():
    """Clean up existing egg-info directories."""
    print("Cleaning up existing egg-info directories...")
    for egg_info in PROJECT_ROOT.glob("*.egg-info"):
        if egg_info.is_dir():
            print(f"Removing {egg_info}")
            shutil.rmtree(egg_info)
    
    # Also check inside the oarc directory for egg_info
    oarc_dir = PROJECT_ROOT / "oarc"
    if oarc_dir.exists():
        for egg_info in oarc_dir.glob("*.egg-info"):
            if egg_info.is_dir():
                print(f"Removing {egg_info}")
                shutil.rmtree(egg_info)

def fix_egg_deprecation(venv_python):
    """Fix egg format deprecation warnings by reinstalling problematic packages."""
    print("Checking for packages installed in deprecated egg format...")
    
    # Common packages that often get installed as eggs and cause warnings
    problem_packages = ["uvicorn", "websockets", "whisper"]
    
    # Get site-packages directory
    result = subprocess.run(
        [str(venv_python), "-c", "import site; print(site.getsitepackages()[0])"],
        check=True,
        capture_output=True,
        text=True
    )
    site_packages_dir = Path(result.stdout.strip())
    
    # Look for .egg directories or .egg-info files
    egg_dirs = list(site_packages_dir.glob("*.egg"))
    
    if egg_dirs:
        print(f"Found {len(egg_dirs)} packages installed in egg format. Fixing...")
        for egg_dir in egg_dirs:
            package_name = egg_dir.stem.split("-")[0]  # Extract package name from egg directory
            print(f"Reinstalling {package_name} to fix egg format...")
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--force-reinstall", package_name],
                check=True
            )
    else:
        print("No packages with egg format detected.")
        
    # Explicit reinstall of common problematic packages
    print("Reinstalling known problematic packages...")
    for package in problem_packages:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--force-reinstall", package],
            check=True
        )
    
    # Ensure numpy is at the right version
    print("Ensuring numpy is at the correct version...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "numpy>=1.24.3", "--force-reinstall"],
        check=True
    )

def get_directory_size(directory_path):
    """Calculate the total size of a directory and its contents in bytes."""
    total_size = 0
    try:
        for path in Path(directory_path).rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
    except (PermissionError, FileNotFoundError) as e:
        print(f"Warning: Couldn't access some files while calculating directory size: {e}")
    return total_size

def format_size(size_bytes):
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
