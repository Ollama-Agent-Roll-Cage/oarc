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
        [str(venv_python), "-m", "pip", "install", "numpy>=1.19.0,<2.0.0", "--force-reinstall"],
        check=True
    )

def install_pyaudio_dependencies(venv_python):
    """Install platform-specific dependencies for PyAudio."""
    print("Installing PyAudio dependencies...")
    
    try:
        if os.name == 'nt':  # Windows
            print("Installing PyAudio directly...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        
        elif sys.platform == 'darwin':  # macOS
            print("Installing PortAudio dependencies for macOS...")
            try:
                subprocess.run(["brew", "install", "portaudio"], check=True)
            except:
                print("Warning: Could not install portaudio with brew. If PyAudio fails to install, install portaudio manually.")
            
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        
        else:  # Linux and others
            print("Installing PortAudio dependencies for Linux...")
            try:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "portaudio19-dev"], check=True)
            except:
                print("Warning: Could not install portaudio with apt. If PyAudio fails to install, install portaudio19-dev manually.")
                
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
            
        print("PyAudio installed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install PyAudio: {e}")
        print("Continuing without PyAudio. Some audio features may not work.")
        return False

def install_tts_from_github(venv_python):
    """Install TTS directly from the GitHub repository."""
    print("Installing TTS from GitHub repository...")
    
    # Check if coqui directory already exists
    if TTS_REPO_DIR.exists():
        print(f"Found existing TTS repository at {TTS_REPO_DIR}")
    else:
        # Create directory and clone repository
        TTS_REPO_DIR.mkdir(exist_ok=True, parents=True)
        print("Cloning TTS repository from GitHub...")
        subprocess.run(
            ["git", "clone", "https://github.com/idiap/coqui-ai-TTS", str(TTS_REPO_DIR)],
            check=True
        )
    
    # Install in development mode
    print("Installing TTS in development mode...")
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(TTS_REPO_DIR)],
            check=True
        )
        print("TTS installed successfully from GitHub!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing TTS: {e}")
        print("TTS installation failed. Exiting.")
        sys.exit(1)  # Exit immediately on failure

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
