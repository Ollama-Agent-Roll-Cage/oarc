#!/usr/bin/env python3
"""
TTS installation utilities for OARC setup process.
"""

import sys
import subprocess
from pathlib import Path

# Import constants from setup_utils
from .setup_utils import PROJECT_ROOT, TTS_REPO_DIR

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
