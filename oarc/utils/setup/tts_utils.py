#!/usr/bin/env python3
"""
TTS installation utilities for OARC setup process.
"""

import subprocess
import sys
from pathlib import Path

TTS_REPO_URL = "https://github.com/idiap/coqui-ai-TTS"
TTS_REPO_NAME = "coqui-ai-TTS"

def install_coqui(venv_python):
    """Install TTS directly from the GitHub repository.
    
    Args:
        venv_python: Path to Python executable in virtual environment
        repo_dir: Path to clone the repo to (required)
        repo_url: GitHub repo URL (required)
    
    Returns:
        bool: True if installation was successful
    
    Raises:
        SystemExit: If required parameters are not provided
    """
    # Fix: Use the project root directory instead of the package directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_dir = project_root / TTS_REPO_NAME
    
    # Validate required parameters
    if repo_dir is None:
        print("Error: repo_dir is required")
        sys.exit(1)
        
    if TTS_REPO_URL is None:
        print("Error: repo_url is required")
        sys.exit(1)
    
    print(f"Installing TTS from GitHub repository to {repo_dir}...")
    
    # Convert to Path object if string is provided
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)
    
    # Check if coqui directory already exists
    if repo_dir.exists():
        print(f"Found existing TTS repository at {repo_dir}")
        # Update the repository
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=str(repo_dir),
                check=True
            )
            print("Updated TTS repository")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to update TTS repository: {e}")
    else:
        # Create directory and clone repository
        repo_dir.parent.mkdir(exist_ok=True, parents=True)
        print(f"Cloning TTS repository from {TTS_REPO_URL}...")
        try:
            subprocess.run(
                ["git", "clone", TTS_REPO_URL, str(repo_dir)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning TTS repository: {e}")
            return False
    
    # Install in development mode
    print("Installing TTS in development mode...")
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
            check=True
        )
        print("TTS installed successfully from GitHub!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing TTS: {e}")
        print("TTS installation failed.")
        return False
