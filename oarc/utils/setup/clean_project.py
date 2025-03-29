#!/usr/bin/env python3
"""
Project cleaning utilities for OARC setup process.
"""

import shutil
from pathlib import Path

from .setup_utils import PROJECT_ROOT, TTS_REPO_DIR, clean_egg_info

def clean_project():
    """Clean up build artifacts from the project directory."""
    print("Cleaning up build artifacts...")
    
    # Clean egg-info directories
    clean_egg_info()
    
    # Clean TTS repository if it exists
    if TTS_REPO_DIR.exists():
        print(f"Removing TTS repository at {TTS_REPO_DIR}")
        try:
            shutil.rmtree(TTS_REPO_DIR)
        except PermissionError as e:
            print(f"Warning: Could not fully remove TTS repository due to permission error: {e}")
            print("Some files may remain. You might need to manually delete them.")
    
    # Remove build directory
    build_dir = PROJECT_ROOT / "build"
    if build_dir.exists():
        print(f"Removing {build_dir}")
        shutil.rmtree(build_dir)
    
    # Remove dist directory
    dist_dir = PROJECT_ROOT / "dist"
    if dist_dir.exists():
        print(f"Removing {dist_dir}")
        shutil.rmtree(dist_dir)
    
    # Remove __pycache__ directories BUT preserve logs directory
    for pycache in PROJECT_ROOT.glob("**/__pycache__"):
        if pycache.is_dir():
            print(f"Removing {pycache}")
            shutil.rmtree(pycache)
    
    print("Clean up completed!")
