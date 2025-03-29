#!/usr/bin/env python3
"""
Build utilities for OARC package.
"""

import subprocess
from pathlib import Path

from .setup_utils import PROJECT_ROOT
from .clean_project import clean_project

def build_package(venv_python, config, logger=None):
    """Build the package as a wheel."""
    print("Building package distribution...")
    if logger:
        logger.info("Starting package build")
    
    if config.getboolean("build", "clean_before_build", fallback=True):
        clean_project()
        if logger:
            logger.info("Cleaned project directories")
    
    # Install build requirements
    subprocess.run([str(venv_python), "-m", "pip", "install", "build", "wheel", "setuptools>=45"], check=True)
    
    # Build the package
    build_cmd = [str(venv_python), "-m", "build"]
    
    # Add options based on config
    if not config.getboolean("build", "build_wheel", fallback=True):
        build_cmd.append("--no-wheel")
    
    if config.getboolean("build", "build_sdist", fallback=False):
        build_cmd.append("--sdist")
    else:
        build_cmd.append("--no-sdist")
        
    if logger:
        logger.info(f"Running build command: {' '.join(str(x) for x in build_cmd)}")
    
    result = subprocess.run(build_cmd, check=False)
    
    if result.returncode == 0:
        print("Package built successfully!")
        if logger:
            logger.info("Package built successfully")
    else:
        print("Package build failed.")
        if logger:
            logger.error("Package build failed")
        return False
    
    # List the built packages
    dist_dir = PROJECT_ROOT / "dist"
    if dist_dir.exists():
        print("\nBuilt packages:")
        for package in dist_dir.glob("*"):
            print(f"  - {package.name}")
            if logger:
                logger.info(f"Built package: {package.name}")
    
    return True
