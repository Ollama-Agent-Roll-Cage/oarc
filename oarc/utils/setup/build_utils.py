#!/usr/bin/env python3
"""
Build utilities for OARC package.
"""

import subprocess
import sys
from pathlib import Path
import shutil


def build_package(venv_python=None, clean=True):
    """Build the OARC package wheel.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                     If None, use the current Python interpreter.
        clean: Whether to clean the build directories before building.
    
    Returns:
        Path: Path to the built wheel file
    
    Raises:
        subprocess.CalledProcessError: If build fails
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    print(f"Building package using Python from: {venv_python}")
    
    # Get the project root directory
    project_dir = Path(__file__).resolve().parents[3]  # Adjust based on actual location
    
    # Clean build directories if requested
    if clean:
        dirs_to_clean = ['build', 'dist', '*.egg-info']
        for dir_pattern in dirs_to_clean:
            for path in project_dir.glob(dir_pattern):
                if path.is_dir():
                    print(f"Cleaning {path}")
                    shutil.rmtree(path)
    
    # Install build dependencies
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "build", "wheel"],
        check=True
    )
    
    # Build the package
    subprocess.run(
        [str(venv_python), "-m", "build", "--wheel"],
        cwd=project_dir,
        check=True
    )
    
    # Find the built wheel
    wheels = list(project_dir.joinpath("dist").glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheel file was created during build")
    
    wheel_path = wheels[-1]  # Get the latest wheel
    print(f"Package built successfully: {wheel_path}")
    
    return wheel_path


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    build_package()
