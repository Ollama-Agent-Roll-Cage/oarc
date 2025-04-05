#!/usr/bin/env python3
"""
TTS installation utilities for OARC setup process.
"""

import os
import stat  # Added missing import
import shutil
import subprocess
import sys
from pathlib import Path

from oarc.utils.log import log

# Updated to use the idiap repository for Coqui TTS
TTS_REPO_URL = "https://github.com/idiap/coqui-ai-TTS.git"
TTS_REPO_NAME = "coqui-ai-TTS"


def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_git_dir(repo_dir):
    """Remove .git directory with special handling for Windows permission issues."""
    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        return True
    
    log.info(f"Removing Git directory from {git_dir}...")
    try:
        # First try normal removal
        shutil.rmtree(git_dir)
        log.info("Git directory removed successfully.")
        return True
    except PermissionError:
        # If permission error, try with onerror handler
        try:
            log.warning("Permission error encountered, trying with special handler...")
            shutil.rmtree(git_dir, onerror=remove_readonly)
            log.info("Git directory removed successfully.")
            return True
        except Exception as e:
            log.error(f"Failed to remove Git directory: {e}")
            return False


def install_coqui(venv_python, force=False):
    """Install Coqui TTS directly from the GitHub repository.
    
    Args:
        venv_python: Path to Python executable in virtual environment
        force: Force reinstallation even if already installed
    
    Returns:
        bool: True if installation was successful
    """
    # Fix: Use the project root directory instead of the package directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_dir = project_root / TTS_REPO_NAME
    
    # Convert to Path object if string is provided
    if isinstance(venv_python, str):
        venv_python = Path(venv_python)
    
    # Check if coqui directory exists
    if repo_dir.exists() and not force:
        log.info(f"TTS repository already exists at {repo_dir}. Use --force to reinstall.")
        return True
    
    log.info(f"{'Reinstalling' if force else 'Installing'} TTS from GitHub repository to {repo_dir}...")
    
    # Check if coqui directory exists - remove it if needed
    if repo_dir.exists():
        log.info(f"Removing existing TTS repository at {repo_dir} for fresh install")
        try:
            shutil.rmtree(repo_dir)
            log.info("Removed existing installation successfully.")
        except Exception as e:
            log.error(f"Error removing existing installation: {e}")
            # Continue even if deletion fails, git clone will handle it
    
    # Clone the repository using Git
    try:
        log.info(f"Cloning TTS repository from {TTS_REPO_URL}...")
        subprocess.run(
            ["git", "clone", TTS_REPO_URL, str(repo_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to clone repository: {e}")
        log.error(f"Stderr: {e.stderr.decode() if e.stderr else 'No error output'}")
        return False
    except Exception as e:
        log.error(f"Error during repository cloning: {e}")
        return False
    
    # Install in development mode
    log.info("Installing TTS in development mode...")
    try:
        # Use a higher timeout for pip install to avoid issues
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
            check=True,
            timeout=300  # 5-minute timeout
        )
        log.info("TTS installed successfully in editable mode!")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Error installing TTS: {e}. TTS installation failed.")
        return False
    except subprocess.TimeoutExpired:
        log.warning("Installation timed out. This may be normal for complex packages; please verify if TTS is working properly.")
        return True


def accept_coqui_license():
    """
    Create .coquirc file and set environment variables to pre-accept the Coqui TTS license and configure model storage.
    
    This ensures that TTS doesn't prompt for license acceptance during initialization and 
    that models are stored in our project directory rather than in AppData.
    
    Returns:
        bool: True if license acceptance was successful
    """
    # Set environment variable to accept Coqui TTS license
    os.environ["COQUI_TTS_AGREED"] = "1"
    
    # Set TTS_HOME to our Coqui directory from Paths API
    from oarc.utils.paths import Paths
    paths_instance = Paths()
    coqui_dir = paths_instance.get_coqui_dir()  # Use the existing Coqui directory
    os.environ["TTS_HOME"] = coqui_dir
    os.makedirs(coqui_dir, exist_ok=True)
    log.info(f"Set TTS_HOME to Coqui directory: {coqui_dir}")
    
    # Create license acceptance file content
    license_content = "license_accepted: true\n"
    
    # Place in user's home directory (default location)
    home_dir = Path.home()
    home_coquirc = home_dir / ".coquirc"
    
    if not home_coquirc.exists():
        log.info(f"Creating .coquirc file in home directory: {home_coquirc}")
        try:
            with open(home_coquirc, "w") as f:
                f.write(license_content)
        except Exception as e:
            log.warning(f"Failed to create .coquirc in home directory: {e}")
    
    # Also place in Coqui directory for redundancy
    coqui_coquirc = Path(coqui_dir) / ".coquirc"
    
    if not coqui_coquirc.exists():
        log.info(f"Creating .coquirc file in Coqui directory: {coqui_coquirc}")
        try:
            with open(coqui_coquirc, "w") as f:
                f.write(license_content)
        except Exception as e:
            log.warning(f"Failed to create .coquirc in Coqui directory: {e}")
    
    log.info("Coqui TTS license pre-accepted via environment variable and config files")
    return True
