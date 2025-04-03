#!/usr/bin/env python3
"""
TTS installation utilities for OARC setup process.
"""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from oarc.utils.log import log

TTS_REPO_URL = "https://github.com/idiap/coqui-ai-TTS/archive/refs/heads/dev.zip"
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


def install_coqui(venv_python):
    """Install TTS directly from the GitHub repository.
    
    Args:
        venv_python: Path to Python executable in virtual environment
    
    Returns:
        bool: True if installation was successful
    
    Raises:
        SystemExit: If required parameters are not provided
    """
    # Fix: Use the project root directory instead of the package directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_dir = project_root / TTS_REPO_NAME
    
    # Validate required parameters
    if TTS_REPO_URL is None:
        log.error("Error: repo_url is required")
        sys.exit(1)

    if repo_dir.exists():
        log.info(f"TTS repository already exists at {repo_dir}. Skipping installation.")
        return True
    
    log.info(f"Installing TTS from GitHub repository to {repo_dir}...")
    
    # Convert to Path object if string is provided
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)
    
    # Check if coqui directory already exists
    if repo_dir.exists():
        log.info(f"Found existing TTS repository at {repo_dir}")
        try:
            # Try to remove .git directory first separately if it exists
            remove_git_dir(repo_dir)
            # Then remove the whole directory
            shutil.rmtree(repo_dir)
            log.info("Removed existing installation successfully.")
        except Exception as e:
            log.error(f"Error removing existing installation: {e}")
            return False

    # Create directory for the repository
    repo_dir.parent.mkdir(exist_ok=True, parents=True)
    
    # Download and extract repository
    try:
        log.info(f"Downloading TTS repository from {TTS_REPO_URL}...")
        
        # Create a temporary file to store the zip
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Download the zip file
        urllib.request.urlretrieve(TTS_REPO_URL, temp_path)
        log.info("Download complete.")
        
        # Extract the zip file
        log.info(f"Extracting to {repo_dir}...")
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            # The GitHub zip contains a top-level directory; extract to a temporary directory first
            temp_extract_dir = tempfile.mkdtemp()
            zip_ref.extractall(temp_extract_dir)
            
            # Find the directory inside the zip (should be only one)
            extracted_dirs = os.listdir(temp_extract_dir)
            if not extracted_dirs:
                raise Exception("Zip file extraction failed - no directories found")
                
            # Move the contents to the final destination
            extracted_dir_path = os.path.join(temp_extract_dir, extracted_dirs[0])
            if not os.path.exists(repo_dir.parent):
                os.makedirs(repo_dir.parent)
                
            shutil.move(extracted_dir_path, repo_dir)
            
            # Clean up temp directory
            shutil.rmtree(temp_extract_dir)
        
        # Clean up temp file
        os.unlink(temp_path)
        log.info("Extraction complete.")
        
        # Remove .git directory if present (sometimes can be in the zip)
        remove_git_dir(repo_dir)
        
    except Exception as e:
        log.error(f"Error downloading or extracting TTS repository: {e}")
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
        log.info("TTS installed successfully from GitHub!")
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
