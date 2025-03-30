#!/usr/bin/env python3
"""
TTS installation utilities for OARC setup process.
"""

import subprocess
import sys
import shutil
import os
import tempfile
import zipfile
import urllib.request
import time
import stat
from pathlib import Path

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
    
    print(f"Removing Git directory from {git_dir}...")
    try:
        # First try normal removal
        shutil.rmtree(git_dir)
        print("Git directory removed successfully.")
        return True
    except PermissionError:
        # If permission error, try with onerror handler
        try:
            print("Permission error encountered, trying with special handler...")
            shutil.rmtree(git_dir, onerror=remove_readonly)
            print("Git directory removed successfully.")
            return True
        except Exception as e:
            print(f"Failed to remove Git directory: {e}")
            print("You may need to manually remove it.")
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
        print("Error: repo_url is required")
        sys.exit(1)
    
    print(f"Installing TTS from GitHub repository to {repo_dir}...")
    
    # Convert to Path object if string is provided
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)
    
    # Check if coqui directory already exists
    if repo_dir.exists():
        print(f"Found existing TTS repository at {repo_dir}")
        print("Removing existing installation...")
        try:
            # Try to remove .git directory first separately if it exists
            remove_git_dir(repo_dir)
            # Then remove the whole directory
            shutil.rmtree(repo_dir)
            print("Removed existing installation successfully.")
        except Exception as e:
            print(f"Error removing existing installation: {e}")
            return False

    # Create directory for the repository
    repo_dir.parent.mkdir(exist_ok=True, parents=True)
    
    # Download and extract repository
    try:
        print(f"Downloading TTS repository from {TTS_REPO_URL}...")
        
        # Create a temporary file to store the zip
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Download the zip file
        urllib.request.urlretrieve(TTS_REPO_URL, temp_path)
        print("Download complete.")
        
        # Extract the zip file
        print(f"Extracting to {repo_dir}...")
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            # The GitHub zip contains a top-level directory, typically named with branch
            # Extract to a temporary directory first
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
        print("Extraction complete.")
        
        # Remove .git directory if present (sometimes can be in the zip)
        remove_git_dir(repo_dir)
        
    except Exception as e:
        print(f"Error downloading or extracting TTS repository: {e}")
        return False
    
    # Install in development mode
    print("Installing TTS in development mode...")
    try:
        # Use a higher timeout for pip install to avoid issues
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
            check=True,
            timeout=300  # 5-minute timeout
        )
        print("TTS installed successfully from GitHub!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing TTS: {e}")
        print("TTS installation failed.")
        return False
    except subprocess.TimeoutExpired:
        print("Installation timed out. This may be normal for complex packages.")
        print("Installation might still be in progress or completed.")
        print("Please check if TTS is working properly.")
        return True
