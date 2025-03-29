#!/usr/bin/env python3
"""
CUDA setup utilities for OARC package.
"""

import os
import subprocess
import platform
import logging
from pathlib import Path

def check_cuda_capable():
    """Check if the current system has CUDA capabilities.
    
    Returns:
        tuple: (is_cuda_available, cuda_version)
            is_cuda_available (bool): True if CUDA is available, False otherwise
            cuda_version (str): CUDA version if available, None otherwise
    """
    print("Checking system for CUDA capabilities...")
    
    # Initialize variables
    is_cuda_available = False
    cuda_version = None
    
    # Check for NVIDIA GPU with different methods depending on the OS
    system = platform.system()
    try:
        if system == "Windows":
            # Check for NVIDIA GPU using Windows Management Instrumentation
            wmic_cmd = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, check=False
            )
            if "NVIDIA" in wmic_cmd.stdout:
                is_cuda_available = True
                
                # Try to get CUDA version using nvcc
                try:
                    nvcc_cmd = subprocess.run(
                        ["nvcc", "--version"],
                        capture_output=True, text=True, check=False
                    )
                    if nvcc_cmd.returncode == 0:
                        # Extract version from nvcc output (e.g., "release 12.4")
                        import re
                        version_match = re.search(r"release (\d+\.\d+)", nvcc_cmd.stdout)
                        if version_match:
                            cuda_version = version_match.group(1)
                except FileNotFoundError:
                    # Try to check CUDA version from environment variable
                    if "CUDA_PATH_V" in os.environ:
                        # Environment variables like CUDA_PATH_V11_8
                        cuda_env_vars = [v for v in os.environ if v.startswith("CUDA_PATH_V")]
                        if cuda_env_vars:
                            # Get the highest version if multiple are installed
                            versions = []
                            for var in cuda_env_vars:
                                try:
                                    # Extract version from variable name (e.g., "CUDA_PATH_V11_8")
                                    ver = var.replace("CUDA_PATH_V", "").replace("_", ".")
                                    versions.append(ver)
                                except ValueError:
                                    pass
                            if versions:
                                cuda_version = max(versions)
        
        elif system == "Linux":
            # Check for NVIDIA GPU using nvidia-smi
            nvidia_smi_cmd = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=False
            )
            if nvidia_smi_cmd.returncode == 0:
                is_cuda_available = True
                
                # Try to get CUDA version using nvcc
                try:
                    nvcc_cmd = subprocess.run(
                        ["nvcc", "--version"],
                        capture_output=True, text=True, check=False
                    )
                    if nvcc_cmd.returncode == 0:
                        import re
                        version_match = re.search(r"release (\d+\.\d+)", nvcc_cmd.stdout)
                        if version_match:
                            cuda_version = version_match.group(1)
                except FileNotFoundError:
                    # Try to determine CUDA version from nvidia-smi
                    import re
                    version_match = re.search(r"CUDA Version: (\d+\.\d+)", nvidia_smi_cmd.stdout)
                    if version_match:
                        cuda_version = version_match.group(1)
        
        elif system == "Darwin":  # macOS
            # macOS with Apple Silicon may have Metal support but not CUDA
            # For Intel Macs, CUDA support was discontinued after 10.13
            print("macOS detected. Apple Silicon uses MPS backend, not CUDA. Intel Macs have limited CUDA support.")
            is_cuda_available = False
            cuda_version = None
    
    except Exception as e:
        print(f"Error checking CUDA capabilities: {e}")
        is_cuda_available = False
        cuda_version = None
    
    # Print result
    if is_cuda_available and cuda_version:
        print(f"CUDA is available! Detected CUDA version: {cuda_version}")
    elif is_cuda_available:
        print("CUDA is available, but could not determine version.")
    else:
        print("CUDA is not available on this system.")
    
    return is_cuda_available, cuda_version

def get_pytorch_cuda_command(cuda_version=None, skip_cuda=False):
    """Get the appropriate pip command for installing PyTorch with CUDA support.
    
    Args:
        cuda_version (str, optional): CUDA version (e.g., "11.8"). If None, 
                                      it will select based on detected version.
        skip_cuda (bool, optional): Force CPU-only installation.
        
    Returns:
        tuple: (pip_command, packages, index_url)
            pip_command (list): Base pip command
            packages (list): Packages to install
            index_url (str): URL for package index, if applicable
    """
    # Define base pip command and packages
    pip_command = ["pip", "install"]
    packages = ["torch", "torchvision", "torchaudio"]
    
    # Return CPU version if skip_cuda is True or no CUDA version provided
    if skip_cuda:
        print("Installing PyTorch CPU version (skip_cuda=True).")
        return pip_command, packages, None
    
    # Map CUDA versions to PyTorch wheel URLs
    cuda_urls = {
        # Format: "major.minor": ("wheel_suffix", "index_url")
        "12.4": ("cu124", "https://download.pytorch.org/whl/cu124"),
        "12.3": ("cu123", "https://download.pytorch.org/whl/cu123"),
        "12.1": ("cu121", "https://download.pytorch.org/whl/cu121"),
    }
    
    # Select the appropriate CUDA version
    if cuda_version and cuda_version in cuda_urls:
        # Exact match
        selected_version = cuda_version
    elif cuda_version:
        # Find the closest compatible version (using major version)
        major_version = cuda_version.split('.')[0]
        compatible_versions = [v for v in cuda_urls.keys() if v.startswith(major_version)]
        if compatible_versions:
            # Use the highest compatible version
            selected_version = max(compatible_versions)
        else:
            print(f"No compatible CUDA version found for {cuda_version}, defaulting to CPU version.")
            return pip_command, packages, None
    else:
        # If no version specified, use the latest supported version
        selected_version = max(cuda_urls.keys())
    
    _, index_url = cuda_urls[selected_version]
    print(f"Using PyTorch with CUDA {selected_version} support from {index_url}")
    
    return pip_command, packages, index_url

def install_pytorch_with_cuda(venv_python, logger=None):
    """Install PyTorch with CUDA support if available.
    
    Args:
        venv_python: Path to Python executable in virtual environment
        logger: Logger object, if available
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    if logger:
        logger.info("Checking CUDA capabilities for PyTorch installation")
    
    # Check if CUDA is available
    is_cuda_available, cuda_version = check_cuda_capable()
    
    # Get installation command
    pip_command, packages, index_url = get_pytorch_cuda_command(
        cuda_version=cuda_version, 
        skip_cuda=not is_cuda_available
    )
    
    # Build the full command
    full_cmd = [str(venv_python), "-m"] + pip_command + packages
    if index_url:
        full_cmd.extend(["--index-url", index_url])
    
    # Log the command
    cmd_str = " ".join(full_cmd)
    print(f"Installing PyTorch with command: {cmd_str}")
    if logger:
        logger.info(f"Installing PyTorch with command: {cmd_str}")
    
    # Run the installation
    try:
        result = subprocess.run(full_cmd, check=True)
        success = result.returncode == 0
        if success:
            print("PyTorch installation completed successfully!")
            if logger:
                logger.info("PyTorch installation completed successfully")
        else:
            print(f"PyTorch installation failed with return code {result.returncode}")
            if logger:
                logger.error(f"PyTorch installation failed with return code {result.returncode}")
        return success
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch: {e}")
        if logger:
            logger.error(f"Error installing PyTorch: {e}")
        return False

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    is_cuda_available, cuda_version = check_cuda_capable()
    print(f"CUDA available: {is_cuda_available}, version: {cuda_version}")
    
    pip_cmd, packages, index_url = get_pytorch_cuda_command(cuda_version)
    print(f"Installation command: pip {' '.join(packages)} {'--index-url ' + index_url if index_url else ''}")
