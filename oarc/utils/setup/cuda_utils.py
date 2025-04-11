#!/usr/bin/env python3
"""
CUDA setup utilities for OARC package.
"""

import os
import subprocess
import platform
import sys
import tempfile

from oarc.utils.log import log


def check_cuda_capable():
    """Check if the current system has CUDA capabilities.
    
    Returns:
        tuple: (is_cuda_available, cuda_version)
            is_cuda_available (bool): True if CUDA is available, False otherwise
            cuda_version (str): CUDA version if available, None otherwise
    """
    log.info("Checking system for CUDA capabilities...")
    
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
            log.info("macOS detected. Apple Silicon uses MPS backend, not CUDA. Intel Macs have limited CUDA support.")
            is_cuda_available = False
            cuda_version = None
    
    except Exception as e:
        log.error(f"Error checking CUDA capabilities: {e}")
        is_cuda_available = False
        cuda_version = None
    
    # Verify that CUDA installation is functional by checking nvidia-smi
    if is_cuda_available:
        try:
            nvidia_smi_cmd = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=False
            )
            if nvidia_smi_cmd.returncode != 0:
                log.warning("nvidia-smi command failed - CUDA might not be properly installed")
                is_cuda_available = False
                cuda_version = None
            else:
                log.info("nvidia-smi detected a working NVIDIA GPU")
        except Exception as e:
            log.warning(f"Failed to run nvidia-smi: {e}")
    
    # Log result
    if is_cuda_available and cuda_version:
        log.info(f"CUDA is available! Detected CUDA version: {cuda_version}")
    elif is_cuda_available:
        log.info("CUDA is available, but could not determine version.")
    else:
        log.info("CUDA is not available on this system.")
    
    return is_cuda_available, cuda_version


def verify_pytorch_cuda():
    """Verify if the currently installed PyTorch has CUDA support.
    
    Returns:
        bool: True if PyTorch has CUDA support, False otherwise
    """
    log.info("Verifying if PyTorch has CUDA support...")
    
    # Create a temporary Python script to check PyTorch CUDA
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write("""
import torch
import sys

if torch.cuda.is_available():
    print(f"PyTorch has CUDA support! Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    sys.exit(0)
else:
    print("PyTorch does not have CUDA support.")
    sys.exit(1)
""")
        script_path = f.name
    
    try:
        # Run the verification script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, check=False
        )
        
        has_cuda = result.returncode == 0
        if has_cuda:
            log.info(result.stdout.strip())
        else:
            log.warning(result.stdout.strip())
        
        # Clean up the temporary file
        os.unlink(script_path)
        return has_cuda
    
    except Exception as e:
        log.error(f"Error verifying PyTorch CUDA support: {e}")
        try:
            os.unlink(script_path)
        except:
            pass
        return False


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
    pip_command = ["pip", "install", "--force-reinstall"]
    packages = ["torch", "torchvision", "torchaudio"]
    
    # Return CPU version if skip_cuda is True or no CUDA version provided
    if skip_cuda:
        log.info("Installing PyTorch CPU version (skip_cuda=True).")
        return pip_command, packages, None
    
    # Map CUDA versions to PyTorch wheel URLs
    cuda_urls = {
        # Format: "major.minor": ("wheel_suffix", "index_url")
        "12.4": ("cu124", "https://download.pytorch.org/whl/cu124"),
        "12.3": ("cu123", "https://download.pytorch.org/whl/cu123"),
        "12.1": ("cu121", "https://download.pytorch.org/whl/cu121"),
        "11.8": ("cu118", "https://download.pytorch.org/whl/cu118"),
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
            log.info(f"No compatible CUDA version found for {cuda_version}, defaulting to CPU version.")
            return pip_command, packages, None
    else:
        # If no version specified, use the latest supported version
        selected_version = max(cuda_urls.keys())
    
    _, index_url = cuda_urls[selected_version]
    log.info(f"Using PyTorch with CUDA {selected_version} support from {index_url}")
    
    return pip_command, packages, index_url


def is_pytorch_installed():
    """Check if PyTorch is already installed.
    
    Returns:
        bool: True if PyTorch is installed, False otherwise
    """
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except ImportError:
        return False


def install_pytorch(venv_python, force=False):
    """Install PyTorch with CUDA support if available.
    
    Args:
        venv_python: Path to Python executable in virtual environment
        force: Force reinstallation even if already installed with CUDA
    
    Returns:
        bool: True if installation was successful
    """
    log.info("Checking CUDA capabilities for PyTorch installation")
    
    is_cuda_available, cuda_version = check_cuda_capable()
    
    if is_pytorch_installed():
        if force:
            log.info("Force flag set. Reinstalling PyTorch even if already installed with CUDA.")
        else:
            has_cuda = verify_pytorch_cuda()
            if has_cuda:
                log.info("Existing PyTorch installation already has CUDA support. No need to reinstall.")
                return True
            else:
                log.warning("PyTorch is installed but doesn't have CUDA support. Reinstalling with CUDA...")

    # First, ensure numpy is at the correct version to prevent conflicts with ultralytics
    try:
        from oarc.utils.setup.setup_utils import install_numpy
        log.info("Pre-installing numpy 2.1.1 to prevent version conflicts...")
        install_numpy(venv_python, version="2.1.1")
    except Exception as e:
        log.warning(f"Failed to pre-install numpy: {e}")

    # Get installation command with no dependencies
    pip_command, packages, index_url = get_pytorch_cuda_command(
        cuda_version=cuda_version, 
        skip_cuda=not is_cuda_available
    )
    
    # Build the PyTorch command with --no-deps flag to prevent dependency installation
    pytorch_packages = [pkg for pkg in packages]  # Create a copy of packages list
    full_cmd = [str(venv_python), "-m"] + pip_command + pytorch_packages + ["--no-deps"]
    if index_url:
        full_cmd.extend(["--index-url", index_url])
    
    # Log the command
    cmd_str = " ".join(full_cmd)
    log.info(f"Installing PyTorch with command: {cmd_str}")
    
    try:
        # Install PyTorch without dependencies
        result = subprocess.run(full_cmd, check=True, timeout=1800)  # 30 minutes
        
        # Now install PyTorch's dependencies (except numpy which we already installed)
        log.info("Installing PyTorch dependencies...")
        
        # Common PyTorch dependencies, excluding numpy
        dependencies = [
            "filelock", "typing-extensions>=4.10.0", "sympy==1.13.1",
            "networkx", "jinja2", "fsspec", "pillow!=8.3.*,>=5.3.0"
        ]
        
        # Install dependencies one by one
        for dep in dependencies:
            try:
                from oarc.utils.setup.setup_utils import install_package
                install_package(dep, venv_python)
            except Exception as e:
                log.warning(f"Failed to install dependency {dep}: {e}")
        
        # Verify numpy is still at the correct version
        try:
            result = subprocess.run(
                [str(venv_python), "-c", "import numpy; print(numpy.__version__)"],
                capture_output=True, text=True, check=False
            )
            numpy_version = result.stdout.strip()
            log.info(f"Current numpy version: {numpy_version}")
            
            if numpy_version != "2.1.1":
                log.warning(f"numpy is at version {numpy_version}, reinstalling correct version...")
                install_numpy(venv_python, version="2.1.1")
        except Exception as e:
            log.warning(f"Failed to check numpy version: {e}")
        
        # Verify PyTorch installation with CUDA
        log.info("Verifying PyTorch installation with CUDA support...")
        has_cuda = verify_pytorch_cuda()
        
        if has_cuda and is_cuda_available:
            log.info("PyTorch with CUDA support installed successfully!")
            return True
        elif not has_cuda and is_cuda_available:
            log.error("PyTorch was installed but CUDA support is not working. Something might be wrong with your CUDA setup.")
            return False
        else:
            log.info("PyTorch (CPU version) installed successfully!")
            return True
    except subprocess.CalledProcessError as e:
        log.error(f"Error installing PyTorch: {e}")
        return False
    except subprocess.TimeoutExpired:
        log.error("PyTorch installation timed out. This might be due to network issues or large download size.")
        return False
