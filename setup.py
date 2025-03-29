#!/usr/bin/env python3
"""
OARC setup script for environment setup and package installation.

This script handles both direct execution (for creating virtual environments 
and installing dependencies) and being imported by pip/setuptools.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path to import our utilities
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import utility modules
from oarc.utils.setup.setup_utils import (
    PROJECT_ROOT, VENV_DIR, read_build_config, clean_egg_info,
    get_directory_size, format_size
)
from oarc.utils.setup.pyaudio_utils import install_pyaudio_dependencies
from oarc.utils.setup.tts_utils import install_tts_from_github
from oarc.utils.setup.venv_utils import (
    get_venv_python, ensure_pip,
    check_pip_functionality
)
from oarc.utils.setup.build_utils import build_package
from oarc.utils.setup.logging_utils import setup_logging
from oarc.utils.setup.clean_project import clean_project
from oarc.utils.setup.cuda_utils import install_pytorch_with_cuda

def install_development_dependencies(venv_python):
    """Install package in development mode with all required dependencies."""
    ensure_pip(venv_python)
    
    # Verify pip functionality before proceeding
    if not check_pip_functionality(venv_python):
        raise RuntimeError("pip is not functioning correctly. Cannot continue with installation. "
                          "Try recreating the virtual environment with: python -m venv .venv --clear")
    
    print("Upgrading pip...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Clean egg_info directories before installation
    clean_egg_info()
    
    # Core build tools and dependencies
    print("Installing setuptools, wheel, build and core dependencies...")
    core_deps = [
        "setuptools>=45", 
        "wheel", 
        "build",
        "numpy>=1.24.3",
        "cython>=0.30.0"
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + core_deps, check=True)
    
    # Base dependencies
    print("Installing base dependencies...")
    base_deps = [
        "fastapi>=0.68.0",
        "keyboard>=0.13.5", 
        "pandas>=1.3.0",
        "python-multipart>=0.0.5",
        "pydantic>=1.8.0",
        "requests>=2.26.0",
        "uvicorn>=0.15.0",
        "websockets>=10.0",
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + base_deps, check=True)
    
    # ML dependencies with CUDA support if available
    print("Installing ML dependencies...")
    if not install_pytorch_with_cuda(venv_python):
        raise RuntimeError("Failed to install PyTorch. Setup cannot continue.")
    
    other_ml_deps = [
        "transformers>=4.0.0", 
        "ollama>=0.5.0"
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + other_ml_deps, check=True)
    
    # Audio processing dependencies
    print("Installing audio dependencies...")
    audio_deps = [
        "SpeechRecognition>=3.8.1",
        "whisper>=1.0",
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + audio_deps, check=True)
    
    # Platform-specific PyAudio installation
    if not install_pyaudio_dependencies(venv_python):
        raise RuntimeError("Failed to install PyAudio. Setup cannot continue.")
    
    # TTS installation from GitHub
    install_tts_from_github(venv_python)
    
    # Ensure torch-related packages are properly installed
    print("Installing remaining dependencies...")
    all_remaining_deps = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchvision>=0.15.0"
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + all_remaining_deps, check=True)
    
    # Install package in development mode
    print("Installing package in development mode...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", ".", "--no-deps"], check=True)
    
    # Verify installation
    print("Verifying package functionality...")
    subprocess.run(
        [str(venv_python), "-c", "import sys; sys.path.insert(0, '.'); import oarc; print(f'OARC package imported successfully')"],
        check=True
    )
    
    # Development tools installation
    print("Installing development tools...")
    dev_packages = ["twine", "pytest", "black", "flake8"]
    subprocess.run([str(venv_python), "-m", "pip", "install"] + dev_packages, check=True)
    
    print("Package dependencies installed successfully.")
    return True

def parse_args_for_direct_execution():
    """Parse command line arguments for direct script execution."""
    parser = argparse.ArgumentParser(
        description="OARC Setup Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py              Create/setup virtual environment
  python setup.py --clean      Clean up build artifacts
  python setup.py --build      Build package distribution
"""
    )
    parser.add_argument('--clean', action='store_true', help='Clean up build artifacts')
    parser.add_argument('--build', action='store_true', help='Build package distributions')
    
    # Only parse known args to avoid errors with setuptools commands
    return parser.parse_known_args()[0]

def main():
    """Main entry point for setup script when run directly."""
    # Set up logging
    logger = setup_logging()
    
    # Record start time for telemetry
    start_time = datetime.now()
    initial_venv_size = 0
    if VENV_DIR.exists():
        initial_venv_size = get_directory_size(VENV_DIR)
    
    # Handle setuptools commands directly
    setuptools_commands = ['egg_info', 'bdist_wheel', 'sdist', 'install', 'develop', 'build_ext']
    
    if len(sys.argv) > 1 and any(cmd in sys.argv[1:] for cmd in setuptools_commands):
        # Clean egg_info directories before delegating to setuptools
        clean_egg_info()
        
        # Let setuptools handle package installation
        from setuptools import setup
        setup()
        return

    # Parse arguments for custom commands
    args = parse_args_for_direct_execution()
    
    if args.clean:
        if logger:
            logger.info("Cleaning up build artifacts")
        clean_project()
    elif args.build:
        if logger:
            logger.info("Starting package build")
        venv_python = get_venv_python()
        ensure_pip(venv_python)
        config = read_build_config()
        if not build_package(venv_python, config, logger):
            sys.exit(1)  # Exit with error code if build fails
    else:
        print("Setting up OARC development environment...")
        if logger:
            logger.info("Starting OARC environment setup")

        venv_python = get_venv_python()
        
        # Install dependencies - any errors will propagate and fail the script
        install_development_dependencies(venv_python)
        
        print("\nOARC development environment is ready!")
        print(f"To activate the environment: ")
        if os.name == 'nt':
            print(f"  {VENV_DIR}\\Scripts\\activate")
        else:
            print(f"  source {VENV_DIR}/bin/activate")
            
        # Telemetry data collection
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        final_venv_size = 0
        if VENV_DIR.exists():
            final_venv_size = get_directory_size(VENV_DIR)
            
        size_change = final_venv_size - initial_venv_size
            
        print("\nSetup Telemetry:")
        print(f"  Total setup time: {elapsed_time}")
        print(f"  Virtual environment size: {format_size(final_venv_size)}")
        if initial_venv_size > 0:
            print(f"  Size change: {format_size(size_change)} ({(size_change/initial_venv_size)*100:.1f}% change)")
        
        if logger:
            logger.info(f"Setup completed successfully in {elapsed_time}")
            logger.info(f"Environment size: {format_size(final_venv_size)}")

# Handle both direct execution and being imported by pip
if __name__ == "__main__":
    main()
else:
    # When imported by pip or setuptools, delegate to setuptools
    from setuptools import setup
    setup()