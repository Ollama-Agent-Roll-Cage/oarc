#!/usr/bin/env python3
"""
OARC setup script for environment setup and package installation.
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

# Import utility functions directly
from oarc.utils.setup.setup_utils import (
    PROJECT_ROOT, VENV_DIR, CONFIG_DIR, LOG_DIR, TTS_REPO_DIR,
    read_build_config, clean_egg_info, fix_egg_deprecation,
    install_pyaudio_dependencies, install_tts_from_github,
    get_directory_size, format_size
)
from oarc.utils.setup.venv_utils import (
    check_python_version, detect_existing_venv, get_venv_python, ensure_pip
)
from oarc.utils.setup.build_utils import build_package
from oarc.utils.setup.logging_utils import setup_logging
from oarc.utils.setup.clean_project import clean_project

def install_development_dependencies(venv_python):
    """Install package in development mode and required dependencies."""
    ensure_pip(venv_python)
    
    print("Upgrading pip...")
    try:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Could not upgrade pip. Continuing with existing version.")
    
    # Clean egg_info directories before installation
    clean_egg_info()
    
    # First install core build tools and common dependencies
    print("Installing setuptools, wheel, build and core dependencies...")
    core_deps = [
        "setuptools>=45", 
        "wheel", 
        "build",
        "numpy<2.0.0,>=1.19.0",
        "cython"
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + core_deps, check=True)
    
    # Install the package in development mode FIRST
    print("Installing package in development mode...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", ".", "--no-deps"],
        check=True
    )
    
    # Then install dependencies separately
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
    
    print("Installing ML dependencies...")
    ml_deps = [
        "torch>=1.9.0",
        "transformers>=4.0.0", 
        "ollama>=0.1.0"
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + ml_deps, check=True)
    
    print("Installing audio dependencies...")
    audio_deps = [
        "SpeechRecognition>=3.8.1",
        "whisper>=1.0",
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + audio_deps, check=True)
    
    # Install PyAudio
    install_pyaudio_dependencies(venv_python)
    
    # Install TTS from GitHub only if it's not already installed
    try:
        subprocess.run(
            [str(venv_python), "-c", "import TTS; print(f'TTS already installed: {TTS.__file__}')"], 
            check=False, capture_output=True, text=True
        )
        print("TTS is already installed, skipping GitHub installation")
    except:
        install_tts_from_github(venv_python)
    
    # Verify package installation
    print("Verifying package installation...")
    verification_result = subprocess.run(
        [str(venv_python), "-c", "import oarc; print(f'OARC package installed successfully: {oarc.__file__}')"],
        check=False,
        capture_output=True,
        text=True
    )
    
    if verification_result.returncode != 0:
        print("Warning: OARC package verification failed. The package may not be correctly installed.")
        print(f"Error: {verification_result.stderr}")
    else:
        print(verification_result.stdout.strip())
    
    # Install dev dependencies
    print("Installing development tools...")
    dev_packages = ["twine", "pytest", "black", "flake8"]
    subprocess.run(
        [str(venv_python), "-m", "pip", "install"] + dev_packages,
        check=True
    )
    
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
        
        # Minimal setup call delegating to pyproject.toml for pip to process the package correctly
        setup()
        return

    # Parse arguments for our own commands
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
        build_package(venv_python, config, logger)
    else:
        print("Setting up OARC development environment...")
        if logger:
            logger.info("Starting OARC environment setup")

        venv_python = get_venv_python()
        
        # Install dependencies with fallback handling
        try:
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
                
        except subprocess.CalledProcessError as e:
            print(f"Error: Installation failed: {e}")
            print("Setup cannot continue. Please resolve the issues above and try again.")
            if logger:
                logger.error(f"Installation failed: {e}")
            sys.exit(1)

# This is the crucial part: setup.py needs to handle both direct execution and being imported by pip/setuptools
if __name__ == "__main__":
    main()
else:
    # When imported by pip or setuptools, just delegate to setuptools
    from setuptools import setup
    setup()