#!/usr/bin/env python3
"""
OARC setup script for environment setup and package installation.
"""

import os
import sys
import subprocess
import argparse
import shutil
import configparser
from datetime import datetime
from pathlib import Path

# Define project constants
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
CONFIG_DIR = PROJECT_ROOT / "oarc" / "config_files"
LOG_DIR = PROJECT_ROOT / "logs"

def setup_logging():
    """Set up logging for the build process."""
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"build_{timestamp}.log"
    
    # Import our logging module
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from oarc.utils.log import Log
        Log.set_default_log_dir(LOG_DIR)
        Log.add_file_handler("build", f"build_{timestamp}.log")
        logger = Log.get_logger("build", with_file=True)
        print(f"Logging to {log_file}")
        return logger
    except ImportError:
        print(f"Warning: Unable to import Log module. Logs will only be shown on console.")
        return None

def read_build_config():
    """Read build configuration from .ini file."""
    config = configparser.ConfigParser()
    build_config_path = CONFIG_DIR / "build.ini"
    
    if build_config_path.exists():
        config.read(build_config_path)
        return config
    else:
        raise ValueError(f"Build configuration file not found: {build_config_path}")

def check_python_version(python_path, required_version=(3, 8)):
    """Check if the Python interpreter meets the version requirements."""
    try:
        # Use a single subprocess call to get the Python version
        version_output = subprocess.check_output(
            [str(python_path), "--version"],
            universal_newlines=True,
            stderr=subprocess.STDOUT  # Capture stderr too as some versions output to stderr
        ).strip()
        
        # Parse version with regex
        import re
        version_match = re.search(r"Python (\d+)\.(\d+)\.(\d+)", version_output)
        
        if not version_match:
            print(f"Warning: Could not parse Python version from: {version_output}")
            return False
        
        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        patch = int(version_match.group(3))
        
        print(f"Using Python version: {major}.{minor}.{patch} from interpreter: {python_path}")
        print(f"Python interpreter: {python_path}")

        # Compare version components
        current_version = (major, minor)
        return current_version >= required_version
        
    except Exception as e:
        print(f"Error checking Python version for {python_path}: {e}")
        return False

def detect_existing_venv():
    """Detect existing virtual environments in the project directory."""
    venv_dirs = [VENV_DIR]
    for venv_dir in venv_dirs:
        if not venv_dir.exists():
            continue
        if os.name == 'nt':
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
        if python_exe.exists() and check_python_version(python_exe):
            return venv_dir
    return None

def get_venv_python():
    """Get or create a Python executable in a virtual environment."""
    # Check if already in a virtual environment
    if sys.prefix != sys.base_prefix:
        print(f"Already in virtual environment: {sys.prefix}")
        return Path(sys.executable)
    
    # Check for existing virtual environment
    existing_venv = detect_existing_venv()
    if existing_venv:
        print(f"Using existing virtual environment at {existing_venv}")
        if os.name == 'nt':
            venv_python = existing_venv / "Scripts" / "python.exe"
        else:
            venv_python = existing_venv / "bin" / "python"
        return venv_python
    
    # Create a new virtual environment
    if os.name == 'nt':
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"
    
    # Use current Python to create venv
    print(f"Creating virtual environment at {VENV_DIR}...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    print("Virtual environment created successfully.")
    
    return venv_python

def ensure_pip(venv_python):
    """Ensure pip is installed before upgrading."""
    print("Ensuring pip is installed...")
    try:
        subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: ensurepip returned an error: {e}")
        print("Continuing with existing pip...")

def clean_egg_info():
    """Clean up existing egg-info directories."""
    print("Cleaning up existing egg-info directories...")
    for egg_info in PROJECT_ROOT.glob("*.egg-info"):
        if egg_info.is_dir():
            print(f"Removing {egg_info}")
            shutil.rmtree(egg_info)
    
    # Also check inside the oarc directory for egg_info
    oarc_dir = PROJECT_ROOT / "oarc"
    if oarc_dir.exists():
        for egg_info in oarc_dir.glob("*.egg-info"):
            if egg_info.is_dir():
                print(f"Removing {egg_info}")
                shutil.rmtree(egg_info)

def install_development_dependencies(venv_python):
    """Install package in development mode and required dependencies."""
    ensure_pip(venv_python)
    
    print("Upgrading pip...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Clean egg_info directories before installation
    clean_egg_info()
    
    # First install core build tools and common dependencies that cause problems
    print("Installing setuptools, wheel, build and core dependencies...")
    core_deps = [
        "setuptools>=45", 
        "wheel", 
        "build", 
        "numpy>=1.19.0",  # TTS dependency
        "cython"  # Often needed for scientific packages
    ]
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + core_deps, check=True)
    
    print("Installing package dependencies...")
    try:
        # Install the package dependencies first without the package itself
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "torch>=1.9.0", "transformers>=4.0.0"],
            check=True
        )
        
        print("Installing TTS and other audio-related packages...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "TTS>=0.8.0", "SpeechRecognition>=3.8.1", "whisper>=1.0"],
            check=False  # Don't fail if one package has issues
        )
        
        print("Installing package in development mode...")
        # Use pip for editable install instead of setup.py develop
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", "."],
            check=False,  # Don't fail if this has warnings
            capture_output=True,
            text=True
        )
        
        # Install extra dev dependencies
        print("Installing development tools...")
        dev_packages = ["twine", "pytest", "black", "flake8"]
        subprocess.run(
            [str(venv_python), "-m", "pip", "install"] + dev_packages,
            check=True
        )
        
        print("Package dependencies installed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing package dependencies: {e}")
        return False

def install_pyaudio_dependencies(venv_python):
    """Install platform-specific dependencies for PyAudio."""
    print("Installing PyAudio dependencies...")
    
    if os.name == 'nt':  # Windows
        print("Installing pipwin for PyAudio on Windows...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "pipwin"], check=True)
        try:
            subprocess.run([str(venv_python), "-m", "pipwin", "install", "pyaudio"], check=True)
        except subprocess.CalledProcessError:
            print("Warning: PyAudio installation through pipwin failed. Falling back to direct pip install.")
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=False)
    
    elif sys.platform == 'darwin':  # macOS
        print("Installing PortAudio dependencies for macOS...")
        try:
            subprocess.run(["brew", "install", "portaudio"], check=False)
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not install PortAudio via Homebrew.")
            print("Please install PortAudio manually: brew install portaudio")
    
    else:  # Linux and others
        print("Installing PortAudio dependencies for Linux...")
        try:
            subprocess.run(["apt-get", "update"], check=False)
            subprocess.run(["apt-get", "install", "-y", "portaudio19-dev"], check=False)
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not install PortAudio via apt-get.")
            print("Please install PortAudio manually: sudo apt-get install portaudio19-dev")

def clean_project():
    """Clean up build artifacts from the project directory."""
    print("Cleaning up build artifacts...")
    
    # Clean egg-info directories
    clean_egg_info()
    
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
    
    # Remove __pycache__ directories
    for pycache in PROJECT_ROOT.glob("**/__pycache__"):
        if pycache.is_dir():
            print(f"Removing {pycache}")
            shutil.rmtree(pycache)
    
    print("Clean up completed!")

def build_package(venv_python, config):
    """Build the package as a wheel."""
    logger = setup_logging()
    
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
    # Handle setuptools commands directly
    setuptools_commands = ['egg_info', 'bdist_wheel', 'sdist', 'install', 'develop', 'build_ext']
    
    if len(sys.argv) > 1 and any(cmd in sys.argv[1:] for cmd in setuptools_commands):
        # Clean egg_info directories before delegating to setuptools
        clean_egg_info()
        
        # Let setuptools handle package installation
        from setuptools import setup
        
        # This is a minimal setup call that delegates to pyproject.toml
        # It's needed for pip to process the package correctly
        setup()
        return

    # Parse arguments for our own commands
    args = parse_args_for_direct_execution()
    
    if args.clean:
        clean_project()
    elif args.build:
        venv_python = get_venv_python()
        ensure_pip(venv_python)
        config = read_build_config()
        build_package(venv_python, config)
    else:
        print("Setting up OARC development environment...")
        print("Note: This will set up the environment and install dependencies.")
        
        venv_python = get_venv_python()
        success = install_development_dependencies(venv_python)
        if success:
            install_pyaudio_dependencies(venv_python)
            print("\nOARC development environment is ready!")
            print(f"To activate the environment: ")
            if os.name == 'nt':
                print(f"  {VENV_DIR}\\Scripts\\activate")
            else:
                print(f"  source {VENV_DIR}/bin/activate")
            
            print("\nYou may see deprecation warnings about egg loading - these are harmless.")
            print("They're related to how some dependencies were installed by pip.")
        else:
            print("Setup encountered issues. Please check the errors above.")

# This is the crucial part: setup.py needs to handle both direct execution
# and being imported by pip/setuptools
if __name__ == "__main__":
    main()
else:
    # When imported by pip or setuptools, just delegate to setuptools
    from setuptools import setup
    setup()