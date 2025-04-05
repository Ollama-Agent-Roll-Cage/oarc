#!/usr/bin/env python3
"""
PyAudio utilities for OARC package.
"""

import sys
import platform
import subprocess
from pathlib import Path

from oarc.utils.log import log
from oarc.utils.setup.setup_utils import install_package


def install_pyaudio_dependencies_linux():
    """Install PyAudio dependencies on Linux systems."""
    log.info("Installing PyAudio dependencies for Linux...")
    
    try:
        # Check Linux distribution
        distro = ""
        if Path("/etc/os-release").exists():
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("ID="):
                        distro = line.split("=")[1].strip().strip('"')
                        break
        
        # Install dependencies based on distribution
        if distro in ["ubuntu", "debian", "linuxmint", "pop"]:
            log.info(f"Detected {distro} distribution, using apt-get...")
            subprocess.run(
                ["sudo", "apt-get", "update"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "portaudio19-dev", "python3-pyaudio"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        elif distro in ["fedora", "centos", "rhel"]:
            log.info(f"Detected {distro} distribution, using dnf/yum...")
            subprocess.run(
                ["sudo", "dnf", "install", "-y", "portaudio-devel", "python3-pyaudio"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        elif distro in ["arch", "manjaro"]:
            log.info(f"Detected {distro} distribution, using pacman...")
            subprocess.run(
                ["sudo", "pacman", "-S", "--noconfirm", "portaudio", "python-pyaudio"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        else:
            log.warning(f"Unsupported Linux distribution: {distro}. You may need to install PyAudio dependencies manually.")
            return False
        
        log.info("PyAudio dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install PyAudio dependencies: {e}")
        log.error("You may need to install them manually. For Debian/Ubuntu: 'sudo apt-get install portaudio19-dev python3-pyaudio'")
        log.debug(f"Stderr: {e.stderr.decode() if e.stderr else 'No error output'}")
        return False
    except Exception as e:
        log.error(f"Unexpected error installing PyAudio dependencies: {e}")
        return False


def install_pyaudio(venv_python=None):
    """
    Install PyAudio with platform-specific optimizations.
    
    Args:
        venv_python: Path to Python executable in virtual environment.
                    If None, use the current Python interpreter.
    
    Returns:
        bool: True if installation was successful
    """
    if venv_python is None:
        venv_python = Path(sys.executable)
    
    log.info("Installing PyAudio...")
    
    system = platform.system().lower()
    
    # Platform-specific preparation
    if system == "linux":
        if not install_pyaudio_dependencies_linux():
            log.warning("Failed to install PyAudio dependencies on Linux. Will try to install PyAudio anyway.")
    elif system == "darwin":  # macOS
        log.info("On macOS, installing PortAudio first...")
        try:
            # Check if Homebrew is installed
            subprocess.run(["brew", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Install PortAudio using Homebrew
            subprocess.run(["brew", "install", "portaudio"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log.info("PortAudio installed successfully with Homebrew.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("Homebrew not found or PortAudio installation failed. You may need to install PortAudio manually.")
    
    # Install PyAudio using our utility function
    # Use --no-build-isolation on non-Windows platforms to use system libraries
    if system in ["linux", "darwin"]:
        install_options = ["--no-build-isolation"]
    else:
        install_options = []
    
    # Use the install_package function from setup_utils
    return install_package("PyAudio", venv_python, install_options)
