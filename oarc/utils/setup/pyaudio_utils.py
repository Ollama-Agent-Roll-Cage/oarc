#!/usr/bin/env python3
"""
PyAudio installation utilities for OARC setup process.
"""

import sys
import subprocess
import os
from oarc.decorators.log import log


@log()
def install_pyaudio(venv_python):
    """Install platform-specific dependencies for PyAudio."""
    log.info("Installing PyAudio dependencies...")
    
    try:
        if os.name == 'nt':  # Windows
            log.info("Installing PyAudio directly...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        
        elif sys.platform == 'darwin':  # macOS
            log.info("Installing PortAudio dependencies for macOS...")
            try:
                subprocess.run(["brew", "install", "portaudio"], check=True)
            except:
                log.warning("Could not install portaudio with brew. If PyAudio fails to install, install portaudio manually.")
            
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        
        else:  # Linux and others
            log.info("Installing PortAudio dependencies for Linux...")
            try:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "portaudio19-dev"], check=True)
            except:
                log.warning("Could not install portaudio with apt. If PyAudio fails to install, install portaudio19-dev manually.")
                
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
            
        log.info("PyAudio installed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install PyAudio: {e}")
        log.warning("Continuing without PyAudio. Some audio features may not work.")
        return False
