#!/usr/bin/env python3
"""
PyAudio installation utilities for OARC setup process.
"""

import sys
import subprocess
import os

def install_pyaudio_dependencies(venv_python):
    """Install platform-specific dependencies for PyAudio."""
    print("Installing PyAudio dependencies...")
    
    try:
        if os.name == 'nt':  # Windows
            print("Installing PyAudio directly...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        
        elif sys.platform == 'darwin':  # macOS
            print("Installing PortAudio dependencies for macOS...")
            try:
                subprocess.run(["brew", "install", "portaudio"], check=True)
            except:
                print("Warning: Could not install portaudio with brew. If PyAudio fails to install, install portaudio manually.")
            
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
        
        else:  # Linux and others
            print("Installing PortAudio dependencies for Linux...")
            try:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y", "portaudio19-dev"], check=True)
            except:
                print("Warning: Could not install portaudio with apt. If PyAudio fails to install, install portaudio19-dev manually.")
                
            subprocess.run([str(venv_python), "-m", "pip", "install", "pyaudio"], check=True)
            
        print("PyAudio installed successfully.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install PyAudio: {e}")
        print("Continuing without PyAudio. Some audio features may not work.")
        return False
