#!/usr/bin/env python3
"""
OARC Setup Module

This module provides the main dependency installation functionality for the OARC project.
It installs key dependencies like Coqui TTS, PyAudio, and PyTorch with appropriate
platform-specific optimizations.

When run as a script, it performs a complete setup of all dependencies.
It can also be imported and used programmatically through the main() function.
"""

import sys
from pathlib import Path
from oarc.utils.setup.setup_utils import ensure_pip
from oarc.utils.setup.tts_utils import install_coqui
from oarc.utils.setup.cuda_utils import install_pytorch
from oarc.utils.setup.pyaudio_utils import install_pyaudio
from oarc.utils.log import log


def main(force=False):
    """Run all dependency installation steps.
    
    Args:
        force (bool): Force reinstallation of dependencies even if already installed
    """
    # Get the current Python executable path
    venv_python = Path(sys.executable)
    log.info(f"Using Python executable: {venv_python}")

    # Setup package managers first (pip and uv)
    log.info("Setting up package managers...")
    pkg_mgr_success = ensure_pip(venv_python)

    # Track errors
    errors = []
    if not pkg_mgr_success:
        errors.append("Package manager setup had issues")

    log.info("Installing PyAudio...")
    pyaudio_success = install_pyaudio(venv_python)
    if not pyaudio_success:
        errors.append("PyAudio installation failed")

    # Run installation steps with proper error handling
    log.info("Installing Coqui TTS...")
    tts_success = install_coqui(venv_python, force=force)
    if not tts_success:
        errors.append("TTS installation failed")

    log.info("Installing PyTorch...")
    pytorch_success = install_pytorch(venv_python, force=force)
    if not pytorch_success:
        errors.append("PyTorch installation failed")

    success = len(errors) == 0
    if success:
        log.info("All dependencies installed successfully!")
    else:
        log.error("Some dependencies could not be installed")
        for error in errors:
            log.error(f"- {error}")

    return success


# Make the module directly runnable
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup OARC dependencies")
    parser.add_argument('--force', action='store_true', help='Force reinstallation of dependencies')
    args = parser.parse_args()

    success = main(force=args.force)
    sys.exit(0 if success else 1)
