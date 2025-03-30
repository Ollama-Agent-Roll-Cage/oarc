"""Module to install additional dependencies for OARC."""

import sys
from pathlib import Path
from oarc.decorators.log import log
from oarc.utils.setup.setup_utils import ensure_pip
from oarc.utils.setup.tts_utils import install_coqui
from oarc.utils.setup.cuda_utils import install_pytorch
from oarc.utils.setup.pyaudio_utils import install_pyaudio
from oarc.utils.setup.setup_utils import install_self


@log()
def main():
    """Run all dependency installation steps."""
    # Get the current Python executable path
    venv_python = Path(sys.executable)
    log.info(f"Using Python executable: {venv_python}")
    
    # Setup pip first
    log.info("Setting up pip...")
    pip_success = ensure_pip(venv_python)
    
    # Run installation steps with proper error handling
    log.info("Installing Coqui TTS...")
    tts_success = install_coqui(venv_python)
    
    log.info("Installing PyAudio...")
    pyaudio_success = install_pyaudio(venv_python)
    
    log.info("Installing PyTorch...")
    pytorch_success = install_pytorch(venv_python)
    # self_success = install_self(venv_python)
    
    if pip_success and pytorch_success and pyaudio_success and tts_success:
        log.info("All dependencies installed successfully!")
    else:
        log.error("Some dependencies could not be installed")
        if not pip_success:
            log.error("- Pip setup failed")
        if not tts_success:
            log.error("- TTS installation failed")
        if not pytorch_success:
            log.error("- PyTorch installation failed")
        if not pyaudio_success:
            print("- PyAudio installation failed")
        # if not self_success:
        #     print("- Self installation failed")

if __name__ == "__main__":
    main()
