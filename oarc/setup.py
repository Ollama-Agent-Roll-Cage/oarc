"""Module to install additional dependencies for OARC."""

import sys
from pathlib import Path
from oarc.utils.setup.setup_utils import ensure_pip
from oarc.utils.setup.tts_utils import install_coqui
from oarc.utils.setup.cuda_utils import install_pytorch
from oarc.utils.setup.pyaudio_utils import install_pyaudio
from oarc.utils.setup.setup_utils import install_self


def main():
    """Run all dependency installation steps."""
    # Get the current Python executable path
    venv_python = Path(sys.executable)
    print(f"Using Python executable: {venv_python}")
    
    # Setup pip first
    pip_success = ensure_pip(venv_python)
    
    # Run installation steps with proper error handling
    tts_success = install_coqui(venv_python)
    pyaudio_success = install_pyaudio(venv_python)
    pytorch_success = install_pytorch(venv_python)
    self_success = install_self(venv_python)
    
    if pip_success and pytorch_success and pyaudio_success and tts_success:
        print("All dependencies installed successfully!")
    else:
        print("Some dependencies could not be installed. Check the logs for details.")
        if not pip_success:
            print("- Pip setup failed")
        if not tts_success:
            print("- TTS installation failed")
        if not pytorch_success:
            print("- PyTorch installation failed")
        if not pyaudio_success:
            print("- PyAudio installation failed")
        if not self_success:
            print("- Self installation failed")

if __name__ == "__main__":
    main()
