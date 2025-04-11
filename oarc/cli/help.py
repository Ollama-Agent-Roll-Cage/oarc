"""Help text definitions for OARC CLI."""

# Main CLI help text shown with --help or when no command is provided
MAIN_HELP = """
   _________     _____ ___________________
   \_____   \   /  _  \\__  __   \_   ___  \\
    /   |    \ /  /_\  \|       _/    \  \/
   /    |     /    |    \    |   \     \______
   \_______  /\____|__  /____|_  /\______    /
   /_________\/_________\/_______\/______\__/ 
   |__Ollama_Agent_Roll_Cage_(OARC_v0.1.1)_/

Usage:
  oarc [--debug] [--config <path>] [--force] <command> [options]

Commands:
  setup      Install and configure all OARC dependencies. Use --force to force reinstallation.
  build      Build the OARC package.
  publish    Publish the OARC package to a repository.

Example Usage:
  oarc setup --force
  oarc build

For more detailed information, please consult the README:
  oarc <command> --help
"""

# Command-specific help texts
SETUP_HELP = """
Setup Command Help

Usage: oarc setup [--force]

Options:
  --force    Force reinstallation of all dependencies even if already installed

Description:
  The setup command installs and configures all required OARC dependencies including:
  - Package managers (pip, uv)
  - ML/AI frameworks:
    * CUDA Toolkit (11.8 recommended)
    * PyTorch (with CUDA support)
    * TensorFlow
  - Speech components (Coqui TTS, PyAudio)
  - Vision components (YOLO, CUDA drivers)

After manual setup, run:
  oarc setup --force
"""

BUILD_HELP = """
Build Command Help

Usage: oarc build

Description:
  Builds the OARC package from source, creating a wheel distribution.
  The built package will be placed in the dist/ directory.
"""

RUN_HELP = """
Run Command Help

Usage: oarc run [--config <path>]

Options:
  --config <path>    Path to custom configuration file

Description:
  Starts the OARC application with the specified configuration.
  If no config is provided, default settings will be used.
"""

PUBLISH_HELP = """
Publish Command Help

Usage: oarc publish [--repository <repo>] [--dist-dir <dir>] [--skip-build]

Options:
  --repository <repo>   Repository to publish to (default: pypi)
  --dist-dir <dir>      Directory containing distribution files (default: dist)
  --skip-build          Skip building the package before publishing

Description:
  Publishes the OARC package to PyPI or another package repository.
  Authentication is handled through your .pypirc file.

  To set up authentication, create a .pypirc file in your home directory:
  
  [pypi]
  username = __token__
  password = pypi-example...
"""
