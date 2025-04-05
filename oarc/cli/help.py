"""Help text definitions for OARC CLI."""

# Main CLI help text shown with --help or when no command is provided
MAIN_HELP = """
__________     _____ _____________________________________
_\_____   \   /  _  \\__  __   \_   ___ \/ ________________
_ /   |    \ /  /_\  \|       _/    \  \/  _______________
_/    |     /    |    \    |   \     \______ _____________
_\_______  /\____|__  /____|_  /\______    / _____________
_/_________\/_________\/_______\/______\__/ ______________
 
Ollama Agent Roll Cage (OARC v0.1.0) • Apache 2.0 License

Usage:
  oarc [--debug] [--config <path>] [--force] <command> [options]

Commands:
  setup      Install and configure all OARC dependencies. Use --force to force reinstallation.
  build      Build the OARC package.
  run        Run the OARC application.

Example Usage:
  oarc setup --force
  oarc build
  oarc run

For more detailed information, please consult the README or run:
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
  - ML/AI frameworks (PyTorch, TensorFlow)
  - Speech components (Coqui TTS, PyAudio)
  - Vision components (YOLO, CUDA drivers)
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
