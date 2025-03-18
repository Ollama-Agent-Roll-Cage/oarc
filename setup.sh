#!/bin/bash

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package in development mode with dev dependencies
pip install -e ".[dev]"

# Install PyAudio dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew install portaudio
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo apt-get install portaudio19-dev
fi

pip install pyaudio

echo
echo "Development environment setup complete!"
echo "To activate the environment: source .venv/bin/activate"