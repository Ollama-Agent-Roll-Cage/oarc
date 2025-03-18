@echo off
:: Create and activate virtual environment
python -m venv .venv
call .venv\Scripts\activate

:: Install package in development mode with dev dependencies
pip install -e ".[dev]"

:: Install PyAudio dependencies on Windows
pip install pipwin
pipwin install pyaudio

echo.
echo Development environment setup complete!
echo To activate the environment: .venv\Scripts\activate