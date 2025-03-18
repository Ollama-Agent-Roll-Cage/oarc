# OARC (ollamaAgentRollCage)

OARC is an open-source toolkit for building Ollama-powered speech-enabled AI agents with vision capabilities.

## Prerequisites

- Python 3.8+
- Ollama installed and running
- PyAudio dependencies (portaudio)

## Installation

### Installation

#### Recommended Method (Using Setup Scripts)

```bash
# On Windows:
setup.bat

# On macOS/Linux:
chmod +x setup.sh  # Make the script executable (first time only)
./setup.sh
```

These scripts will:

- Set up a virtual environment (for development)
- Install all required dependencies
- Handle platform-specific PyAudio requirements
- Configure the package appropriately

#### Manual Installation

For Users:
```bash
# Install PyAudio dependencies (platform-specific)
# Windows:
pip install pipwin && pipwin install pyaudio
# macOS:
# brew install portaudio && pip install pyaudio
# Linux:
# sudo apt-get install portaudio19-dev && pip install pyaudio

# Install package
pip install oarc
```

For Developers:
```bash
# Clone the repository
git clone https://github.com/Leoleojames1/oarc.git
cd oarc

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install PyAudio dependencies (platform-specific)
# Follow the platform-specific instructions above
```

## Quick Start

```python
from oarc import oarcAPI
from oarc.speechToSpeech import textToSpeech, speechToText

# Initialize API
api = oarcAPI()

# Use speech components
stt = speechToText()
tts = textToSpeech()

# Start listening
text = stt.recognizer(audio_file)

# Generate speech
tts.generate_audio("Hello, I am an Ollama AI assistant!")
```

## Advanced

```python
# Example usage
from oarc import oarcAPI
from oarc.speechToSpeech import textToSpeech
from oarc.promptModel import multiModalPrompting
from oarc.yoloProcessor import yoloProcessor

# Initialize components
api = oarcAPI()
tts = textToSpeech(developer_tools_dict={...})
vision = yoloProcessor()
llm = multiModalPrompting()

# Process speech and vision
text = api.recognize_speech(audio_file)
vision_data = vision.process_image(image_file)
response = llm.send_prompt(f"{text} {vision_data}")
audio = tts.generate_audio(response)
```

## Features

- Speech-to-Text with Whisper and Google Speech Recognition
- Text-to-Speech with Coqui TTS
- Vision AI with YOLO
- LLM integration with Ollama
- Conversation history with PandasDB
- WebSocket support for real-time audio

## Project Structure

```bash
oarc/
├── oarc/           # Package source code
├── tests/          # Test directory
├── pyproject.toml  # Project configuration
├── README.md       # Documentation
└── LICENSE         # License file
```

## Publishing to PyPI

```bash
# Ensure build tools are installed
python -m pip install --upgrade build twine

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```
