# How to start the OARC API

## 1. Environment Setup

First, set up your environment variables in Windows PowerShell, starting with your MODEL_GIT:

```powershell
# Select a location preferably at the base level of one of your hard drives, this will act as our source model git for hugging face, ollama, github, and others.
setx MODEL_GIT "MODEL_GIT path"
setx OARC_API "OARC_API path"
setx OARC_MODEL_GIT "MODEL_GIT/OARC_MODEL_GIT"

setx OLLAMA_API_BASE "http://127.0.0.1:11434"
setx OLLAMA_ORIGINS *
setx OLLAMA_MODELS "MODEL_GIT/OLLAMA_MODELS"

setx YOLO_MODEL_GIT "MODEL_GIT/YOLO_MODEL_GIT"

#TODO SETUP LATER FOR EXTERIOR TOOLS LIKE UNSLOTH UI, GESTURE BOX, AND DREAM CAM
setx DEPTH_ANYTHING_V2_PATH
setx UNSLOTH_FINETUNED_MODELS

setx HF_HOME "MODEL_GIT/HF_HOME"
```

## 2. Directory Structure

Create the required directories:

```powershell
# Create base directories
mkdir "$env:OARC_MODEL_GIT\coqui\XTTS-v2_c3po"
mkdir "$env:OARC_MODEL_GIT\coqui\voice_reference_pack\c3po"
mkdir "$env:OARC_MODEL_GIT\generated"
mkdir "$env:OARC_MODEL_GIT\whisper"
mkdir "$env:OARC_MODEL_GIT\huggingface"
mkdir "$env:OARC_MODEL_GIT\agentFiles\publicAgents"
mkdir "$env:OARC_MODEL_GIT\agentFiles\ignoredAgents"
```

## 3. Test Script

Create a test script to run the API:

```python
# test_oarc.py
from oarc import oarcAPI
import uvicorn

def main():
    # Initialize API
    api = oarcAPI()
    
    # Run FastAPI server
    uvicorn.run(
        api.app, 
        host="0.0.0.0", 
        port=2020,
        reload=True
    )

if __name__ == "__main__":
    main()
```

## 4. Running the API

1. Start Ollama service first
2. Open terminal in VS Code
3. Run the test script:
```bash
python test_oarc.py
```

## 5. Testing Endpoints

Test the API endpoints using curl:

```bash
# Test root endpoint
curl http://localhost:2020/

# Test TTS endpoint
curl -X POST "http://localhost:2020/tts/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello! I am C-3PO!", "voice_name": "c3po"}'

# List available voices
curl "http://localhost:2020/tts/voices"
```

## 6. Available Endpoints

- `/` - Root endpoint
- `/tts/synthesize` - Text-to-Speech synthesis
- `/tts/voices` - List available TTS voices
- `/api/speech/recognize` - Speech recognition
- `/api/chat/complete` - Chat completion
- `/api/yolo/stream` - YOLO video processing
- `/ws/audio-stream` - WebSocket for real-time audio

## 7. Requirements

- Python 3.8+
- Ollama installed and running
- Required Python packages:
  - fastapi
  - uvicorn
  - numpy
  - torch
  - TTS
  - python-multipart
  - websockets

## 8. Model Setup

Ensure your models are in the correct locations:

```
%OARC_MODEL_GIT%/
├── coqui/
│   ├── XTTS-v2_c3po/
│   │   ├── config.json
│   │   ├── model.pth
│   │   └── reference.wav
│   └── voice_reference_pack/
│       └── c3po/
│           └── clone_speech.wav
├── whisper/
├── generated/
└── huggingface/
```

## 9. Debugging

- Check FastAPI docs at: http://localhost:2020/docs
- View logs in the terminal
- Monitor model loading progress
- Check environment variables are set correctly

## 10. Common Issues

- Ensure OARC_MODEL_GIT is set and accessible
- Verify Ollama is running
- Check model files exist in correct locations
- Monitor GPU memory usage if using CUDA