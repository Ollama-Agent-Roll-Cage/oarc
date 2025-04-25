# How to Start the OARC API

## 1. Environment Setup

Set up your environment variables in Windows PowerShell. Begin by configuring your `MODEL_GIT` path:

```powershell
# Define paths for model libraries and APIs
setx MODEL_GIT "MODEL_LIB path"
setx OARC_API "OARC_API path"
setx OARC_MODEL_GIT "MODEL_LIB/OARC_MODEL_GIT"

setx OLLAMA_API_BASE "http://127.0.0.1:11434"
setx OLLAMA_ORIGINS *
setx OLLAMA_MODELS "MODEL_LIB/OLLAMA_MODELS"

setx YOLO_MODEL_GIT "MODEL_LIB/YOLO_MODEL_GIT"

# Placeholder for additional tools
setx DEPTH_ANYTHING_V2_PATH
setx UNSLOTH_FINETUNED_MODELS

setx HF_HOME "MODEL_GIT/HF_HOME"
```

## 2. Directory Structure

Create the necessary directories for the OARC API:

```powershell
# Base directories for models and generated files
mkdir "$env:OARC_MODEL_GIT\coqui\XTTS-v2_c3po"
mkdir "$env:OARC_MODEL_GIT\coqui\voice_reference_pack\c3po"
mkdir "$env:OARC_MODEL_GIT\generated"
mkdir "$env:OARC_MODEL_GIT\whisper"
mkdir "$env:OARC_MODEL_GIT\huggingface"
mkdir "$env:OARC_MODEL_GIT\agentFiles\publicAgents"
mkdir "$env:OARC_MODEL_GIT\agentFiles\ignoredAgents"
```

## 3. Test Script

Create a Python script to test the API:

```python
# test_oarc.py
from oarc import oarcAPI
import uvicorn

def main():
    # Initialize the API
    api = oarcAPI()
    
    # Run the FastAPI server
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

1. Start the Ollama service.
2. Open a terminal in VS Code.
3. Run the test script:

```bash
python test_oarc.py
```

## 5. Testing Endpoints

Use `curl` to test the API endpoints:

```bash
# Test the root endpoint
curl http://localhost:2020/

# Test the Text-to-Speech (TTS) endpoint
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
  - `fastapi`
  - `uvicorn`
  - `numpy`
  - `torch`
  - `TTS`
  - `python-multipart`
  - `websockets`

## 8. Model Setup

Ensure your models are organized as follows:

```bash
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

- Access FastAPI docs at: [http://localhost:2020/docs](http://localhost:2020/docs)
- Check logs in the terminal.
- Monitor model loading progress.
- Verify environment variables are correctly set.

## 10. Common Issues

- Ensure `OARC_MODEL_GIT` is set and accessible.
- Verify the Ollama service is running.
- Confirm model files are in the correct locations.
- Monitor GPU memory usage if using CUDA.
