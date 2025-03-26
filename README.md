# OARC (ollamaAgentRollCage)

## Building a Multimodal Agent

### Basic Setup
```python
from oarc import oarcAPI
from oarc.speechToSpeech import textToSpeech, speechToText
from oarc.promptModel import multiModalPrompting
from oarc.yoloProcessor import YoloProcessor
from oarc.pandasDB import PandasDB

# Initialize API and components
api = oarcAPI()
db = PandasDB()

# Setup speech components
stt = speechToText()
tts = textToSpeech(developer_tools_dict={
    'current_dir': '/path/to/current',
    'parent_dir': '/path/to/parent',
    'speech_dir': '/path/to/speech',
    'recognize_speech_dir': '/path/to/stt',
    'generate_speech_dir': '/path/to/tts',
    'tts_voice_ref_wav_pack_path_dir': '/path/to/voices'
}, voice_type="finetuned", voice_name="c3po")

# Setup vision components
yolo = YoloProcessor(
    model_path="yolov8n.pt",
    conf_threshold=0.5
)

# Initialize multimodal prompting
llm = multiModalPrompting()
```

### Loading an Agent
```python
# Create/load agent configuration
agent_config = {
    "agent_id": "minecraft_agent",
    "models": {
        "largeLanguageModel": "llama2",
        "largeLanguageAndVisionAssistant": "llava",
        "yoloVision": "yolov8n"
    },
    "prompts": {
        "llmSystem": "You are a Minecraft assistant...",
        "visionSystem": "Analyze the game environment..."
    },
    "flags": {
        "TTS_FLAG": True,
        "STT_FLAG": True,
        "LLAVA_FLAG": True,
        "YOLO_FLAG": True
    }
}

# Store agent in PandasDB
db.storeAgent(agent_config)

# Load agent for use
agent = db.setAgent("minecraft_agent")
```

### Real-time Multimodal Loop
```python
async def run_agent():
    while True:
        # Speech recognition
        if agent.flags["STT_FLAG"]:
            audio = await stt.listen()
            text = stt.recognizer(audio)

        # Vision processing 
        if agent.flags["YOLO_FLAG"]:
            frame = yolo.capture_screen()
            vision_data, detections = yolo.process_frame(frame, return_detections=True)

        # LLaVA vision-language processing
        if agent.flags["LLAVA_FLAG"]:
            llava_response = await llm.llava_prompt(
                text,
                vision_data,
                agent.prompts["visionSystem"]
            )

        # LLM response generation
        response = await llm.send_prompt(
            agent,
            db.conversation_handler,
            {
                "text": text,
                "vision": llava_response,
                "detections": detections
            }
        )

        # Text-to-speech output
        if agent.flags["TTS_FLAG"]:
            audio = await tts.process_tts_responses(response)
            tts.play_audio(audio)

        # Store in conversation history
        await db.store_message(
            role="user",
            content=text,
            metadata={
                "audio": {"stt": audio},
                "vision": {
                    "llava": llava_response,
                    "yolo": detections
                }
            }
        )
```

### Command Library Usage
```python
# Define agent commands
commands = {
    "voice on": lambda: agent.set_flag("TTS_FLAG", True),
    "voice off": lambda: agent.set_flag("TTS_FLAG", False),
    "vision on": lambda: agent.set_flag("LLAVA_FLAG", True),
    "yolo on": lambda: agent.set_flag("YOLO_FLAG", True),
    "swap model": lambda model: llm.swap(model)
}

# Command processing
async def handle_command(text):
    if text.startswith("/"):
        cmd = text[1:].split()[0]
        if cmd in commands:
            await commands[cmd]()
            return True
    return False
```

### WebUI Integration
```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        data = await websocket.receive_json()
        
        if data["type"] == "audio":
            # Handle speech input
            text = stt.recognizer(data["audio"])
            
        elif data["type"] == "vision":
            # Handle vision input
            frame = np.array(data["image"])
            detections = yolo.process_frame(frame)
            
        # Generate response
        response = await llm.send_prompt(agent, text, detections)
        
        # Send response with audio
        audio = await tts.process_tts_responses(response)
        await websocket.send_json({
            "text": response,
            "audio": audio.tolist()
        })
```

## Environment Setup
```bash
# Required environment variables
export OARC_MODEL_GIT="/path/to/models"
export OLLAMA_API_BASE="http://localhost:11434"
```