#!/usr/bin/env python3
"""
Entry point for the OARC API server.

This module sets up the FastAPI application, integrates various tool APIs,
and defines routes for functionalities such as speech recognition, text-to-speech,
video processing, and agent interactions.
"""

import json

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from oarc.decorators.log import log
from oarc.api import LLMPromptAPI, AgentAPI, SpellLoader, AgentAccess
from oarc.database import AgentStorage, PandasDB
from oarc.promptModel import multiModalPrompting
from oarc.speechToSpeech import speechToText, textToSpeech, TextToSpeechAPI, SpeechToTextAPI
from oarc.yoloProcessor import YoloAPI, YoloProcessor

@log()
class oarcAPI():
    def __init__(self):
        self.spellLoader = SpellLoader()
        self.app = FastAPI()
        self.setup_middleware()
        
        # Initialize all tool APIs
        self.tool_apis = {
            'tts': TextToSpeechAPI(),
            'stt': SpeechToTextAPI(),
            'yolo': YoloAPI(),
            'llm': LLMPromptAPI(),
            'agent': AgentAPI()
        }
        
        # Include all tool routers
        for api in self.tool_apis.values():
            self.app.include_router(api.router)
            
        self.setup_routes()
        
        # Initialize PandasDB
        self.db = PandasDB()
        
    def setup_routes(self):
        # Main API routes
        @self.app.get("/")
        async def root():
            return {"message": "Welcome to OARC API"}

        @self.app.post("/api/speech/recognize")
        async def recognize_speech(audio: UploadFile):
            stt = speechToText()
            text = await stt.recognizer(audio.file)
            return {"text": text}

        @self.app.post("/api/chat/complete") 
        async def chat_completion(text: str):
            prompt = multiModalPrompting()
            response = await prompt.send_prompt(text)
            return {"response": response}

        @self.app.post("/api/speech/synthesize")
        async def synthesize_speech(text: str):
            tts = textToSpeech()
            audio = await tts.process_tts_responses(text)
            return {"audio": audio}

        @self.app.websocket("/ws/audio-stream")
        async def audio_websocket(websocket: WebSocket):
            await websocket.accept()
            while True:
                audio_data = await websocket.receive_bytes()
                # Process streaming audio
                
        @self.app.post("/api/yolo/stream")
        async def yolo_stream(video: UploadFile):
            yolo = YoloProcessor()
            results = await yolo.process_video(video.file)
            return {"results": results}

        # WebUI specific routes
        @self.app.post("/api/agent/load")
        async def load_agent(request: AgentAccess):
            try:
                agent_storage = AgentStorage()
                agent = await agent_storage.load_agent(request.agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/agents/list")
        async def list_agents():
            try:
                agent_storage = AgentStorage()
                agents = await agent_storage.list_available_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat/stream")
        async def chat_stream(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_json()
                    
                    # Handle different message types
                    if data["type"] == "text":
                        response = await self.handle_text_message(data)
                    elif data["type"] == "audio":
                        response = await self.handle_audio_message(data)
                    elif data["type"] == "vision":
                        response = await self.handle_vision_message(data)
                        
                    await websocket.send_json(response)
                    
            except WebSocketDisconnect:
                log.info("WebSocket disconnected")

        @self.app.post("/api/conversation/export")
        async def export_conversation(agent_id: str):
            try:
                db = PandasDB()
                conversation = db.export_conversation(format="json")
                return {"conversation": json.loads(conversation)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
# # Example WebUI connection
# import websockets
# import json

# async def connect_to_agent():
#     uri = "ws://localhost:2020/api/chat/stream"
#     async with websockets.connect(uri) as websocket:
#         # Load agent
#         await websocket.send(json.dumps({
#             "type": "load_agent",
#             "agent_id": "my_agent"
#         }))
        
#         # Send message
#         await websocket.send(json.dumps({
#             "type": "text",
#             "content": "Hello!",
#             "metadata": {
#                 "audio": {"stt": "audio_data"},
#                 "vision": {"yolo": ["detection_results"]}
#             }
#         }))
        
#         # Receive response
#         response = await websocket.recv()
#         print(json.loads(response))