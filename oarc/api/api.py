#!/usr/bin/env python3
"""
Entry point for the OARC API server.

This module sets up the FastAPI application, integrates various tool APIs,
and defines routes for functionalities such as speech recognition, text-to-speech,
video processing, and agent interactions.
"""

import json

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from oarc.api.agent_api import AgentAPI
from oarc.api.agent_access import AgentAccess
from oarc.api.llm_prompt_api import LLMPromptAPI
from oarc.api.spell_loader import SpellLoader
from oarc.database.agent_storage import AgentStorage
from oarc.database.pandas_db import PandasDB
from oarc.promptModel import MultiModalPrompting
from oarc.speech import SpeechToText, SpeechToTextAPI, TextToSpeech, TextToSpeechAPI
from oarc.utils.decorators.singleton import singleton
from oarc.utils.log import log
from oarc.yolo.processor import YoloProcessor
from oarc.yolo.server_api import YoloServerAPI

@singleton
class API():
    """
    Main API class for the OARC system.

    This class initializes and configures the FastAPI application, sets up middleware,
    integrates various tool APIs, and defines routes for functionalities such as speech
    recognition, text-to-speech, video processing, and agent interactions.
    """


    def __init__(self):
        """
        Initialize the API class, set up the FastAPI application, and configure
        all necessary components such as middleware, routes, and tool APIs.
        """
        
        log.info("Initializing OARC API")
        
        log.info("Initializing SpellLoader")
        self.spellLoader = SpellLoader()
        
        log.info("Creating FastAPI application")
        self.app = FastAPI()
        self.setup_middleware()
        
        # Initialize all tools
        log.info("Initializing tool APIs")
        self.tool_apis = {}
        
        try:
            log.info("Initializing TextToSpeechAPI")
            self.tool_apis['tts'] = TextToSpeechAPI()
            
            log.info("Initializing SpeechToTextAPI")
            self.tool_apis['stt'] = SpeechToTextAPI()
            
            log.info("Initializing YoloAPI")
            self.tool_apis['yolo'] = YoloServerAPI()
            
            log.info("Initializing LLMPromptAPI")
            self.tool_apis['llm'] = LLMPromptAPI()
            
            log.info("Initializing AgentAPI")
            self.tool_apis['agent'] = AgentAPI()
            
            log.info("All tool APIs initialized successfully")
        except Exception as e:
            log.error(f"Error initializing tool APIs: {e}", exc_info=True)
            raise
            
        # Initialize YOLO processor for direct use in endpoints
        log.info("Initializing YoloProcessor")
        self.yolo_processor = YoloProcessor()
        
        # Include all routers
        log.info("Including API routers")
        for api_name, api in self.tool_apis.items():
            log.info(f"Adding router for {api_name}")
            self.app.include_router(api.router)
            
        self.setup_routes()
        
        # Initialize database
        log.info("Initializing database")
        self.pandas_db = PandasDB()
        
        log.info("OARC API initialization complete")
    
    def setup_middleware(self):
        """
        Configure middleware for the FastAPI application.

        This method sets up middleware components such as CORS to handle cross-origin requests,
        ensuring secure and flexible communication between the API and external clients.
        """
        log.info("Setting up CORS middleware")
        # Set up CORS middleware to allow cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # For development; restrict this in production
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )
        log.info("CORS middleware configured")
        
    def setup_routes(self):
        """
        Define and configure the API routes.

        This method sets up the endpoints for various functionalities provided by the OARC API,
        including speech recognition, text-to-speech, video processing, agent interactions, 
        and WebSocket-based streaming. Each route is associated with a specific handler function 
        to process incoming requests and return appropriate responses.
        """
        log.info("Setting up API routes")
        
        # Core
        @self.app.get("/")
        async def root():
            """
            Root endpoint for the API.

            This endpoint serves as a basic health check and welcome message for the OARC API.
            It provides a simple JSON response to confirm that the API is running and accessible.
            """
            log.info("Root endpoint accessed")
            return {"message": "Welcome to OARC API"}

        @self.app.post("/api/speech/recognize")
        async def recognize_speech(audio: UploadFile):
            """
            Endpoint to process an uploaded audio file for speech recognition.

            This endpoint accepts an audio file, processes it using the SpeechToText
            module, and returns the recognized text as a response.
            """
            log.info(f"Speech recognition request received: {audio.filename}")
            try:
                stt = SpeechToText()
                text = await stt.recognizer(audio.file)
                log.info(f"Speech recognition successful: '{text}'")
                return {"text": text}
            except Exception as e:
                log.error(f"Speech recognition failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")

        @self.app.post("/api/chat/complete") 
        async def chat_completion(text: str):
            """
            Handles POST requests to the "/api/chat/complete" endpoint for generating
            chat completions using the MultiModalPrompting module.

            Args:
                text (str): The input text prompt for generating a chat response.

            Returns:
                dict: A dictionary containing the generated chat response.

            Raises:
                HTTPException: If an error occurs during chat completion, an HTTP 500
                response is returned with details about the failure.

            Logs:
                - Logs the received chat completion request.
                - Logs the successful completion of the chat response.
                - Logs the generated chat response.
                - Logs errors and exceptions if the chat completion fails.
            """
            log.info(f"Chat completion request received: '{text}'")
            try:
                prompt = MultiModalPrompting()
                response = await prompt.send_prompt(text)
                log.info("Chat completion successful")
                log.info(f"Chat completion response: '{response}'")
                return {"response": response}
            except Exception as e:
                log.error(f"Chat completion failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

        @self.app.post("/api/speech/synthesize")
        async def synthesize_speech(text: str):
            """
            Asynchronous endpoint to synthesize speech from the provided text.

            Args:
                text (str): The input text to be converted into speech.

            Returns:
                dict: A dictionary containing the synthesized audio data.

            Logs:
                - Logs the receipt of a speech synthesis request with the input text.
                - Logs a success message upon successful speech synthesis.
                - Logs an error message with exception details if speech synthesis fails.

            Raises:
                HTTPException: If speech synthesis fails, an HTTP 500 error is raised with the error details.
            """
            log.info(f"Speech synthesis request received: '{text}'")
            try:
                tts = TextToSpeech()
                audio = await tts.process_tts_responses(text)
                log.info("Speech synthesis successful")
                return {"audio": audio}
            except Exception as e:
                log.error(f"Speech synthesis failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

        @self.app.websocket("/ws/audio-stream")
        async def audio_websocket(websocket: WebSocket):
            """
            WebSocket endpoint for real-time audio streaming.

            This endpoint allows clients to establish a WebSocket connection
            for sending and receiving audio data in real-time. The server
            processes the incoming audio stream and can optionally send
            responses back to the client.

            Args:
            websocket (WebSocket): The WebSocket connection instance.

            Logs:
            - Logs the initiation and acceptance of the WebSocket connection.
            - Logs the size of received audio data in bytes.
            - Logs disconnection or errors during the WebSocket session.

            Raises:
            WebSocketDisconnect: If the WebSocket connection is closed by the client.
            Exception: For any other errors during the WebSocket session.
            """
            log.info("WebSocket connection initiated for audio streaming")
            try:
                await websocket.accept()
                log.info("WebSocket connection accepted")
                while True:
                    audio_data = await websocket.receive_bytes()
                    log.info(f"Received {len(audio_data)} bytes of audio data")
                    # TODO Process streaming audio
            except WebSocketDisconnect:
                log.info("WebSocket disconnected")
            except Exception as e:
                log.error(f"WebSocket error: {e}", exc_info=True)
                
        @self.app.post("/api/yolo/stream")
        async def yolo_stream(video: UploadFile):
            """
            Endpoint to process a video file using YOLO for object detection.

            This endpoint accepts a video file, processes it using the YOLO object detection
            module, and returns the detection results. The results include information about
            detected objects, such as their labels, confidence scores, and bounding box coordinates.

            Args:
            video (UploadFile): The uploaded video file to be processed.

            Returns:
            dict: A dictionary containing the detection results, including object details.

            Logs:
            - Logs the receipt of the video processing request with the file name.
            - Logs a success message upon successful YOLO processing.
            - Logs an error message with exception details if YOLO processing fails.

            Raises:
            HTTPException: If YOLO processing fails, an HTTP 500 error is raised with the error details.
            """
            log.info(f"YOLO video processing request received: {video.filename}")
            try:
                results = await self.yolo_processor.process_video(video.file)
                log.info("YOLO video processing successful")
                return {"results": results}
            except Exception as e:
                log.error(f"YOLO video processing failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"YOLO video processing failed: {str(e)}")

        # WebUI
        @self.app.post("/api/agent/load")
        async def load_agent(request: AgentAccess):
            """
            Endpoint to load an agent configuration.

            This endpoint accepts a request containing the agent ID, retrieves the
            corresponding agent configuration from storage, and returns it as a response.

            Args:
            request (AgentAccess): The request object containing the agent ID to load.

            Returns:
            dict: A dictionary containing the status of the operation and the loaded agent configuration.

            Logs:
            - Logs the receipt of the agent load request with the agent ID.
            - Logs a success message upon successfully loading the agent configuration.
            - Logs an error message with exception details if the agent loading fails.

            Raises:
            HTTPException: If the agent loading fails, an HTTP 500 error is raised with the error details.
            """
            log.info(f"Agent load request received: {request.agent_id}")
            try:
                agent_storage = AgentStorage()
                agent = await agent_storage.load_agent(request.agent_id)
                log.info(f"Agent {request.agent_id} loaded successfully")
                return {"status": "success", "agent": agent}
            except Exception as e:
                log.error(f"Agent load failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/agents/list")
        async def list_agents():
            """
            Endpoint to retrieve a list of all available agents.

            This endpoint queries the agent storage to fetch and return a list of
            all agents currently available in the system. The response includes
            details about each agent, such as their IDs and configurations.

            Logs:
            - Logs the receipt of the agent list request.
            - Logs the number of agents retrieved and their details.
            - Logs an error message with exception details if the retrieval fails.

            Raises:
            HTTPException: If the agent list retrieval fails, an HTTP 500 error is raised with the error details.
            """
            log.info("Agent list request received")
            try:
                agent_storage = AgentStorage()
                agents = await agent_storage.list_available_agents()
                log.info(f"Retrieved {len(agents)} agents")
                log.info(f"Agents retrieved: {agents}")
                return {"agents": agents}
            except Exception as e:
                log.error(f"Agent list retrieval failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat/stream")
        async def chat_stream(websocket: WebSocket):
            """
            WebSocket endpoint for real-time chat streaming.

            This endpoint allows clients to establish a WebSocket connection for
            real-time chat interactions. Clients can send messages of various types
            (e.g., text, audio, vision), and the server processes these messages
            and sends appropriate responses back to the client.

            Args:
            websocket (WebSocket): The WebSocket connection instance.

            Logs:
            - Logs the initiation and acceptance of the WebSocket connection.
            - Logs the type of each received message and its processing status.
            - Logs disconnection or errors during the WebSocket session.

            Raises:
            WebSocketDisconnect: If the WebSocket connection is closed by the client.
            Exception: For any other errors during the WebSocket session.
            """
            log.info("Chat stream WebSocket connection requested")
            try:
                await websocket.accept()
                log.info("Chat stream WebSocket connection accepted")
                
                while True:
                    data = await websocket.receive_json()
                    log.info(f"Received chat message: {data['type']}")
                    
                    # Handle different message types
                    if data["type"] == "text":
                        log.info("Processing text message")
                        response = await self.handle_text_message(data)
                    elif data["type"] == "audio":
                        log.info("Processing audio message")
                        response = await self.handle_audio_message(data)
                    elif data["type"] == "vision":
                        log.info("Processing vision message")
                        response = await self.handle_vision_message(data)
                    else:
                        log.warning(f"Unknown message type: {data['type']}")
                        response = {"error": "Unknown message type"}
                        
                    await websocket.send_json(response)
                    log.info("Response sent back to client")
                    
            except WebSocketDisconnect:
                log.info("Chat stream WebSocket disconnected")
            except Exception as e:
                log.error(f"Chat stream error: {e}", exc_info=True)

        @self.app.post("/api/conversation/export")
        async def export_conversation(agent_id: str):
            """
            Endpoint to export the conversation history for a specific agent.

            This endpoint retrieves the conversation history associated with the
            specified agent ID from the database and exports it in JSON format.

            Args:
                agent_id (str): The unique identifier of the agent whose conversation
                    history is to be exported.

            Returns:
                dict: A dictionary containing the exported conversation history in JSON format.

            Logs:
                - Logs the receipt of the conversation export request with the agent ID.
                - Logs a success message upon successfully exporting the conversation history.
                - Logs an error message with exception details if the export operation fails.

            Raises:
                HTTPException: If the conversation export fails, an HTTP 500 error is raised
                    with the error details.
            """
            log.info(f"Conversation export request for agent: {agent_id}")
            try:
                pandas_db = PandasDB()
                conversation = pandas_db.export_conversation(format="json")
                log.info(f"Conversation exported successfully for agent: {agent_id}")
                return {"conversation": json.loads(conversation)}
            except Exception as e:
                log.error(f"Conversation export failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        log.info("All API routes setup complete")