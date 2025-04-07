""" oarcFastApi.py

This module contains the FastAPI implementation for the OARC Digital Data Highway. This API is used to interact 
with OARC chatbots, including sending and receiving chat messages, executing commands, toggling speech 
recognition, speech generation, vision, and any other models supported in the OARC framework.

    written by: @LBorcherding
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel
from pprint import pformat

from oarc.wizards.__ollama_chatbot_wizard_OLD import ollamaAgentRollCage

# Ensure this is only called once in the main entry point of your app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent duplicate handlers
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, (dict, list)):
            record.msg = f"\n{pformat(record.msg, indent=2, width=80)}"
        return super().format(record)
    
class ModelRequest(BaseModel):
    model_name: str
    agent_id: str
    
class CreateAgentRequest(BaseModel):
    agent_id: str
    template_name: str
    
class PurgeRequest(BaseModel):
    agent_id: str
    
class rollCage:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.agentCage: Dict[str, ollamaAgentRollCage] = {}  # Ensure this is a dictionary of ollamaAgentRollCage instances
        self.audio_streams: Dict[str, asyncio.Queue] = {}

    async def initialize(self):
        """Async initialization method"""
        try:
            await self.initialize_default_agent()
        except Exception as e:
            logger.error(f"Error during async initialization: {e}")
            raise

    async def initialize_default_agent(self):
        """Initialize a default agent on startup"""
        try:
            if "default" not in self.agentCage:
                return await self.initialize_ollama_agent_roll_cage("default")
            else:
                logger.info("Default agent already initialized.")
                return True
        except Exception as e:
            logger.error(f"Error initializing default agent: {e}")
            return False
        
    async def connect(self, websocket: WebSocket, agent_id: str, connection_type: str = "chat"):
        await websocket.accept()
        
        if agent_id not in self.active_connections:
            self.active_connections[agent_id] = {}
            
        self.active_connections[agent_id][connection_type] = websocket
        
        if connection_type == "audio":
            self.audio_streams[agent_id] = asyncio.Queue()
            
        logger.info(f"New {connection_type} connection for agent {agent_id}")

    async def disconnect(self, agent_id: str, connection_type: str):
        if agent_id in self.active_connections:
            if connection_type in self.active_connections[agent_id]:
                del self.active_connections[agent_id][connection_type]
                
            if not self.active_connections[agent_id]:
                del self.active_connections[agent_id]
                
            if connection_type == "audio" and agent_id in self.audio_streams:
                del self.audio_streams[agent_id]
                
        logger.info(f"Disconnected {connection_type} for agent {agent_id}")

    async def initialize_ollama_agent_roll_cage(self, agent_id: str):
        """Initialize or get existing chatbot instance"""
        if agent_id not in self.agentCage:
            try:
                loaded_agent = ollamaAgentRollCage(agent_id=agent_id)
                self.agentCage[agent_id] = loaded_agent  # Correctly add the instance to the dictionary
                logger.info(f"Initialized new chatbot for agent {agent_id}")
                return True
            except Exception as e:
                logger.error(f"Error initializing chatbot for agent {agent_id}: {e}")
                logger.exception("Detailed traceback:")
                return False
        return True

    async def getAvailableModels(self, agent_id: str = None):
        """Get list of available models"""
        try:
            all_models = []
            if agent_id and agent_id in self.agentCage:
                loaded_agent = self.agentCage[agent_id]
                models = await loaded_agent.get_available_models()
                all_models.extend(models)
            else:
                for loaded_agent in self.agentCage.values():
                    models = await loaded_agent.get_available_models()
                    all_models.extend(models)
            unique_models = list(set(all_models))  # Remove duplicates
            logger.info("Available models:\n%s", pformat(unique_models, indent=2))
            return unique_models
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
        
    async def setModel(self, agent_id: str, model_name: str) -> bool:
        try:
            if agent_id not in self.agentCage:
                logger.error(f"Agent {agent_id} not found")
                return False

            agent = self.agentCage[agent_id]
            agent.set_model(model_name)
            logger.info(f"Model set to {model_name} for agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting model for agent {agent_id}: {e}")
            return False
        
    async def setAgent(self, agent_id: str) -> bool:
        try:
            if agent_id not in self.agentCage:
                success = await self.initialize_ollama_agent_roll_cage(agent_id)
                if not success:
                    logger.error(f"Failed to initialize agent {agent_id}")
                    return False

            agent = self.agentCage[agent_id]
            agent.setAgent(agent_id)
            logger.info(f"Agent {agent_id} set successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting agent {agent_id}: {e}")
            return False

    async def getCommandLibrary(self, agent_id: str = None):
        """Get command library"""
        try:
            if agent_id and agent_id in self.agentCage:
                chatbot = self.agentCage['agent_core'][agent_id]
                return await chatbot.get_command_library()
            else:
                # Get commands from default agent
                for agent_id, chatbot in self.agentCage.items():
                    return await chatbot.get_command_library()
            return []
        except Exception as e:
            logger.error(f"Error getting command library: {e}")
            return []

    async def listAvailableAgents(self):
        """Get list of available agents"""
        try:
            agents = []
            for agent_id, chatbot in self.agentCage.items():
                agent_state = chatbot.get_agent_state() if hasattr(chatbot, 'get_agent_state') else {}
                agents.append({
                    "id": agent_id,
                    "state": agent_state
                })
            return agents
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return []

    async def purgeAgentMatrix(self, agent_id):
        """Purge the agent matrix"""
        try:
            if agent_id in self.agentCage:
                agent = self.agentCage[agent_id]
                # purge singular agent from the agent matrix
                agent.runPurge(agent_id)
                logger.info(f"Agent {agent_id} purged successfully")
                return True
            else:
                agents = self.agentCage
                # purge all agents from the agent matrix
                agents.runPurge()
                logger.error(f"Agent {agent_id} not found in agentCage")
                return False
            
        except Exception as e:
            logger.error(f"Error purging agent matrix: {e}")
            return False
        
# Initialize the connection manager
roll_cage = rollCage()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    logger.info("Starting up FastAPI application")
    try:
        # Initialize the digital data highway during startup
        await roll_cage.initialize()
        logger.info("Digital Data Highway initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Digital Data Highway: {e}")
        raise
    yield
    logger.info("Shutting down FastAPI application")

app = FastAPI(
    title="OARC Digital Data Highway API",
    description="API for interacting with OARC-based agents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/getAvailableAgents")
async def getAvailableAgents():
    """Get list of available agents"""
    try:
        agents = await roll_cage.listAvailableAgents()
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Error in get_available_agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getAvailableModels")
async def getAvailableModels():
    try:
        models = await roll_cage.getAvailableModels()
        if isinstance(models, list):
            # Clean and format model names
            cleaned_models = sorted(set(m.split(':')[0] for m in models))
            return {"models": cleaned_models}
        return {"models": []}
    except Exception as e:
        logger.error(f"Error in get_available_models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setModel")
async def setModel(request: ModelRequest):
    """Set the model for the agent"""
    try:
        model_name = request.model_name
        agent_id = request.agent_id
        logger.info(f"Received request to set model: {model_name} for agent: {agent_id}")

        success = await roll_cage.setModel(agent_id, model_name)
        if success:
            return {"status": "success", "model": model_name}
        else:
            raise HTTPException(status_code=400, detail="Failed to set model")
    except Exception as e:
        logger.error(f"Error setting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/setAgent")
async def setAgent(request: Dict[str, Any]):
    agent_id = request.get("agent_id")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    success = await roll_cage.setAgent(agent_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to set agent {agent_id}")
    return {"status": "success", "agent_id": agent_id}
    
@app.post("/deleteAgent")
async def deleteAgent(request: PurgeRequest):
    """Purge and reinitialize the agent"""
    try:
        roll_cage.purgeAgentMatrix(request.agent_id)
    except Exception as e:
        logger.error(f"Error purging and initializing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/deleteAgentMatrix")
async def deleteAgentMatrix(request: PurgeRequest):
    """Purge and reinitialize the agent"""
    try:
        roll_cage.purgeAgentMatrix()
    except Exception as e:
        logger.error(f"Error purging and initializing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/getAvailableModels")
async def getCommandLibrary():
    """Get available commands"""
    try:
        commands = await roll_cage.getCommandLibrary()
        return {"commands": commands}
    except Exception as e:
        logger.error(f"Error in get_command_library: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/deleteAgent")
async def deleteAgent(request: PurgeRequest):
    """Purge and reinitialize the agent"""
    try:
        roll_cage.purgeAgentMatrix(request.agent_id)
    except Exception as e:
        logger.error(f"Error purging and initializing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/deleteAgentMatrix")
async def deleteAgentMatrix(request: PurgeRequest):
    """Purge and reinitialize the agent"""
    try:
        roll_cage.purgeAgentMatrix()
    except Exception as e:
        logger.error(f"Error purging and initializing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for chat interactions"""
    try:
        success = await roll_cage.initialize_ollama_agent_roll_cage(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
            
        await roll_cage.connect(websocket, agent_id)
        logger.info("connection open")
        
        while True:
            try:
                data = await websocket.receive_json()
                loaded_agent = roll_cage.agentCage.get(agent_id)
                if loaded_agent:
                    # Properly await the command check
                    result = await loaded_agent.commandPromptCheck(data.get("content"))
                    if result:
                        # Ensure result is JSON serializable
                        if isinstance(result, dict):
                            result = {k: v for k, v in result.items() if not callable(v)}
                        await websocket.send_json({
                            "type": "command_result",
                            "content": result
                        })
                else:
                    logger.error(f"No loaded_agent instance found for agent {agent_id}")
            except WebSocketDisconnect:
                await roll_cage.disconnect(agent_id, "chat")
                break
                
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}")
        if websocket.client_state.CONNECTED:
            await websocket.close()
            
@app.websocket("/audio-stream/{agent_id}")
async def audio_stream(websocket: WebSocket, agent_id: str):
    """Handle audio streaming for an agent"""
    try:
        success = await roll_cage.initialize_ollama_agent_roll_cage(agent_id)
        if not success:
            logger.error(f"Failed to initialize agent {agent_id}")
            await websocket.close(code=1011)
            return
            
        await roll_cage.connect(websocket, agent_id, "audio")
        logger.info("connection open")
        
        chatbot = roll_cage.agentCage.get(agent_id)
        if not chatbot:
            logger.error(f"No chatbot instance found for agent {agent_id}")
            await websocket.close(code=1011)
            return

        while True:
            try:
                data = await websocket.receive_text()
                audio_data = json.loads(data)
                
                if audio_data['audio_type'] == 'stt':
                    await chatbot.process_stt_audio(audio_data['audio_data'])
                elif audio_data['audio_type'] == 'tts':
                    await chatbot.process_tts_audio(audio_data['audio_data'])
                else:
                    logger.warning(f"Unknown audio type: {audio_data['audio_type']}")
            except WebSocketDisconnect:
                await roll_cage.disconnect(agent_id, "audio")
                break
                
    except json.JSONDecodeError:
        logger.error("Invalid JSON received in audio stream")
    except Exception as e:
        logger.error(f"Error in audio stream: {e}")
        if websocket.client_state.CONNECTED:
            await websocket.close()
    finally:
        await roll_cage.disconnect(agent_id, "audio")

@app.post("/createAgent")
async def create_agent(request: CreateAgentRequest):
    """Create a new agent from a template"""
    try:
        agent_id = request.agent_id
        template_name = request.template_name
        logger.info(f"Received request to create agent: {agent_id} from template: {template_name}")

        agent_config = roll_cage.create_agent_from_template(template_name, agent_id)
        if agent_config:
            return {"status": "success", "agent_id": agent_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to create agent")
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2020)