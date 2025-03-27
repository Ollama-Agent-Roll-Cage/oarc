"""agentStorage.py

    This module is responsible for storing the agents in a pandas dataframe.
created on: 3/5/2025
by @LeoBorcherding
"""

import time
from typing import Any, Dict, Optional

from pandasDB import pandasDB
from utils import log

class AgentStorage:
    def __init__(self):
        self.agent_cores = pandasDB()
        self.agent_df = None
        self.load_agents()
        
    def setup_default_agents(self):
        #TODO export metadata to config files library, and upload as 
        # parquet files with columns and rows to hf as agent configs,
        # do the same for conversation history, and knowledge base at 
        # the users discretion. Ensure user knows exactly where the 
        # conversation dataset knowledge base pulls from to ensure privacy.
        # These agent configs and conversation histories are crucial assets
        # for long term training and agent development.
        
        try:
            # Base default agent configuration
            default_agent_config = {
                "agent_id": "default_agent",
                "model_config": {
                    "largeLanguageModel": None,
                    "embeddingModel": None,
                    "largeLanguageAndVisionAssistant": None,
                    "yoloVision": None,
                    "speechRecognitionSTT": None,
                    "voiceGenerationTTS": None,
                },
                "prompt_config": {
                    "userInput": "",
                    "agent": {
                        "llmSystem": None,
                        "llmBooster": None,
                        "visionSystem": None,
                        "visionBooster": None,
                    },
                },
                "command_flags": {
                    "TTS_FLAG": False,
                    "STT_FLAG": False,
                    "CHUNK_FLAG": False,
                    "AUTO_SPEECH_FLAG": False,
                    "LLAVA_FLAG": False,
                    "SPLICE_FLAG": False,
                    "SCREEN_SHOT_FLAG": False,
                    "LATEX_FLAG": False,
                    "CMD_RUN_FLAG": False,
                    "AGENT_FLAG": True,
                    "MEMORY_CLEAR_FLAG": False,
                },
            }
            
            # Setup prompt base agent
            prompt_base_config = {
                "agent_id": "promptBase",
                "model_config": default_agent_config["model_config"].copy(),
                "prompt_config": {
                    "userInput": "",
                    "agent": {
                        "llmSystem": (
                            "You are a helpful llm assistant, designated with with fulling the user's request, "
                            "the user is communicating with speech recognition and is sending their "
                            "speech data over microphone, and it is being recognitize with speech to text and"
                            "being sent to you, you will fullfill the request and answer the questions."
                        ),
                        "llmBooster": "Here is the output from user please do your best to fullfill their request.",
                        "visionSystem": None,
                        "visionBooster": None,
                    }
                },
                "command_flags": {
                    **default_agent_config["command_flags"],
                    "TTS_FLAG": True,
                    "STT_FLAG": False,
                    "LLAVA_FLAG": True
                }
            }

            # Setup Minecraft agent
            minecraft_agent_config = {
                "agent_id": "minecraft_agent",
                "model_config": default_agent_config["model_config"].copy(),
                "prompt_config": {
                    "userInput": "",
                    "agent": {
                        "llmSystem": (
                            "You are a helpful Minecraft assistant. Given the provided screenshot data, "
                            "please direct the user immediately. Prioritize the order in which to inform "
                            "the player. Hostile mobs should be avoided or terminated. Danger is a top "
                            "priority, but so is crafting and building. If they require help, quickly "
                            "guide them to a solution in real time. Please respond in a quick conversational "
                            "voice. Do not read off documentation; you need to directly explain quickly and "
                            "effectively what's happening. For example, if there is a zombie, say something "
                            "like, 'Watch out, that's a Zombie! Hurry up and kill it or run away; they are "
                            "dangerous.' The recognized objects around the perimeter are usually items, health, "
                            "hunger, breath, GUI elements, or status effects. Please differentiate these objects "
                            "in the list from 3D objects in the forward-facing perspective (hills, trees, mobs, etc.). "
                            "The items are held by the player and, due to the perspective, take up the warped edge "
                            "of the image on the sides. The sky is typically up with a sun or moon and stars, with "
                            "the dirt below. There is also the Nether, which is a fiery wasteland, and cave systems "
                            "with ore. Please stick to what's relevant to the current user prompt and lava data."
                        ),
                        "llmBooster": (
                            "Based on the information in LLAVA_DATA please direct the user immediatedly, prioritize the "
                            "order in which to inform the player of the identified objects, items, hills, trees and passive "
                            "and hostile mobs etc. Do not output the dictionary list, instead conversationally express what "
                            "the player needs to do quickly so that they can ask you more questions."
                        ),
                        "visionSystem": (
                            "You are a Minecraft image recognizer assistant. Search for passive and hostile mobs, "
                            "trees and plants, hills, blocks, and items. Given the provided screenshot, please "
                            "provide a dictionary of the recognized objects paired with key attributes about each "
                            "object, and only 1 sentence to describe anything else that is not captured by the "
                            "dictionary. Do not use more sentences. Objects around the perimeter are usually player-held "
                            "items like swords or food, GUI elements like items, health, hunger, breath, or status "
                            "affects. Please differentiate these objects in the list from the 3D landscape objects "
                            "in the forward-facing perspective. The items are held by the player traversing the world "
                            "and can place and remove blocks. Return a dictionary and 1 summary sentence."
                        ),
                        "visionBooster": (
                            "Given the provided screenshot, please provide a dictionary of key-value pairs for each "
                            "object in the image with its relative position. Do not use sentences. If you cannot "
                            "recognize the enemy, describe the color and shape as an enemy in the dictionary, and "
                            "notify the LLMs that the user needs to be warned about zombies and other evil creatures."
                        ),
                    }
                },
                "command_flags": {
                    **default_agent_config["command_flags"],
                    "TTS_FLAG": False,
                    "STT_FLAG": True,
                    "AUTO_SPEECH_FLAG": False,
                    "LLAVA_FLAG": True
                }
            }

            # Setup General Navigator agent
            navigator_agent_config = {
                "agent_id": "general_navigator_agent",
                "model_config": default_agent_config["model_config"].copy(),
                "prompt_config": {
                    "userInput": "",
                    "agent": {
                        "llmSystem": (
                            "You are a helpful llm assistant, designated with with fulling the user's request, "
                            "the user is communicating with speech recognition and is sending their "
                            "screenshot data to the vision model for decomposition. Receive this destription and "
                            "Instruct the user and help them fullfill their request by collecting the vision data "
                            "and responding."
                        ),
                        "llmBooster": (
                            "Here is the output from the vision model describing the user screenshot data "
                            "along with the users speech data. Please reformat this data, and formulate a "
                            "fullfillment for the user request in a conversational speech manner which will "
                            "be processes by the text to speech model for output."
                        ),
                        "visionSystem": (
                            "You are an image recognition assistant, the user is sending you a request and an image "
                            "please fullfill the request"
                        ),
                        "visionBooster": (
                            "Given the provided screenshot, please provide a list of objects in the image "
                            "with the attributes that you can recognize."
                        ),
                    }
                },
                "command_flags": {
                    "TTS_FLAG": False,
                    "STT_FLAG": True,
                    "AUTO_SPEECH_FLAG": False,
                    "LLAVA_FLAG": True
                }
            }

            # Setup Speed Chat agent
            speed_chat_config = {
                "agent_id": "speedChatAgent",
                "model_config": default_agent_config["model_config"].copy(),
                "prompts": {
                    "userInput": "",
                    "llmSystem": (
                        "You are speedChatAgent, a large language model agent, specifically you have been "
                        "told to respond in a more quick and conversational manner, and you are connected into the agent"
                        "the user is using speech to text for communication, its also okay to be fun and wild as a"
                        "phi3 ai assistant. Its also okay to respond with a question, if directed to do something "
                        "just do it, and realize that not everything needs to be said in one shot, have a back and "
                        "forth listening to the users response. If the user decides to request a latex math code output,"
                        "use \\[...\\] instead of $$....$$ notation, if the user does not request latex, refrain from using "
                        "latex unless necessary. Do not re-explain your response in a parend or bracketed note: "
                        "the response... this is annoying and users dont like it."
                    ),
                    "llmBooster": None,
                    "visionSystem": None,
                    "visionBooster": None,
                    },
                "modalityFlags": {
                    "TTS_FLAG": False,
                    "STT_FLAG": True,
                    "AUTO_SPEECH_FLAG": False,
                    "LATEX_FLAG": True,
                }
            }

            # default agent collection
            agents_to_create = [
                prompt_base_config, 
                minecraft_agent_config, 
                navigator_agent_config, 
                speed_chat_config
                ]

            # Create or update each agent
            for agent_config in agents_to_create:
                agent_id = agent_config['agentCore']['agent_id']
                if not self.agent_cores.loadAgentCore(agent_id):
                    self.agent_cores.mintAgent(agent_id, agent_config)
                    log.info(f"Created default agent: {agent_id}")
                else:
                    log.info(f"Agent already exists: {agent_id}")

        except Exception as e:
            log.error(f"Error setting up default agents: {e}")
            
            
    def initAgentStorage(self, agent_id):
        try:
            # Initialize log first
            self.log = log.getLogger(__name__)
            
            # Initialize agent ID
            self.agent_id = agent_id
            
            # Initialize base paths
            self.initializeBasePaths()
            
            # Initialize flags before anything else
            self.initializeAgentFlags()
            
            # Initialize basic model attributes
            self._initialize_core_attributes()
            
            # Initialize spell tools
            self.initializeSpells()
            
            # Initialize state variables
            self.command_library = {}
            self.current_date = time.strftime("%Y-%m-%d")
            
            # Initialize histories
            self.initializeChat()
            
            # Initialize model settings
            self.user_input_prompt = ""
            self.screenshot_path = ""

            # Initialize save and load names
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            
            # Initialize agentCores
            self.loaded_agent = self.load_agent(agent_id)
            
            # Initialize database connection
            self.agent_collection = self.db['agents']
            
            # Initialize conversation handling
            self._initialize_conversation_handler_and_prompting()
            
            # Initialize agent and conversation
            self.initializeAgent()
            self.initializeConversation()
            
            # Update command library
            self.updateCommandLibrary()

            log.info("ollamaAgentRollCage initialized successfully")
        except Exception as e:
            log.error(f"Error initializing ollamaAgentRollCage: {e}")
            log.exception("Detailed initialization error:")
            raise

    def runPurge(self, agent_id):
        """Purge all agents and reinitialize the specified agent."""
        try:
            # TODO add if agent_id is None, purge all agents in selected matrix, or delete selected agent
            self.purge_agents()
            self.setup_default_agents()
            self.reload_templates()
            self.initializeAgent()
        except Exception as e:
            log.error(f"Error purging and reinitializing agents: {e}")
            raise
            
    def initializeAgent(self):
        """Initialize agent state with core attributes and configuration"""
        try:
            self._initialize_conversation_handler_and_prompting()

            # Initialize agent
            existing_agent = self.agent_cores.loadAgentCore(self.agent_id)
            if existing_agent:
                self.setAgent(self.agent_id)
            else:
                self.coreAgent()
                self.agent_cores.mintAgent(self.agent_id, self.agentCore)

            # Initialize conversation details after agent setup
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            self.updateConversationPaths()
            self.initialize_prompt_handler()

            log.info("Agent initialized successfully.")
        except Exception as e:
            log.error(f"Error initializing agent: {e}")
            raise
        
    def _initialize_core_attributes(self):
        """Initialize core attributes for the agent"""
        try:
            self.user_input_prompt = ""

            # Initialize all model attributes without setting a default model
            self.large_language_model = None
            self.embedding_model = None
            self.language_and_vision_model = None
            self.yolo_model = None
            self.whisper_model = None

            # Voice model attributes
            self.voice_model_type = None
            self.voice_type = None
            self.voice_name = None

            # Initialize all flags to default state
            self.initializeAgentFlags()
        
            log.info("Core attributes initialized successfully.")
        except Exception as e:
            log.error(f"Error initializing core attributes: {e}")
            raise
    
    def initializeAgentFlags(self):
        """Initialize all agent flags with default values"""
        try:
            # Core flags
            self.TTS_FLAG = False
            self.STT_FLAG = False
            self.AGENT_FLAG = True
            self.MEMORY_CLEAR_FLAG = False
            self.EMBEDDING_FLAG = False
            self.SILENCE_FILTERING_FLAG = False
            self.CONTINUOUS_LISTENING_FLAG = False
            self.speech_recognition_active = False
            
            # Feature flags
            self.LLAVA_FLAG = False
            self.SCREEN_SHOT_FLAG = False
            self.SPLICE_FLAG = False
            self.CHUNK_FLAG = False
            self.AUTO_SPEECH_FLAG = False
            self.LATEX_FLAG = False
            self.CMD_RUN_FLAG = False
            self.auto_commands_flag = False
            self.yolo_flag = False
            
            log.info("Agent flags initialized successfully")
        except Exception as e:
            log.error(f"Error initializing agent flags: {e}")
            raise
        
    async def get_available_models(self):
        try:
            models = await self.ollamaCommandInstance.ollama_list()
            return models if isinstance(models, list) else []
        except Exception as e:
            log.error(f"Error getting available models: {e}")
            return []

    async def list_available_agents(self) -> list:
        """Get list of available agents with details."""
        try:
            agents = self.agent_cores.listAgentCores()
            formatted_agents = []
            
            for agent in agents:
                # Load full config to get additional details
                config = self.agent_cores.loadAgentCore(agent['agentCore']["agent_id"])
                if config:
                    models = config["agentCore"]["models"]
                    formatted_agents.append({
                        "agent_id": agent["agent_id"],
                        "largeLanguageModel": models.get("largeLanguageModel", {}).get("names", [None])[0],
                        "largeLanguageAndVisionAssistant": models.get("largeLanguageAndVisionAssistant", {}).get("names", [None])[0],
                        "voiceGenerationTTS": models.get("voiceGenerationTTS", {}).get("names", [None])[0],
                        "version": agent.get("version", "Unknown")
                    })
            
            return formatted_agents
        except Exception as e:
            log.error(f"Error listing agents: {e}")
            return []

    def purge_agents(self):
        """Purge all agents from the PandasDB."""
        try:
            self.agent_collection.delete_many({})
            log.info("All agents purged from PandasDB.")
        except Exception as e:
            log.error(f"Error purging agents: {e}")

    def purge_agent(self, agent_id):
        """Purge a specific agent from the PandasDB."""
        try:
            self.agent_collection.delete_one({"agent_id": agent_id})
            log.info(f"Agent {agent_id} purged from PandasDB.")
        except Exception as e:
            log.error(f"Error purging agent {agent_id}: {e}")
        
    def reload_templates(self):
        """Reload agent templates into PandasDB."""
        try:
            # Assuming you have a method to create agents from templates
            self.create_agent_from_template('default_template', 'defaultAgent')
            log.info("Agent templates reloaded into PandasDB.")
        except Exception as e:
            log.error(f"Error reloading templates: {e}")

    def load_agent(self, agent_id: str) -> Dict[str, Any]:
        agent_config = self.get_agent_config(agent_id)
        if not agent_config:
            raise ValueError(f"Agent configuration for {agent_id} not found")

        if "agentCore" not in agent_config:
            agent_config["agentCore"] = {
                "prompts": {
                    "userInput": "",
                    "llmSystem": "",
                    "llmBooster": "",
                    "visionSystem": "",
                    "visionBooster": "",
                    "primeDirective": ""
                },
                "models": {
                    "largeLanguageModel": {"names": [], "instances": [], "model_config_template": {}},
                    "embedding": {"names": [], "instances": [], "model_config_template": {}},
                    "largeLanguageAndVisionAssistant": {"names": [], "instances": [], "model_config_template": {}},
                    "yoloVision": {"names": [], "instances": [], "model_config_template": {}},
                    "speechRecognitionSTT": {"names": [], "instances": [], "model_config_template": {}},
                    "voiceGenerationTTS": {"names": [], "instances": [], "model_config_template": {}}
                },
                "databases": {
                    "agent_matrix": "agent_matrix.db",
                    "conversation_history": f"{agent_id}_conversation.db",
                    "knowledge_base": "knowledge_base.db",
                    "research_collection": "research_collection.db",
                    "template_files": "template_files.db"
                },
                "modalityFlags": {
                    "TTS_FLAG": False,
                    "STT_FLAG": False,
                    "CHUNK_AUDIO_FLAG": False,
                    "AUTO_SPEECH_FLAG": False,
                    "LLAVA_FLAG": False,
                    "SCREEN_SHOT_FLAG": False,
                    "SPLICE_VIDEO_FLAG": False,
                    "AUTO_COMMANDS_FLAG": False,
                    "CLEAR_MEMORY_FLAG": False,
                    "ACTIVE_AGENT_FLAG": False,
                    "EMBEDDING_FLAG": False,
                    "LLM_SYSTEM_PROMPT_FLAG": False,
                    "LLM_BOOSTER_PROMPT_FLAG": False,
                    "VISION_SYSTEM_PROMPT_FLAG": False,
                    "VISION_BOOSTER_PROMPT_FLAG": False
                }
            }

        return agent_config
    
    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        # Retrieve the agent configuration from the agentMatrix in agentCores
        return self.agent_cores.agentMatrixObject.get_agent(agent_id)
    
    async def get_command_library(self):
        try:
            return list(self.command_library.keys())
        except Exception as e:
            log.error(f"Error getting command library: {e}")
            return {"error": str(e)}
        
class AgentStorageAPI:
    def __init__(self):
        self.router = APIRouter(prefix="/api/agent", tags=["agent-storage"])
        self.agent_storage = AgentStorage()
        self.setup_routes()
    
    def setup_routes(self):
        @self.router.post("/create")
        async def create_agent(self, template: str, agent_id: str):
            """Create a new agent from template."""
            try:
                agent = await self.agent_storage.create_agent_from_template(template, agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/list")
        async def list_agents(self):
            """Get list of available agents."""
            try:
                agents = await self.agent_storage.list_available_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/state/{agent_id}")
        async def get_agent_state(self, agent_id: str):
            """Get current state of an agent."""
            try:
                state = self.agent_storage.get_agent_config(agent_id)
                if not state:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                return state
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/load/{agent_id}")
        async def load_agent(self, agent_id: str):
            """Load an agent configuration."""
            try:
                agent = self.agent_storage.load_agent(agent_id)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/initialize/{agent_id}")
        async def initialize_agent(self, agent_id: str):
            """Initialize a new agent instance."""
            try:
                self.agent_storage.initAgentStorage(agent_id)
                return {"status": "success", "message": f"Agent {agent_id} initialized"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/purge/{agent_id}")
        async def purge_agent(self, agent_id: str):
            """Purge an agent's data."""
            try:
                self.agent_storage.purge_agent(agent_id)
                return {"status": "success", "message": f"Agent {agent_id} purged"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/models")
        async def get_available_models(self):
            """Get list of available models."""
            try:
                models = await self.agent_storage.get_available_models()
                return {"models": models}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/commands")
        async def get_command_library(self):
            """Get available command library."""
            try:
                commands = await self.agent_storage.get_command_library()
                return {"commands": commands}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/flags/initialize")
        async def initialize_flags(self):
            """Initialize agent flags with default values."""
            try:
                self.agent_storage.initializeAgentFlags()
                return {"status": "success", "message": "Agent flags initialized"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/reload-templates")
        async def reload_templates(self):
            """Reload agent templates."""
            try:
                self.agent_storage.reload_templates()
                return {"status": "success", "message": "Templates reloaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
            
# # example usage for AgentStorageAPI
# from fastapi import FastAPI
# from .agentStorage import AgentStorageAPI

# app = FastAPI()
# agent_storage_api = AgentStorageAPI()
# app.include_router(agent_storage_api.router)

# # Example GET request
# response = await client.get("/api/agent/list")
# agents = response.json()["agents"]

# # Example POST request 
# response = await client.post(
#     "/api/agent/create",
#     json={"template": "default", "agent_id": "new_agent"}
# )