""" ollamaChatbotWizard.py

        ollama_agent_roll_cage, is an opensource toolkit api for speech to text, text to speech 
    commands, multi-modal agent building with local LLM api's, including tools such as ollama, 
    transformers, keras, as well as closed source api endpoint integration such as openAI, 
    anthropic, groq, and more!
    
    ===============================================================================================
    
        OARC has its own chatbot agent endpoint which you can find in the fastAPI at the bottom 
    of this file. This custom api is what empowers oarc to bundle/wrap AI models & other api endpoints 
    into one cohesive agent including the following models;
    
    Ollama -
        Llama: Text to Text 
        LLaVA: Text & Image to Text
        
    CoquiTTS -
        XTTSv2: Non-Emotive Transformer Text to Speech, With Custom Finetuned Voices
        Bark: Emotional Diffusion Text to Speech Model
        
    F5_TTS -
        Emotional TTS model, With Custom Finetuned Voices (coming soon) 
        
    YoloVX - 
        Object Recognition within image & video streams, providing bounding box location data.
        Supports YoloV6, YoloV7, YoloV8, and beyond! I would suggest YoloV8 seems to have the 
        highest accuracy. 
        
    Whisper -
        Speech to Text recognition, allowing the user to interface with any model directly
        using a local whisper model.
        
    Google python speech-recognition -
        A free alternative Speech to Text offered by google, powered by their api servers, this
        STT api is a good alternative especially if you need to offload the speech recognition 
        to the google servers due to your computers limitations.
        
    Musetalk -
        A local lypc sync, Avatar Image & Audio to Video model. Allowing the chatbot agent to
        generate in real time, the Avatar lypc sync for the current chatbot agent in OARC.
        
    ===============================================================================================
    
        This software was designed by Leo Borcherding with the intent of creating 
    an easy to use ai interface for anyone, through Speech to Text and Text to Speech.
        
        With ollama_agent_roll_cage we can provide hands free access to LLM data. 
    This has a host of applications and I want to bring this software to users 
    suffering from blindness/vision loss, and children suffering from austism spectrum 
    disorder as way for learning and expanding communication and speech. 
    
        The C3PO ai is a great imaginary friend! I could envision myself 
    talking to him all day telling me stories about a land far far away! 
    This makes learning fun and accessible! Children would be directly 
    rewarded for better speech as the ai responds to subtle differences 
    in language ultimately educating them without them realizing it.

    Development for this software was started on: 4/20/2024 
    All users have the right to develop and distribute ollama agent roll cage,
    with proper citation of the developers and repositories. Be weary that some
    software may not be licensed for commerical use.

    By: Leo Borcherding, 4/20/2024
        on github @ 
            leoleojames1/ollama_agent_roll_cage
"""

# =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-= =-= =-= =-= =-= =-= =-= 
# =-= =-= =-= =-= =-= =-= General System Packages =-= =-= =-= =-= =-= =-= =-= =-= =-= =-= =-= =-= 

import os
import re
import time
import json
import asyncio
from typing import Dict, Any, Optional
import logging
import websockets
from pprint import pformat

# database packages
from pymongo import MongoClient

# =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-= 
# =-= =-= =-= =-= =-= =-= Custom Oarc Class Modules =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-= 

__all__ = ['ollamaAgentRollCage']

# ai tools
# TODO these need to be imported properly
from wizardSpellBook.publicWand.ollamaAddOnLibrary import ollamaCommands
from wizardSpellBook.publicWand.speechToSpeech import speechToText
from wizardSpellBook.publicWand.speechToSpeech import textToSpeech

# file handling and construction
# TODO these need to be imported properly
from wizardSpellBook.publicWand.writeModelfile import model_write_class
from wizardSpellBook.publicWand.createConvertModel import create_convert_manager
from wizardSpellBook.publicWand.directoryManager import directory_manager_class
from wizardSpellBook.publicWand.dataSetManipulator import data_set_constructor
from wizardSpellBook.publicWand.dataSetManipulator import screen_shot_collector

# networking imports
# TODO these need to be imported properly
from wizardSpellBook.publicWand.nodeCustomMethods import FileSharingNode

# agentCores utils
# TODO these need to be imported properly
from wizardSpellBook.publicWand.agentCores.agentCores import agentCores
from wizardSpellBook.publicWand.agentCores.agentMatrix import agentMatrix
from wizardSpellBook.publicWand.promptModel import multiModalPrompting
from wizardSpellBook.publicWand.promptModel import conversationHandler

# =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-= 
# =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-=  =-= =-= =-= =-= =-= =-= 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, (dict, list)):
            record.msg = f"\n{pformat(record.msg, indent=2, width=80)}"
        return super().format(record)
    
class ollamaAgentRollCage:
    def __init__(self, agent_id):
        try:
            # Initialize logger first
            self.logger = logging.getLogger(__name__)
            
            # Initialize agent ID
            self.agent_id = agent_id
            
            # Initialize base paths
            self.initializeBasePaths()
            
            # Initialize flags before anything else
            self.initializeAgentFlags()
            
            # Initialize basic model attributes
            self._initialize_core_attributes()
            
            # Initialize ollama commands
            self.ollamaCommandInstance = ollamaCommands()
            
            # Initialize spell tools
            self.initializeSpells()
            
            # Initialize state variables
            self.command_library = {}
            self.current_date = time.strftime("%Y-%m-%d")
            self.agent_matrix = agentMatrix()
            self.agent_cores = agentCores()
            
            # Initialize histories
            self.initializeChat()
            
            # Initialize model settings
            self.user_input_prompt = ""
            self.screenshot_path = ""

            # Initialize save and load names
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            
            # Initialize database connection
            self.client = MongoClient('mongodb://localhost:27017/')
            self.db = self.client['ollama_db']
            self.conversation_handler = conversationHandler(self.db, self.agent_id)
            self.agent_collection = self.db['agents']
            
            # Initialize conversation handling
            self._initialize_conversation_handler_and_prompting()
            
            # Initialize agent and conversation
            self.initializeAgent()
            self.initializeConversation()
            
            # Update command library
            self.updateCommandLibrary()

            logger.info("ollamaAgentRollCage initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ollamaAgentRollCage: {e}")
            logger.exception("Detailed initialization error:")
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
            logger.error(f"Error purging and reinitializing agents: {e}")
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
                self.agent_cores.mintAgent(self.agent_id, self.agent_core)

            # Initialize conversation details after agent setup
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            self.updateConversationPaths()
            self.initialize_prompt_handler()

            logger.info("Agent initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
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
        
            logger.info("Core attributes initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing core attributes: {e}")
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
            
            logger.info("Agent flags initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent flags: {e}")
            raise
        
    async def get_available_models(self):
        try:
            models = self.ollamaCommandInstance.ollama_list()
            return models if isinstance(models, list) else []
        except Exception as e:
            logging.error(f"Error getting available models: {e}")
            return []

    async def list_available_agents(self) -> list:
        """Get list of available agents with details."""
        try:
            agents = self.agent_cores.listAgentCores()
            formatted_agents = []
            
            for agent in agents:
                # Load full config to get additional details
                config = self.agent_cores.loadAgentCore(agent['agent_core']["agent_id"])
                if config:
                    models = config["agent_core"]["models"]
                    formatted_agents.append({
                        "agent_id": agent["agent_id"],
                        "largeLanguageModel": models.get("largeLanguageModel", {}).get("names", [None])[0],
                        "largeLanguageAndVisionAssistant": models.get("largeLanguageAndVisionAssistant", {}).get("names", [None])[0],
                        "voiceGenerationTTS": models.get("voiceGenerationTTS", {}).get("names", [None])[0],
                        "version": agent.get("version", "Unknown")
                    })
            
            return formatted_agents
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return []

    def purge_agents(self):
        """Purge all agents from the MongoDB."""
        try:
            self.agent_collection.delete_many({})
            logger.info("All agents purged from MongoDB.")
        except Exception as e:
            logger.error(f"Error purging agents: {e}")

    def purge_agent(self, agent_id):
        """Purge a specific agent from the MongoDB."""
        try:
            self.agent_collection.delete_one({"agent_id": agent_id})
            logger.info(f"Agent {agent_id} purged from MongoDB.")
        except Exception as e:
            logger.error(f"Error purging agent {agent_id}: {e}")
        
    def reload_templates(self):
        """Reload agent templates into MongoDB."""
        try:
            # Assuming you have a method to create agents from templates
            self.create_agent_from_template('default_template', 'defaultAgent')
            logger.info("Agent templates reloaded into MongoDB.")
        except Exception as e:
            logger.error(f"Error reloading templates: {e}")
               
    async def get_command_library(self):
        try:
            return list(self.command_library.keys())
        except Exception as e:
            logger.error(f"Error getting command library: {e}")
            return {"error": str(e)}
    
    def initializeBasePaths(self):
        """Initialize the base file path structure"""
        try:
            # Get base directories
            self.current_dir = os.getcwd()
            self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
            
            # Get model_git_dir from environment variable
            model_git_dir = os.getenv('OARC_MODEL_GIT')
            if not model_git_dir:
                raise EnvironmentError(
                    "OARC_MODEL_GIT environment variable not set. "
                    "Please set it to your model git directory path."
                )
            
            # Initialize base path structure
            self.pathLibrary = {
                # Main directories
                'current_dir': self.current_dir,
                'parent_dir': self.parent_dir,
                'model_git_dir': model_git_dir,
                
                # Chatbot Wand directories
                'public_chatbot_base_wand_dir': os.path.join(self.current_dir, 'publicWand'),
                'ignored_chatbot_custom_wand_dir': os.path.join(self.current_dir, 'ignoredWand'),
                
                # Agent directories
                'ignored_agents_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredAgents'),
                'agent_files_dir': os.path.join(self.parent_dir, 'agentFiles', 'publicAgents'),
                'ignored_agentfiles': os.path.join(self.parent_dir, 'agentFiles', 'ignoredAgentfiles'),
                'public_agentfiles': os.path.join(self.parent_dir, 'agentFiles', 'publicAgentfiles'),
                
                # Pipeline directories
                'ignored_pipeline_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline'),
                'llava_library_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'llavaLibrary'),
                'conversation_library_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'conversationLibrary'),
                
                # Data constructor directories
                'image_set_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'dataConstructor', 'imageSet'),
                'video_set_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'dataConstructor', 'videoSet'),
                
                # Speech directories
                'speech_library_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary'),
                'recognize_speech_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'recognizeSpeech'),
                'generate_speech_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'generateSpeech'),
                'tts_voice_ref_wav_pack_dir': os.path.join(self.parent_dir, 'agentFiles', 'ignoredPipeline', 'speechLibrary', 'publicVoiceReferencePack'),
            }
            logger.info("Base paths initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing base paths: {e}")
            raise 

    def initializeSpells(self):
        """Initialize all spell classes for the chatbot wizard"""
        try:
            # Get directory data
            self.directory_manager_class = directory_manager_class()
            # Get data
            self.screen_shot_collector_instance = screen_shot_collector(self.pathLibrary)
            # Splice data
            self.data_set_video_process_instance = data_set_constructor(self.pathLibrary)
            # Write model files
            self.model_write_class_instance = model_write_class(self.pathLibrary)
            # Create model manager
            self.create_convert_manager_instance = create_convert_manager(self.pathLibrary)
            # Peer2peer node
            self.FileSharingNode_instance = FileSharingNode(host="127.0.0.1", port=9876)
            # TTS processor (initialize as None, will be created when needed)
            self.tts_processor_instance = None
            
            logger.info("Spells initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing spells: {e}")
            raise
    
    def initializeChat(self):
        """ a method to initilize the chatbot agent conversation
        """
        # initialize chat history
        self.chat_history = []
        self.llava_history = []
        
        # loaded agent
        self.loaded_agent = {}
        
        # # TODO -> Direct ollama api access, currently unused 
        # self.url = "http://localhost:11434/api/chat"

        # # Setup chat_history
        # self.headers = {'Content-Type': 'application/json'}
        
    def initializeConversation(self):
        try:
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            
            # Update conversation paths after initializing defaults
            self.updateConversationPaths()
            logger.info("Conversation initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            raise
        
    def updateConversationPaths(self):
        """Update conversation-specific paths"""
        try:
            agent_conversation_dir = os.path.join(self.pathLibrary['conversation_library_dir'], self.agent_id)
            os.makedirs(agent_conversation_dir, exist_ok=True)
            logger.info("Conversation paths updated successfully.")
        except Exception as e:
            logger.error(f"Error updating conversation paths: {e}")
            raise
        
    def initialize_prompt_handler(self):
        try:
            self.conversation_handler = conversationHandler(self.db, self.agent_id)
            logger.info("Conversation handler initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing conversation handler: {e}")
            raise

    async def store_message(self, role, content, session_id=None, metadata=None):
        try:
            document = await self.conversation_handler.store_message(role, content, session_id, metadata)
            logger.info(f"Stored message: {document}")
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            raise

    async def get_conversation_history(self, session_id=None, limit=100):
        try:
            history = await self.conversation_handler.get_conversation_history(session_id, limit)
            logger.info(f"Retrieved conversation history: {history}")
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            raise

    async def save_conversation(self, save_name, session_id):
        try:
            await self.conversation_handler.save_conversation(save_name, session_id)
            logger.info(f"Saved conversation with name: {save_name}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            raise

    async def load_conversation(self, save_name):
        try:
            conversation = await self.conversation_handler.load_conversation(save_name)
            logger.info(f"Loaded conversation with name: {save_name}")
            return conversation
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            raise

    async def clear_history(self, session_id=None):
        try:
            await self.conversation_handler.clear_history(session_id)
            logger.info(f"Cleared conversation history for session_id: {session_id}")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            raise
        
    def coreAgent(self):
        # Define model configurations
        models_config = {
            "largeLanguageModel": {
                "names": [self.large_language_model] if self.large_language_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "embedding": {
                "names": [self.embedding_model] if self.embedding_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "largeLanguageAndVisionAssistant": {
                "names": [self.language_and_vision_model] if self.language_and_vision_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "yoloVision": {
                "names": [self.yolo_model] if self.yolo_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "speechRecognitionSTT": {
                "names": [self.whisper_model] if self.whisper_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "voiceGenerationTTS": {
                "names": [self.voice_name] if self.voice_name else [],
                "instances": [],
                "model_config_template": {
                    "voice_type": self.voice_type
                }
            }
        }

        # Define prompt configurations
        prompts_config = {
            "userInput": self.user_input_prompt,
            "llmSystem": "",
            "llmBooster": "",
            "visionSystem": "",
            "visionBooster": "",
            "primeDirective": ""
        }

        # Define database configurations
        # TODO FIX HOME AGENT MATRIX NAME
        database_config = {
            "agent_matrix": self.save_name,
            "conversation_history": f"{self.agent_id}_conversation.db",
            "knowledge_base": "knowledge_base.db",
            "research_collection": "research_collection.db",
            "template_files": "template_files.db"
        }

        # Define modality flags
        modality_flags = {
            "TTS_FLAG": self.TTS_FLAG,
            "STT_FLAG": self.STT_FLAG,
            "CHUNK_AUDIO_FLAG": self.CHUNK_FLAG,
            "AUTO_SPEECH_FLAG": self.AUTO_SPEECH_FLAG,
            "LLAVA_FLAG": self.LLAVA_FLAG,
            "SCREEN_SHOT_FLAG": self.SCREEN_SHOT_FLAG,
            "SPLICE_VIDEO_FLAG": self.SPLICE_FLAG,
            "AUTO_COMMANDS_FLAG": False,
            "CLEAR_MEMORY_FLAG": self.MEMORY_CLEAR_FLAG,
            "ACTIVE_AGENT_FLAG": self.AGENT_FLAG,
            "EMBEDDING_FLAG": self.EMBEDDING_FLAG,
            "LLM_SYSTEM_PROMPT_FLAG": False,
            "LLM_BOOSTER_PROMPT_FLAG": False,
            "VISION_SYSTEM_PROMPT_FLAG": False,
            "VISION_BOOSTER_PROMPT_FLAG": False
        }

        # Define evolutionary settings
        evolution_config = {
            "mutation": None,
            "pain": None,
            "hunger": None,
            "fasting": None,
            "rationalizationFactor": None
        }

        # Define special arguments
        special_config = {
            "blocks": None,
            "tokens": None,
            "layers": None,
            "temperature": 0.7,
            "context_window": 4096,
            "streaming": True,
            "top_p": 1.0,
            "top_k": 40,
            "stop_sequences": [],
            "max_tokens": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }

        # Create the complete agent core structure
        self.agent_core = {
            "agent_core": {
                "identifyers": {
                    "agent_id": self.agent_id,
                    "uid": None,
                    "template_version_info": {
                        "version": "1.0.0",
                        "compatible_versions": ["1.0.0"],
                        "last_updated": self.current_date,
                        "format_version": "1.0.0"
                    },
                    "creationDate": self.current_date,
                    "cpuNoiseHex": None,
                    "identity_info": {
                        "core_type": "oarc_native",
                        "origin_info": {
                            "source": "oarc_wizard",
                            "creation_date": self.current_date,
                            "collection_date": self.current_date
                        }
                    }
                },
                "models": models_config,
                "prompts": prompts_config,
                "databases": database_config,
                "modalityFlags": modality_flags,
                "evolutionarySettings": evolution_config,
                "specialArgs": special_config
            }
        }

    def setAgent(self, agent_id):
        """Load agent configuration from agentCores and update current state"""
        try:
            logger.info("Loading agent configuration for agent_id: %s", agent_id)
            # From agent cores, load the agent configuration for the selected agent_id
            agent_config = self.agent_cores.loadAgentCore(agent_id)
            # set id in config
            agent_config["agent_core"]["agent_id"] = agent_id 
            
            # Remove the redundant agent_id key if it exists
            if 'agent_id' in agent_config:
                del agent_config['agent_id']
            
            if not agent_config:
                raise ValueError(f"Agent {agent_id} not found")

            # Ensure models key is present
            if "models" not in agent_config:
                agent_config["agent_core"]["models"] = {
                    "largeLanguageModel": {"names": [], "instances": [], "model_config_template": {}},
                    "embedding": {"names": [], "instances": [], "model_config_template": {}},
                    "largeLanguageAndVisionAssistant": {"names": [], "instances": [], "model_config_template": {}},
                    "yoloVision": {"names": [], "instances": [], "model_config_template": {}},
                    "speechRecognitionSTT": {"names": [], "instances": [], "model_config_template": {}},
                    "voiceGenerationTTS": {"names": [], "instances": [], "model_config_template": {}}
                }
                logger.info("Added missing 'models' key to agent_config")

            # Ensure prompts key is present
            if "prompts" not in agent_config:
                agent_config["agent_core"]["prompts"] = {
                    "userInput": "",
                    "llmSystem": "",
                    "llmBooster": "",
                    "visionSystem": "",
                    "visionBooster": "",
                    "primeDirective": ""
                }
                logger.info("Added missing 'prompts' key to agent_config")

            # Ensure modalityFlags key is present
            if "modalityFlags" not in agent_config:
                agent_config["agent_core"]["modalityFlags"] = {
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
                logger.info("Added missing 'modalityFlags' key to agent_config")

            # Ensure conversation key is present
            if "conversation" not in agent_config:
                agent_config["agent_core"]["conversation"] = {
                    "save_name": "",
                    "load_name": ""
                }
                logger.info("Added missing 'conversation' key to agent_config")

            # Update agent state from configuration
            self.agent_id = agent_config["agent_core"]["agent_id"]
            # models
            self.large_language_model = agent_config["agent_core"]["models"]["largeLanguageModel"]["names"][0] if agent_config["agent_core"]["models"]["largeLanguageModel"]["names"] else None
            self.embedding_model = agent_config["agent_core"]["models"]["embedding"]["names"][0] if agent_config["agent_core"]["models"]["embedding"]["names"] else None
            self.language_and_vision_model = agent_config["agent_core"]["models"]["largeLanguageAndVisionAssistant"]["names"][0] if agent_config["agent_core"]["models"]["largeLanguageAndVisionAssistant"]["names"] else None
            self.yolo_model = agent_config["agent_core"]["models"]["yoloVision"]["names"][0] if agent_config["agent_core"]["models"]["yoloVision"]["names"] else None
            self.whisper_model = agent_config["agent_core"]["models"]["speechRecognitionSTT"]["names"][0] if agent_config["agent_core"]["models"]["speechRecognitionSTT"]["names"] else None
            self.voice_name = agent_config["agent_core"]["models"]["voiceGenerationTTS"]["names"][0] if agent_config["agent_core"]["models"]["voiceGenerationTTS"]["names"] else None
            self.voice_type = agent_config["agent_core"]["models"]["voiceGenerationTTS"]["model_config_template"].get("voice_type", None)
            # prompts
            self.user_input_prompt = agent_config["agent_core"]["prompts"]["userInput"]
            self.llmSystemPrompt = agent_config["agent_core"]["prompts"]["llmSystem"]
            self.llmBoosterPrompt = agent_config["agent_core"]["prompts"]["llmBooster"]
            self.visionSystemPrompt = agent_config["agent_core"]["prompts"]["visionSystem"]
            self.visionBoosterPrompt = agent_config["agent_core"]["prompts"]["visionBooster"]
            # flags
            self.LLM_SYSTEM_PROMPT_FLAG = agent_config["agent_core"]["modalityFlags"]["LLM_SYSTEM_PROMPT_FLAG"]
            self.LLM_BOOSTER_PROMPT_FLAG = agent_config["agent_core"]["modalityFlags"]["LLM_BOOSTER_PROMPT_FLAG"]
            self.VISION_SYSTEM_PROMPT_FLAG = agent_config["agent_core"]["modalityFlags"]["VISION_SYSTEM_PROMPT_FLAG"]
            self.VISION_BOOSTER_PROMPT_FLAG = agent_config["agent_core"]["modalityFlags"]["VISION_BOOSTER_PROMPT_FLAG"]
            self.TTS_FLAG = agent_config["agent_core"]["modalityFlags"]["TTS_FLAG"]
            self.STT_FLAG = agent_config["agent_core"]["modalityFlags"]["STT_FLAG"]
            self.CHUNK_FLAG = agent_config["agent_core"]["modalityFlags"]["CHUNK_AUDIO_FLAG"]
            self.AUTO_SPEECH_FLAG = agent_config["agent_core"]["modalityFlags"]["AUTO_SPEECH_FLAG"]
            self.LLAVA_FLAG = agent_config["agent_core"]["modalityFlags"]["LLAVA_FLAG"]
            self.SCREEN_SHOT_FLAG = agent_config["agent_core"]["modalityFlags"]["SCREEN_SHOT_FLAG"]
            self.SPLICE_FLAG = agent_config["agent_core"]["modalityFlags"]["SPLICE_VIDEO_FLAG"]
            self.AGENT_FLAG = agent_config["agent_core"]["modalityFlags"]["ACTIVE_AGENT_FLAG"]
            self.MEMORY_CLEAR_FLAG = agent_config["agent_core"]["modalityFlags"]["CLEAR_MEMORY_FLAG"]
            self.EMBEDDING_FLAG = agent_config["agent_core"]["modalityFlags"]["EMBEDDING_FLAG"]
            # conversation metadata
            self.save_name = agent_config["agent_core"]["conversation"]["save_name"]
            self.load_name = agent_config["agent_core"]["conversation"]["load_name"]

            # Update paths
            self.updateConversationPaths()
            logger.info(f"Agent {agent_id} loaded successfully:\n%s", pformat(agent_config, indent=2, width=80))

        except Exception as e:
            logger.error(f"Error loading agent {agent_id}: {e}")
            raise
            
    async def list_available_agents(self) -> list:
        """Get list of available agents with details."""
        try:
            agents = self.agent_cores.listAgentCores()
            formatted_agents = []
            
            for agent in agents:
                # Load full config to get additional details
                config = self.agent_cores.loadAgentCore(agent["agent_id"])
                if config:
                    models = config["models"]
                    formatted_agents.append({
                        "id": agent["agent_id"],
                        "llm": models.get("largeLanguageModel", {}).get("names", [None])[0],
                        "vision": models.get("largeLanguageAndVisionAssistant", {}).get("names", [None])[0],
                        "voice": models.get("voiceGenerationTTS", {}).get("names", [None])[0],
                        "version": agent.get("version", "Unknown")
                    })
            
            return formatted_agents
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return []

    def create_agent_from_template(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
        """Create a new agent from a template"""
        try:
            agent_config = self.agent_cores.mintAgent(
                agent_id=agent_id,
                model_config=custom_config.get("models") if custom_config else None,
                prompt_config=custom_config.get("prompts") if custom_config else None,
                command_flags=custom_config.get("modalityFlags") if custom_config else None
            )
            
            # Initialize the new agent
            self.setAgent(agent_id)
            return agent_config
            
        except Exception as e:
            logger.error(f"Error creating agent from template: {e}")
            raise

    def save_agent_state(self):
        try:
            # Create current state configuration matching agent_core structure
            current_state = {
                "agent_core": {
                    "identifyers": {
                        "agent_id": self.agent_id,
                        "creationDate": self.current_date
                    },
                    "models": {
                        "largeLanguageModel": {
                            "names": [self.large_language_model] if self.large_language_model else [],
                            "instances": []
                        },
                        "embedding": {
                            "names": [self.embedding_model] if self.embedding_model else [],
                            "instances": []
                        },
                        "largeLanguageAndVisionAssistant": {
                            "names": [self.language_and_vision_model] if self.language_and_vision_model else [],
                            "instances": []
                        },
                        "yoloVision": {
                            "names": [self.yolo_model] if self.yolo_model else [],
                            "instances": []
                        },
                        "speechRecognitionSTT": {
                            "names": [self.whisper_model] if self.whisper_model else [],
                            "instances": []
                        },
                        "voiceGenerationTTS": {
                            "names": [self.voice_name] if self.voice_name else [],
                            "model_config_template": {
                                "voice_type": self.voice_type
                            }
                        }
                    },
                    "prompts": {
                        "userInput": self.user_input_prompt,
                        "agent": {
                            "llmSystem": self.llmSystemPrompt,
                            "llmBooster": self.llmBoosterPrompt,
                            "visionSystem": self.visionSystemPrompt,
                            "visionBooster": self.visionBoosterPrompt
                        }
                    },
                    "modalityFlags": {
                        "TTS_FLAG": self.TTS_FLAG,
                        "STT_FLAG": self.STT_FLAG,
                        "CHUNK_AUDIO_FLAG": self.CHUNK_FLAG,
                        "AUTO_SPEECH_FLAG": self.AUTO_SPEECH_FLAG,
                        "LLAVA_FLAG": self.LLAVA_FLAG,
                        "SCREEN_SHOT_FLAG": self.SCREEN_SHOT_FLAG,
                        "SPLICE_VIDEO_FLAG": self.SPLICE_FLAG,
                        "AUTO_COMMANDS_FLAG": False,
                        "CLEAR_MEMORY_FLAG": self.MEMORY_CLEAR_FLAG,
                        "ACTIVE_AGENT_FLAG": self.AGENT_FLAG,
                        "EMBEDDING_FLAG": self.EMBEDDING_FLAG,
                        "LLM_SYSTEM_PROMPT_FLAG": self.LLM_SYSTEM_PROMPT_FLAG,
                        "LLM_BOOSTER_PROMPT_FLAG": self.LLM_BOOSTER_PROMPT_FLAG,
                        "VISION_SYSTEM_PROMPT_FLAG": self.VISION_BOOSTER_PROMPT_FLAG,
                        "VISION_BOOSTER_PROMPT_FLAG": self.VISION_BOOSTER_PROMPT_FLAG

                    }
                }
            }
            self.agent_cores.storeAgentCore(self.agent_id, current_state)
            logger.info(f"Saved agent state: {self.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            return False

    def load_from_json(self, load_name, large_language_model):
        """Load conversation history from JSON"""
        # Update load name and model
        self.load_name = load_name if load_name else f"conversation_{self.agent_id}_{self.current_date}"
        self.large_language_model = large_language_model
        
        # Update paths with new load name
        temp_save_name = self.save_name
        self.save_name = self.load_name
        self.updateConversationPaths()
        
        # Load conversation
        try:
            with open(self.pathLibrary['default_conversation_path'], 'r') as json_file:
                self.chat_history = json.load(json_file)
            print(f"Conversation loaded from: {self.pathLibrary['default_conversation_path']}")
            
            # Load from MongoDB
            self.chat_history = asyncio.run(self.conversation_handler.load_conversation(self.save_name))
            
        except Exception as e:
            print(f"Error loading conversation: {e}")
        finally:
            # Restore original save name
            self.save_name = temp_save_name
            self.updateConversationPaths()
            
    def save_to_json(self, save_name, large_language_model):
        """Save conversation history to JSON"""
        # Update save name and model
        self.save_name = save_name if save_name else f"conversation_{self.agent_id}_{self.current_date}"
        self.large_language_model = large_language_model
        
        # Update paths with new save name
        self.updateConversationPaths()
        
        # Save conversation
        try:
            with open(self.pathLibrary['default_conversation_path'], 'w') as json_file:
                json.dump(self.chat_history, json_file)
            print(f"Conversation saved to: {self.pathLibrary['default_conversation_path']}")
            
            # Save to MongoDB
            asyncio.run(self.conversation_handler.save_conversation(self.save_name, self.agent_id))
            
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def file_name_conversation_history_filter(self, input):
        # Use regex to replace all spaces with underscores and convert to lowercase
        output = re.sub(' ', '_', input).lower()
        return output
    
    async def commandPromptCheck(self, user_input_prompt):
        try:
            if not user_input_prompt:
                raise ValueError("User input prompt is None or empty")
            
            command_payload = None
            if user_input_prompt.startswith("/") or user_input_prompt.startswith("activate "):
                for command, command_details in self.command_library.items():
                    if user_input_prompt.startswith(command):
                        argsList = user_input_prompt[len(command):].strip().split()
                        command_payload = {
                            "command": command,
                            "method": command_details["method"],
                            "is_async": command_details.get("is_async", False),
                            "argsList": argsList
                        }
                        break
            else:
                # Not a command, send to multi modal prompting
                logger.info(f"⌛Loading Agent {self.agent_id} in progress⌛:\n%s", pformat(self.loaded_agent, indent=2, width=80))
                response = await self.multi_modal_prompting.send_prompt(
                    self.loaded_agent,
                    self.conversation_handler,
                    self.chat_history
                )
                return {"status": "success", "response": response}

            if command_payload:
                command = command_payload["command"]
                method = command_payload["method"]
                argsList = command_payload["argsList"]
                is_async = command_payload["is_async"]

                if is_async:
                    await method(*argsList)
                else:
                    method(*argsList)

                return {"status": "success", "command": command}

            return {"status": "error", "message": "Command not found"}

        except Exception as e:
            logger.error(f"Error in commandPromptCheck: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
            
    def swap(self, model_name):
        try:
            logger.info(f"Swapping model to: {model_name}")
            self.large_language_model = model_name
            logger.info(f"Model swapped to {model_name}, inheriting the previous chat history")
        except Exception as e:
            logger.error(f"Error swapping model: {e}")
        
    def swapClear(self, swap_model_selection):
        try:
            logger.info(f"Swapping model to: {swap_model_selection} and clearing chat history")
            self.large_language_model = swap_model_selection
            self.chat_history = []  # Clear chat history or perform other necessary actions
            logger.info(f"Model swapped to {swap_model_selection}, with new chat history")
            return True
        except Exception as e:
            logger.error(f"Error swapping model and clearing chat history: {e}")
            return False

    def set_voice(self, voice_type: str, voice_name: str):
        try:
            if not voice_type or not voice_name:
                raise ValueError("Voice type and name must be provided")
                
            self.voice_type = voice_type
            self.voice_name = voice_name
            
            # Initialize TTS processor if needed
            if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
                self.tts_processor_instance = self.instance_tts_processor(voice_type, voice_name)
            else:
                self.tts_processor_instance.update_voice(voice_type, voice_name)
                
            self.TTS_FLAG = True  # Enable TTS when voice is set
            return True
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
        
    def cleanup(self):
        try:
            if self.tts_processor_instance:
                self.tts_processor_instance.cleanup()
            if self.speech_recognizer_instance:
                self.speech_recognizer_instance.cleanup()
            # Clear chat history
            self.chat_history = []
            self.llava_history = []
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_chat_state(self):
        return {
            "chat_history": self.chat_history,
            "llava_history": self.llava_history,
            "current_model": self.large_language_model,
            "voice_state": {
                "type": self.voice_type,
                "name": self.voice_name,
                "tts_enabled": self.TTS_FLAG,
                "stt_enabled": self.STT_FLAG
            },
            "vision_state": {
                "llava_enabled": self.LLAVA_FLAG,
                "language_and_vision_model": getattr(self, 'language_and_vision_model', None)
            }
        }

    def start_speech_recognition(self):
        self.speech_recognition_active = True
        self.STT_FLAG = True
        self.AUTO_SPEECH_FLAG = True
        if self.SILENCE_FILTERING_FLAG:
            self.speech_recognizer_instance.start_continuous_listening()
        else:
            self.speech_recognizer_instance.listen()
        return {"status": "Speech recognition started"}
        
    def stop_speech_recognition(self):
        self.speech_recognition_active = False
        self.STT_FLAG = False
        self.AUTO_SPEECH_FLAG = False
        self.speech_recognizer_instance.cleanup()
        return {"status": "Speech recognition stopped"}
    
    def toggle_speech_recognition(self):
        if self.speech_recognition_active:
            return self.stop_speech_recognition()
        else:
            return self.start_speech_recognition()
                
    def process_tts_and_send_audio(self, response):
        audio_data = self.tts_processor_instance.process_tts_responses(response, self.voice_name)
        if audio_data:
            # Send audio data to API endpoint for frontend visualization
            asyncio.run(self.send_audio_to_frontend(audio_data, "tts"))
        return audio_data
    
    async def send_audio_to_frontend(self, audio_data, audio_type):
        async with websockets.connect('ws://localhost:2020/audio-stream') as websocket:
            await websocket.send(json.dumps({
                'audio_type': audio_type,
                'audio_data': list(audio_data)
            }))
    
    def continuous_listening(self):
        if self.CONTINUOUS_LISTENING_FLAG:
            self.speech_recognizer_instance.start_continuous_listening()
        else:
            self.speech_recognizer_instance.listen()
            
    def interrupt_speech(self):
        if hasattr(self, 'tts_processor_instance'):
            self.tts_processor_instance.interrupt_generation()
    
    def get_user_audio_data(self):
        return self.speech_recognizer_instance.get_audio_data()
    
    def get_llm_audio_data(self):
        return self.tts_processor_instance.get_audio_data()
    
    def instance_tts_processor(self, voice_type, voice_name):
        try:
            if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
                self.tts_processor_instance = textToSpeech(self.pathLibrary, voice_type, voice_name)
            return self.tts_processor_instance
        except Exception as e:
            logger.error(f"Error instantiating TTS processor: {e}")
            return None
        
    def _initialize_conversation_handler_and_prompting(self):
        self.multi_modal_prompting = multiModalPrompting()
        logger.info("Multi-modal prompting initialized successfully.")
        
    def get_available_voices(self):
        #TODO TODO MOVE TO TTS FILE AND CLEAN UP CHATBOT WIZARD
        # Get list of fine-tuned models
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_voices = [d[8:] for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d)) and d.startswith("XTTS-v2_")]
        
        # Get list of voice reference samples
        reference_voices = [d for d in os.listdir(self.tts_voice_ref_wav_pack_path) if os.path.isdir(os.path.join(self.tts_voice_ref_wav_pack_path, d))]
        
        return fine_tuned_voices, reference_voices
    
    def get_voice_selection(self):
        print(f"<<< Available voices >>>")
        fine_tuned_voices, reference_voices = self.get_available_voices()
        all_voices = fine_tuned_voices + reference_voices
        for i, voice in enumerate(all_voices):
            print(f"{i + 1}. {voice}")
        
        while True:
            selection = input("Select a voice (enter the number): ")
            try:
                index = int(selection) - 1
                if 0 <= index < len(all_voices):
                    selected_voice = all_voices[index]
                    if selected_voice in fine_tuned_voices:
                        self.voice_name = selected_voice
                        self.voice_type = "fine_tuned"
                    else:
                        self.voice_name = selected_voice
                        self.voice_type = "reference"
                    return
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
             
    async def get_available_voices(self):
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_voices = [d[8:] for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d)) and d.startswith("XTTS-v2_")]
        reference_voices = [d for d in os.listdir(self.tts_voice_ref_wav_pack_path) if os.path.isdir(os.path.join(self.tts_voice_ref_wav_pack_path, d))]
        return {"fine_tuned": fine_tuned_voices, "reference": reference_voices}
    
    def set_model(self, model_name: str) -> bool:
        try:
            logger.info(f"Attempting to set model to: {model_name}")
            logger.info(f"Current model before setting: {self.large_language_model}")
            self.large_language_model = model_name
            # Update conversation paths for new model
            self.updateConversationPaths()
            logger.info(f"Model successfully set to: {model_name}")
            logger.info(f"Current model after setting: {self.large_language_model}")
            return True
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return False
            
    async def get_available_models(self):
        return await self.ollamaCommandInstance.ollama_list()
    
    async def get_command_library(self):
        return list(self.command_library.keys())
    
    def get_agent_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "flags": {
                "TTS_FLAG": self.TTS_FLAG,
                "STT_FLAG": self.STT_FLAG,
                "LLAVA_FLAG": self.LLAVA_FLAG,
                "AUTO_SPEECH_FLAG": self.AUTO_SPEECH_FLAG
            },
            "voice": {
                "type": self.voice_type,
                "name": self.voice_name
            },
            "model": self.large_language_model
        }
     
    def update_state(self):
        if "flags" in self.new_state:
            for flag, value in self.new_state["flags"].items():
                if hasattr(self, flag):
                    setattr(self, flag, value)
                    
        if "voice" in self.new_state:
            self.set_voice(
                self.new_state["voice"].get("type"),
                self.new_state["voice"].get("name")
            )
            
        if "model" in self.new_state:
            self.set_model(self.large_language_model)
        
    def voice(self, flag):
        # Update TTS_FLAG
        self.TTS_FLAG = flag

        if self.TTS_FLAG:
            print("- text to speech activated -")
            print("🎙️ You can press shift+alt to interrupt speech generation. 🎙️")
            
            # Initialize TTS processor if not already initialized
            if not self.tts_processor_instance:
                self.tts_processor_instance = textToSpeech(
                    developer_tools_dict={
                        'current_dir': self.current_dir,
                        'parent_dir': self.parent_dir,
                        'speech_dir': self.speech_dir,
                        'recognize_speech_dir': self.recognize_speech_dir,
                        'generate_speech_dir': self.generate_speech_dir,
                        'tts_voice_ref_wav_pack_path_dir': self.tts_voice_ref_wav_pack_path
                    },
                    voice_type=self.voice_type,
                    voice_name=self.voice_name
                )
        else:
            print("- text to speech deactivated -")

        print(f"TTS_FLAG FLAG STATE: {self.TTS_FLAG}")
        return
    
    def speech(self, flag1, flag2):
        if flag1 and flag2 == True:
            print("🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️")
            self.get_voice_selection()
            self.tts_processor_instance = self.instance_tts_processor(self.voice_type, self.voice_name)
            
        return
       
    def llava_flow(self, flag):
        self.LLAVA_FLAG = flag
        print(f"LLAVA_FLAG FLAG STATE: {self.LLAVA_FLAG}")
        return
    
    def voice_swap(self, voice_name_selection):
        # Search for the name after 'forward slash voice swap'
        print(f"Agent voice swapped to {voice_name_selection}")
        print(f"<<< USER >>> ")
        
        return
        
    def listen(self):
        if not self.STT_FLAG:
            self.STT_FLAG = True
            print("- speech to text activated -")
            print("🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️")
        else:
            print("- speech to text deactivated -")

        return

    def auto_commands(self, flag):
        self.auto_commands_flag = flag
        print(f"auto_commands FLAG STATE: {self.auto_commands_flag}")
        return
    
    def wake_commands(self, flag):
        self.speech_recognizer_instance.use_wake_commands = flag
        print(f"use_wake_commands FLAG STATE: {self.speech_recognizer_instance.use_wake_commands}")
        return

    def yolo_state(self, flag):
        self.yolo_flag = flag
        print(f"use_wake_commands FLAG STATE: {self.yolo_flag}")
        return
    
    def updateCommandLibrary(self):
        self.command_library = {
            "swap": {
                "method": lambda swap_model_selection: self.swapClear(swap_model_selection),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /swap, changes the main llm model of the agent. This "
                    "command allows the user or the agent to swap in a new llm on the fly for intensive "
                    "agent modularity. "
                ),
            },
            "voice swap": {
                "method": lambda voice_name: self.voice_swap(voice_name),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /voice swap, swaps the current text to speech model out "
                    "for the specified voice name."
                ),
            },
            "agent select": {
                "method": lambda agent_id: self.setAgent(agent_id),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "Lists available agents in the agentCores matrix and allows selection. "
                    "Loads the selected agent's configuration including models, prompts, and flags."
                ),
            },
            "agent save": {
                "method": lambda: self.save_agent_state(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": "Save current agent state to agentCores"
            },
            "agent create": {
                "method": lambda template_name, agent_id: self.create_agent_from_template(template_name, agent_id),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": "Create a new agent from template"
            },
            "agent list": {
                "method": lambda: self.list_available_agents(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": "List all available agents"
            },
            "save as": {
                "method": lambda save_name, large_language_model: self.save_to_json(save_name, large_language_model),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /save as, allows the user to save the current conversation "
                    "history with the provided save name, allowing the conversation to be stored in a json. "
                ),
            },
            "load as": {
                "method": lambda load_name, large_language_model: self.load_from_json(load_name, large_language_model),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /load as, allows the user to provide the desired conversation "
                    "history which pulls from the conversation library, loading it into the agent allowing the "
                    "conversation to pick up where it left off. "
                ),
            },
            "write modelfile": {
                "method": lambda: self.model_write_class_instance.write_model_file(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /write modelfile, allows the user to design, customize, and build "
                    "their own modelfile for custom systemprompt loading, as well as gguf model selection, LoRA, adapter "
                    "merging, context length modification, as well as other ollama modelfile assets. For more Description "
                    "on ollama modelfiles check out the ollama documentation at: "
                    "https://github.com/ollama/ollama/blob/main/docs/modelfile.md "
                ),
                "documentation": "https://github.com/ollama/ollama/blob/main/docs/modelfile.md"
            },
            "convert tensor": {
                "method": lambda tensor_name: self.create_convert_manager_instance.safe_tensor_gguf_convert(tensor_name),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /convert tensor, allows the user to run the custom batch tool, "
                    "calling upon the llama.cpp repo for the convert_hf_to_gguf.py tool. For more information about "
                    "this llama.cpp tool, check out the following link to the documentation: "
                    "https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py"
                ),
                "documentation": "https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py"
            },
            "convert gguf": {
                "method": lambda STT_FLAG, model_git: self.model_write_class_instance.write_model_file_and_run_agent_create_gguf(STT_FLAG, model_git),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /convert gguf, allows the user to convert any gguf model to an ollama model by constructing "
                    "the modelfile, and specifying the path to the gguf used for creating the model, in addition to other metadata."
                    "For more information you can check out the documentation at: "
                    "https://github.com/ollama/ollama/blob/main/docs/modelfile.md "
                ),
                "documentation": "https://github.com/ollama/ollama/blob/main/docs/modelfile.md",
            },
            "listen on": {
                "method": lambda: self.listen(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /listen on, changes the state of the listen flag & allows the " 
                    "user to activate the speech generation for the agent. "
                ),
            },
            "listen off": {
                "method": lambda: self.listen(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /listen off, changes the state of the listen flag & allows the " 
                    "user to deactivate the speech generation for the agent. "
                ),
            },
            "voice on": {
                "method": lambda: self.voice(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "the command, /voice on, changes the state of the voice flag," 
                    "in turn enabling the text to speech model in the agent."
                ),
            },
            "voice off": {
                "method": lambda: self.voice(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /voice off, changes the state of the voice flag," 
                    "in turn disabling the text to speech model in the agent."
                ),
            },
            "speech on": {
                "method": lambda: self.speech(True, True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /speech on, changes the state of the listen & voice "
                    "flags enabling speech recognition and speech generation for the agent."
                ),
            },
            "speech off": {
                "method": lambda: self.speech(False, False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /speech off, changes the state of the listen & voice "
                    "flags disabling speech recognition and speech generation for the agent. "
                ),
            },
            "wake on": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /wake on, changes the state of the wake_flag, allowing the user "
                    "to enable wake names for the speech recognition, this can allow the agent to "
                    "be awoken with a phrase, and with advanced mode can respond to conversation "
                    "data said prior to the wake command through organized listening & chunk processing "
                    "of the user input audio in the past ~5 min cache, then sending this processed chunk "
                    "which had all silence removed, to the whisper speech to text model. "
                ),
            },
            "wake off": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /wake on, changes the state of the wake_flag, allowing the user "
                    "to disable wake names for the speech recognition, this can allow the agent to "
                    "be awoken with a phrase, and with advanced mode can respond to conversation "
                    "data said prior to the wake command through organized listening & chunk processing "
                    "of the user input audio in the past ~5 min cache, then sending this processed chunk "
                    "which had all silence removed, to the whisper speech to text model. "
                ),
            },
            "latex on": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /latex on, allows the user to activate the specilized latex rendering utility. "
                    "This is a specific rendering feature and is highly related to the system prompt, as well as "
                    "the artifact generation from the model output. Enabling this flag will allow for latex "
                    "mathematics rendering. "
                ),
            },
            "latex off": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /latex off, allows the user to deactivate the specilized latex rendering utility. "
                    "This is a specific rendering feature and is highly related to the system prompt, as well as "
                    "the artifact generation from the model output. Enabling this flag will allow for latex "
                    "mathematics rendering. "
                ),
            },
            "command auto on": {
                "method": lambda: self.auto_commands(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /command auto on, allows the user to activate the auto commanding feature of the agent. "
                    "This feature enabled the ollama agent roll cage chatbot agent to project, infer, and execute commands in "
                    "the agent library automatically based on the user request speech data. Auto commands allows the agent to submit "
                    "/command prompts and command lists for tool execution. "
                ),
            },
            "command auto off": {
                "method": lambda: self.auto_commands(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /command auto off, allows the user to deactivate the auto commanding feature of the agent. "
                    "This feature disables the ollama agent roll cage chatbot agent to project, infer, and execute commands in "
                    "the agent library automatically based on the user request speech data. Auto commands allows the agent to submit "
                    "/command prompts and command lists for tool execution. "
                ),
            },
            "llava flow": {
                "method": lambda: self.llava_flow(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /llava flow, allows the user to activate the llava vision model in ollama, within the chatbot agent. "
                    "This is done through specialized a custom LLAVA_SYSTEM_PROMPT & LLAVA_BOOSTER_PROMPT, these prompts are provided in "
                    "The agent library. Once collected from the library the system & booster prompts are seeded in with the user speech "
                    "or text request to create llava vision prompts. "
                ),
            },
            "llava freeze":  {
                "method": lambda: self.llava_flow(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /llava freeze, allows the user to activate the llava vision model in ollama, within the chatbot agent. "
                    "This is done through specialized a custom LLAVA_SYSTEM_PROMPT & LLAVA_BOOSTER_PROMPT, these prompts are provided in "
                    "The agent library. Once collected from the library the system & booster prompts are seeded in with the user speech "
                    "or text request to create llava vision prompts. "
                ),
            },
            "yolo on": {
                "method": lambda: self.yolo_state(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /yolo on, allows the user to activate Yolo real time object recognition model. Yolo stands for `You only "
                    "look once`. This model is able to provide bounding box data for objects on the computer screen, in the webcam, and more. "
                    "Activating yolo in the ollama agent roll cage chatbot agent framework, will allow the agent to utilizing Yolo data for "
                    "various agent frameworks. This includes the minecraft agent, the general navigator vision agent, the webcam ai chat, security "
                    "camera monitoring, and more, within the oarc environment. "
                ),
            },
            "yolo off": {
                "method": lambda: self.yolo_state(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /yolo off, allows the user to deactivate Yolo real time object recognition model. Yolo stands for `You only "
                    "look once`. This model is able to provide bounding box data for objects on the computer screen, in the webcam, and more. "
                    "Deactivating yolo in the ollama agent roll cage framework, will disallow the agent to utilizing Yolo data for "
                    "various agent frameworks. This includes the minecraft agent, the general navigator vision agent, the webcam ai chat, security "
                    "camera monitoring, and more, within the oarc environment. "
                ),
            },
            "auto speech on": {
                "method": lambda: self.auto_speech_set(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /auto speech on, allows the user to activate automatic speech to speech."
                ),
            },
            "auto speech off": {
                "method": lambda: self.auto_speech_set(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /auto speech on, allows the user to deactivate automatic speech to speech."
                ),
            },
            "quit": {
                "method": lambda: self.ollamaCommandInstance.quit(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /quit, allows the user to quit the ollama chatbot instance Shuting down "
                    "all chatbot agent processes."
                ),
            },
            "ollama create": {
                "method": lambda STT_FLAG: self.ollamaCommandInstance.ollama_create(),
                "args": False,
                "is_async": False,
                "LOCK": "STT_FLAG",
                "adminOnly": True,
                "description": (
                    "The command, /ollama create, allows the user run the ollama model creation command. Starting "
                    "the model creation menu, accepting the modelfile from /write modelfile. This will run the base "
                    "ollama create command with the specified arguments."
                    # TODO ADD LOCK ARG: ONLY RUN IN TEXT TO TEXT MODE
                    # IF LISTEN & LEAP ARE NOT DISABLED, NO OLLAMA CREATE
                    # TODO Add full speech lockdown commands, /quit, /stop, /freeze, /rewind <turns>, for spacial vision
                    # navigation and agentic action output spaces, such as robotics, voice commands, from admin users,
                    # who have been voice recognized as the correct person, these users can activate admin commands,
                    # to access lockdown protocols, since voice recognition is not full proof, this feature can
                    # be swapped in for a password, or a 2 factor authentification connected to an app on your phone.
                    # from there the admin control pannel voice commands, and buttons can be highly secure for
                    # admin personel only.
                    # TODO add encrypted speech and text output, allowing voice and text in, with encrypted packages.
                    # goal: encrypt speech to speech for interaction with the agent, but all output is garbled, this
                    # will act like a cipher, and only those with the key, or those who did the prompting will have
                    # access to. The general output of files, and actions will still be committed, and this essentially 
                    # lets you hide any one piece of information before deciding if you want to make it public with your
                    # decryption method and method of sharing and visualizing the chat.
                ),
            },
            "quit": {
                "method": lambda: self.ollamaCommandInstance.quit(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /quit, allows the user to quit the ollama chatbot instance Shuting down "
                    "all chatbot agent processes."
                ),
            },
            "ollama show": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_modelfile(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "is_async": True,
                "description": (
                    "The command, /ollama show, allows the user to quit the ollama chatbot instance Shuting down "
                    "all chatbot agent processes."
                ),
            },
            "ollama template": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_template(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama template, displays the model template from the modelfile "
                    "for the currently loaded ollama llm in the chatbot agent. The template structure defines the llm "
                    "response patterns, and specifies the defined template for user, system, assistant roles, as well "
                    "as prompt structure. "
                ),
            },
            "ollama license": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_license(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama license, displays the license from the LLM modelfile of the current "
                    "model in the agent. This license comes from the distributor of the model and defines its usage "
                    "capabilities. "
                ),
            },
            "ollama list": {
                "method": lambda: self.ollamaCommandInstance.ollama_list(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama list, displays the list of ollama models on the users machine, specificially "
                    "providing the response from the ollama list command through the ollama api. "
                ),
            },
            "ollama loaded": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_loaded_models(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama loaded, displayes all currently loaded ollama models. "
                    "This information is retrieved with the ollama.ps() method."
                ),
            },
            "splice video": {
                "method": lambda: self.data_set_video_process_instance.generate_image_data(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /splice video, splices the provided video into and image set that can be used for labeling. "
                    "Once this data is labeled in a tool such as Label Studio, it can be used for training Yolo, LlaVA and "
                    "other vision models. "
                ),
            },
            "start node": {
                "method": lambda: self.FileSharingNode_instance.start_node(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /start node, activates the peer-2-peer encrypted network node. This module "
                    "provides the necessary toolset for encrypted agent networking for various tasks. "
                ),
            },
            "conversation parquet": {
                "method": lambda: self.generate_synthetic_data(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /conversation parquet, converts the specified conversation name to a parquet dataset. "
                    "This dataset can be exported to huggingface for llm finetuning, and can be found in the conversation "
                    "history library under the parquetDatasets folder."
                ),
            },
            "convert wav": {
                "method": lambda: self.data_set_video_process_instance.call_convert(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /convert wav, calls the audio wav conversion tool. (WIP: may not be functioning)"
                ),
            },
            "shot prompt": {
                "method": lambda: self.shot_prompt(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /shot prompt, prompts the ollama model with the args following the command. "
                    "This prompt is done in a new conversation"
                ),
            },
        }
        
    def setup_default_agents(self):
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
                agent_id = agent_config['agent_core']['agent_id']
                if not self.agent_cores.loadAgentCore(agent_id):
                    self.agent_cores.mintAgent(agent_id, agent_config)
                    logger.info(f"Created default agent: {agent_id}")
                else:
                    logger.info(f"Agent already exists: {agent_id}")

        except Exception as e:
            logger.error(f"Error setting up default agents: {e}")