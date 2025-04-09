"""
This module is responsible for storing the agents in a pandas dataframe.
"""

import time
import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class AgentStorage:
    """
    AgentStorage class for managing agent configurations, interactions, and lifecycle operations.
    This class provides functionality to:
    - Initialize and manage agent configurations, including default agents.
    - Handle agent-specific attributes, flags, and states.
    - Interact with a database for storing and retrieving agent data.
    - Support agent initialization, purging, and reloading templates.
    - Provide utilities for listing available agents and models.
    - Manage agent-specific commands and conversation handling.
    Key Features:
    - Default agent setup with pre-defined configurations for various use cases.
    - Integration with PandasDB for agent data persistence.
    - Modular design for initializing core attributes, flags, and models.
    - Support for asynchronous operations like fetching available models.
    - Comprehensive error handling and logging for robust operation.
    """

    def __init__(self):
        """Initialize the AgentStorage class with default configurations and attributes.
        
        This constructor sets up the necessary components for managing agent data, 
        including initializing the PandasDB instance and preparing the agent dataframe.
        """
        from oarc.database.pandas_db import PandasDB
        self.pandas_db = PandasDB()
        self.agent_df = None
        self.db = PandasDB()
        

    def setup_default_agents(self):
        """
        Set up default agents with pre-defined configurations.

        This method initializes a collection of default agents with their respective configurations.
        It ensures that each agent is created and stored in the database if it does not already exist.
        """
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
                if not self.pandas_db.loadAgentCore(agent_id):
                    self.pandas_db.mintAgent(agent_id, agent_config)
                    log.info(f"Created default agent: {agent_id}")
                else:
                    log.info(f"Agent already exists: {agent_id}")

        except Exception as e:
            log.error(f"Error setting up default agents: {e}")
            
            
    def initialize_agent_storage(self, agent_id):
        """
        Initialize the agent storage with the given agent ID.
        This method sets up the necessary components and configurations for the agent's storage,
        including logging, paths, flags, attributes, tools, state variables, histories, model
        settings, database connections, and conversation handling.
        Steps performed:
        - Sets up logging for the agent.
        - Assigns the provided agent ID to the instance.
        - Initializes base paths required for the agent's operations.
        - Configures agent-specific flags.
        - Sets up core model attributes.
        - Prepares spell tools for the agent.
        - Initializes state variables such as command library and current date.
        - Sets up chat history management.
        - Configures model-specific settings like user input prompt and screenshot path.
        - Defines save and load names for conversation persistence.
        - Loads the agent core using the provided agent ID.
        - Establishes a connection to the database and retrieves the agent collection.
        - Initializes conversation handling mechanisms.
        - Finalizes agent and conversation initialization.
        - Updates the command library with the latest commands.
        Logs a success message upon successful initialization or raises an exception
        with detailed error information if any step fails.
        """
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
            self.initialize_agent()
            self.initializeConversation()
            
            # Update command library
            self.updateCommandLibrary()

            log.info("ollamaAgentRollCage initialized successfully")
        except Exception as e:
            log.error(f"Error initializing ollamaAgentRollCage: {e}")
            log.exception("Detailed initialization error:")
            raise


    def runPurge(self, agent_id):
        """Purge all agents from the database and reinitialize the specified agent.
        
        This method performs the following steps:
        - Deletes all existing agents from the database.
        - Sets up default agents with pre-defined configurations.
        - Reloads agent templates into the database.
        - Initializes the specified agent to ensure it is ready for use.
        
        Args:
            agent_id (str): The ID of the agent to reinitialize after purging.
        """
        try:
            # TODO add if agent_id is None, purge all agents in selected matrix, or delete selected agent
            self.purge_agents()
            self.setup_default_agents()
            self.reload_templates()
            self.initialize_agent()
        except Exception as e:
            log.error(f"Error purging and reinitializing agents: {e}")
            raise
            

    def initialize_agent(self):
        """
        Initializes the agent's state with core attributes, configuration, and conversation details.
        This method performs the following steps:
        1. Sets up the conversation handler and prompting mechanism.
        2. Checks if an agent with the given ID already exists in the database:
           - If it exists, loads the agent's core attributes.
           - If it does not exist, creates a new agent and stores it in the database.
        3. Configures conversation-related details, including save and load paths.
        4. Initializes the prompt handler for managing interactions.
        5. Logs the successful initialization or raises an error if any step fails.
        Raises:
            Exception: If an error occurs during the initialization process.
        """
        try:
            self._initialize_conversation_handler_and_prompting()

            # Initialize agent
            existing_agent = self.pandas_db.loadAgentCore(self.agent_id)
            if (existing_agent):
                self.setAgent(self.agent_id)
            else:
                self.coreAgent()
                self.pandas_db.mintAgent(self.agent_id, self.agent_core)

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
        """
        Initialize core attributes for the agent.

        This method sets up the foundational attributes required for the agent's operation.
        It initializes user input prompts, model attributes, voice settings, and flags to their
        default states. This ensures the agent is in a clean and consistent state before further
        configuration or usage.
        """
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
        """
        Initialize all agent flags with their default values.

        This method sets up the initial state for various agent flags, ensuring
        that the agent starts with a consistent and predictable configuration.
        These flags control different functionalities and behaviors of the agent.
        """
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
        """
        Retrieves a list of available models from the Ollama library.

        This method interacts with the `ollamaCommandInstance` to fetch the list
        of models. If the operation is successful, it returns the list of models.
        If the result is not a list or an error occurs during the operation, it
        logs the error and returns an empty list.

        Returns:
            list: A list of available models, or an empty list if an error occurs
            or the result is not a valid list.
        """
        try:
            models = await self.ollamaCommandInstance.ollama_list()
            return models if isinstance(models, list) else []
        except Exception as e:
            log.error(f"Error getting available models: {e}")
            return []


    async def list_available_agents(self) -> list:
        """
        Temporarily return an empty list until it's fully implemented.
        """
        try:
            return []
        except Exception as e:
            log.error(f"Error listing agents: {e}")
            return []


    def purge_agents(self):
        """
        Remove all agents from the PandasDB.

        This method deletes all agent records stored in the database, effectively
        clearing the agent collection. It is useful for resetting the database or
        preparing for a fresh setup of agents.
        """
        try:
            self.agent_collection.delete_many({})
            log.info("All agents purged from PandasDB.")
        except Exception as e:
            log.error(f"Error purging agents: {e}")


    def purge_agent(self, agent_id):
        """
        Remove a specific agent from the PandasDB.

        This method deletes the agent record associated with the given agent ID
        from the database. It is useful for removing an individual agent's data
        without affecting other agents in the database.

        Args:
            agent_id (str): The unique identifier of the agent to be removed.

        Logs:
            Logs a success message if the agent is successfully removed.
            Logs an error message if the operation fails.
        """
        try:
            self.agent_collection.delete_one({"agent_id": agent_id})
            log.info(f"Agent {agent_id} purged from PandasDB.")
        except Exception as e:
            log.error(f"Error purging agent {agent_id}: {e}")
        

    def reload_templates(self):
        """
        Reload agent templates into PandasDB.

        This method is responsible for reloading predefined agent templates into the database.
        It ensures that the templates are available for creating new agents or updating existing ones.
        If an error occurs during the process, it logs the error message for debugging purposes.
        """
        try:
            # Assuming you have a method to create agents from templates
            self.create_agent_from_template('default_template', 'defaultAgent')
            log.info("Agent templates reloaded into PandasDB.")
        except Exception as e:
            log.error(f"Error reloading templates: {e}")


    def load_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Load the agent configuration from the database.

        This method retrieves the configuration details for the specified agent ID
        from the database. It ensures that the agent's core attributes, models, 
        prompts, and other settings are properly loaded and returned as a dictionary.

        Args:
            agent_id (str): The unique identifier of the agent to be loaded.

        Returns:
            Dict[str, Any]: A dictionary containing the agent's configuration details.

        Raises:
            ValueError: If the agent configuration cannot be found in the database.
        """
        agent_config = self.get_agent_config(agent_id)
        if not agent_config:
            raise ValueError(f"Agent configuration for {agent_id} not found")

        if "agent_core" not in agent_config:
            agent_config["agent_core"] = {
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
        """
        Retrieve the configuration details for a specific agent.
        """
        return self.pandas_db.loadAgentCore(agent_id)
    

    async def get_command_library(self):
        """
        Asynchronously retrieves the list of command names available in the agent's command library.

        Returns:
            list: A list of command names (keys) from the command library if successful.
            dict: A dictionary containing an error message if an exception occurs.

        Logs:
            Logs an error message if an exception is raised during the retrieval process.
        """
        try:
            return list(self.command_library.keys())
        except Exception as e:
            log.error(f"Error getting command library: {e}")
            return {"error": str(e)}

    def create_agent_from_template(self, template_name: str, agent_id: str, custom_config: dict = None):
        """
        Create an agent from a template.

        This method creates a new agent using a predefined template and stores it in the database.
        It allows for optional customization of the agent's configuration.

        Args:
            template_name (str): The name of the template to use for creating the agent.
            agent_id (str): The unique identifier for the new agent.
            custom_config (dict, optional): Additional configuration to customize the agent.

        Logs:
            Logs a success message if the agent is successfully created.
            Logs an error message if the operation fails.
        """
        try:
            existing_agent = self.get_agent_config(agent_id)
            if existing_agent:
                log.info(f"Agent '{agent_id}' already exists. Skipping creation.")
                return existing_agent

            log.info(f"Creating agent '{agent_id}' from template '{template_name}'")
            self.db.create_agent_from_template(template_name, agent_id, custom_config)
            new_agent = self.load_agent(agent_id)
            log.info(f"Agent '{agent_id}' created successfully from template '{template_name}'")
            return new_agent

        except Exception as e:
            log.error(f"Error creating agent from template: {e}")
            raise
