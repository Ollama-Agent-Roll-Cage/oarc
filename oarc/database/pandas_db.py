"""
This module defines the PandasDB class, which serves as a database interface using a pandas DataFrame to manage conversation history, agent configurations, and query executions. It integrates asynchronous methods for conversation and agent storage handling with a FastAPI-based routing setup, ensuring smooth interactions with natural language queries. The class supports setting up a query engine, storing and exporting multimodal conversation data, and managing agent states with detailed logging for debugging and system monitoring.
"""

import os
import re
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
from fastapi import HTTPException
from pprint import pformat
from llama_index.experimental.query_engine import PandasQueryEngine

from oarc.utils.log import log
from oarc.database.agent_storage import AgentStorage
from oarc.database.prompt_template import PromptTemplate
from oarc.utils.decorators.singleton import singleton


@singleton
class PandasDB:
    def __init__(self):
        """
        Initialize a new instance of PandasDB with default settings for managing conversation data.
        Attributes:
            df (pd.DataFrame): A DataFrame initialized with columns ['timestamp', 'role', 'content', 'metadata'] to store conversation records.
            query_engine: Placeholder attribute for the query engine instance.
            conversation_handler: Placeholder attribute for managing conversation interactions.
            agent_cores: Placeholder attribute for core functions related to agent processing.
            current_date (str): Current date formatted as YYYYMMDD.
            pathLibrary (dict): Dictionary containing path settings:
                - 'conversation_library_dir': Directory name for storing conversation logs.
                - 'default_conversation_path': Placeholder for the default conversation file path (currently None).
        Side Effects:
            Configures conversation storage by invoking setup_conversation_storage().
        """
        """Initialize PandasDB with necessary attributes"""
        self.df = pd.DataFrame(columns=[
            'timestamp',
            'role',
            'content',
            'metadata'
        ])
        self.setup_conversation_storage()
        self.engine = None
        self.handler = None # TODO we need initialization logic for this, not init??
        self.agent_cores = None
        self.current_date = datetime.now().strftime("%Y%m%d")
        
        # Set up path dictionary for conversation storage
        self.path_dict = {
            'conversation_library_dir': 'conversations',
            'default_conversation_path': None
        }


    def setup_query_engine(self, df: pd.DataFrame, verbose: bool = True, synthesize_response: bool = True):
        """
        Initializes and configures the Pandas Query Engine using the provided DataFrame.
        This method sets up the internal DataFrame and query engine, optionally enabling verbose logging 
        and response synthesis. It also updates the engine's prompt templates to incorporate any customizations.
        Parameters:
            df (pd.DataFrame): The DataFrame from which the query engine will retrieve data.
            verbose (bool, optional): Flag to enable verbose output for debugging and logging. Defaults to True.
            synthesize_response (bool, optional): Flag to enable response synthesis. Defaults to True.
        Returns:
            bool: True if the query engine was successfully set up; False if an error occurred during initialization.
        Exceptions:
            Logs any exception encountered during setup and handles it by returning False.
        """
        """Set up the Pandas Query Engine"""
        try:
            self.df = df
            self.engine = PandasQueryEngine(
                df=self.df,
                verbose=verbose,
                synthesize_response=synthesize_response
            )
            
            # Update prompts with custom templates
            self.update_query_engine_prompts()
            log.info("Query engine setup successful")
            return True
        except Exception as e:
            log.error(f"Error setting up query engine: {e}")
            return False


    def update_query_engine_prompts(self):
        """
        Configure the query engine by updating its prompts with a customized template.
        This template details the dataframe structure (e.g., a sample output from df.head())
        and provides instructions on how the query should be formulated.
        """
        try:
            prompt = PromptTemplate(
                """
                You are working with a pandas dataframe in Python.
                The name of the dataframe is `df`.
                This is the result of `print(df.head())`:
                {df_str}

                Follow these instructions:
                {instruction_str}
                Query: {query_str}

                Expression: """
            )
            self.engine.update_prompts({"pandas_prompt": prompt})
            log.info("Query engine prompts updated")
        except Exception as e:
            log.error(f"Error updating query engine prompts: {e}")


    async def query_data(self, query_str: str) -> str:
        """
        Executes a natural language query against the dataframe using the initialized query engine.
        Parameters:
            query_str (str): The natural language query to execute.
        Returns:
            str: The string representation of the query result, or an error message if execution fails.
        Raises:
            ValueError: If the query engine is not initialized (i.e., setup_query_engine has not been called).
        """
        try:
            if not self.engine:
                raise ValueError("Query engine not initialized. Call setup_query_engine first.")
                
            response = self.engine.query(query_str)
            log.info(f"Query executed: {query_str}")
            return str(response)
        except Exception as e:
            log.error(f"Error executing query: {e}")
            return f"Error: {str(e)}"
    

    def chatbot_pandas_db(self, query_str: str):
        """Handle a natural language query for the chatbot by processing the input, executing it via the query engine, storing both the query and the generated response in the conversation history if enabled, and returning the result."""
        try:
            # Ensure we have a dataframe loaded
            if self.df is None:
                raise ValueError("No dataframe loaded. Please load data first.")

            # Execute the query
            response = asyncio.run(self.query_data(query_str))
            
            # Add to conversation history if needed
            if self.handler:
                asyncio.run(self.store_message("user", query_str))
                asyncio.run(self.store_message("assistant", str(response)))
            
            return response
        except Exception as e:
            log.error(f"Error in chatbot query: {e}")
            return f"Error processing query: {str(e)}"
    

    def store_agent(self):
        """
        Store the current agent configuration in the pandas database.

        This method serializes the agent's JSON configuration—including associated models, flags,
        and prompts—and appends it to the internal DataFrame. It provides a streamlined way for users
        to persist and subsequently retrieve the agent's settings.
        """
        try:
            entry = {
                'timestamp': datetime.now(),
                'role': 'system',
                'content': 'agent_configuration',
                'metadata': json.dumps({
                    'agent_id': self.agent_id,
                    'models': {
                        'largeLanguageModel': {
                            'name': self.llm,
                            'type': 'llm'
                        },
                        'largeLanguageAndVisionAssistant': {
                            'name': self.lvm,
                            'type': 'vision'
                        },
                        'yoloVision': {
                            'name': self.vision,
                            'type': 'detection'
                        },
                        'speechRecognitionSTT': {
                            'name': self.sst,
                            'type': 'stt'
                        },
                        'voiceGenerationTTS': {
                            'name': self.voice_name,
                            'type': 'tts',
                            'voice_type': self.voice_type
                        }
                    },
                    'flags': {
                        'TTS_FLAG': self.TTS_FLAG,
                        'STT_FLAG': self.STT_FLAG,
                        'LLAVA_FLAG': self.LLAVA_FLAG,
                        'SCREEN_SHOT_FLAG': self.SCREEN_SHOT_FLAG,
                        'SPLICE_FLAG': self.SPLICE_FLAG,
                        'EMBEDDING_FLAG': self.EMBEDDING_FLAG
                    },
                    'prompts': {
                        'userInput': self.user_input_prompt,
                        'llmSystem': self.llm_system_prompt,
                        'llmBooster': self.llm_booster_prompt,
                        'visionSystem': self.vision_system_prompt,
                        'visionBooster': self.vision_booster_prompt
                    }
                })
            }

            # Add to DataFrame
            self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True)
            
            # Save to agent cores if available
            if self.agent_cores:
                self.save_agent_state()
                
            log.info(f"Agent {self.agent_id} configuration stored successfully")
            return True
            
        except Exception as e:
            log.error(f"Error storing agent configuration: {e}")
            return False
        

    def load_from_json(self, load_name, large_language_model):
        """Load conversation history from a JSON file and update internal state accordingly."""
        # Update load name and model
        self.load_name = load_name if load_name else f"conversation_{self.agent_id}_{self.current_date}"
        self.llm = large_language_model
        
        # Update paths with new load name
        temp_save_name = self.save_name
        self.save_name = self.load_name
        self.update_conversation_paths()
        
        # Load conversation
        try:
            with open(self.path_dict['default_conversation_path'], 'r') as json_file:
                self.history = json.load(json_file)
            print(f"Conversation loaded from: {self.path_dict['default_conversation_path']}")
            
            # Load from PandasDB
            self.history = asyncio.run(self.handler.load_conversation(self.save_name))
            
        except Exception as e:
            print(f"Error loading conversation: {e}")
        finally:
            # Restore original save name
            self.save_name = temp_save_name
            self.update_conversation_paths()


    def save_to_json(self, save_name, large_language_model):
        """
        Persist the complete conversation history to a JSON file.
        This method writes all recorded messages along with their metadata—including timestamps,
        roles, content, and any associated details—to a designated JSON file.
        
        Parameters:
            save_name (str): The file path or name where the conversation history should be saved.
            large_language_model: The configuration or reference for the large language model used 
                                  during the conversation session.
        
        Returns:
            None
        
        Side Effects:
            Creates or overwrites the specified JSON file with the full conversation history.
        """
        # Update save name and model
        self.save_name = save_name if save_name else f"conversation_{self.agent_id}_{self.current_date}"
        self.llm = large_language_model
        
        # Update paths with new save name
        self.update_conversation_paths()
        
        # Save conversation
        try:
            with open(self.path_dict['default_conversation_path'], 'w') as json_file:
                json.dump(self.history, json_file)
            print(f"Conversation saved to: {self.path_dict['default_conversation_path']}")
            
            # Save to PandasDB
            asyncio.run(self.handler.save_conversation(self.save_name, self.agent_id))
            
        except Exception as e:
            log.error(f"Error saving conversation: {e}")
            raise RuntimeError(f"Error saving conversation: {e}")
        

    def file_name_conversation_history_filter(self, input):
        """
        Converts a given string into a valid filename by replacing all spaces with underscores and
        transforming all characters to lowercase, ensuring consistency and compatibility with typical
        file naming requirements.

        Parameters:
            input (str): The input string that may include spaces and uppercase letters.

        Returns:
            str: The transformed string suitable for use as a filename.

        Notes:
            This function uses a regular expression substitution to replace space characters,
            ensuring that the conversion is performed consistently.
        """
        # Replace spaces with underscores and convert the input to lowercase using a regex substitution.
        output = re.sub(' ', '_', input).lower()
        return output
    

    def set_agent(self, agent_id):
        """
        Load and apply the configuration for the specified agent.

        This method retrieves the agent's configuration from agentCores using the provided
        agent_id, then updates the current instance’s state with the loaded parameters.
        It ensures that models, prompts, and flags are synchronized with the agent configuration.

        Parameters:
            agent_id (str): The identifier of the agent to load.

        Raises:
            Exception: Propagates any exceptions encountered during configuration loading.
        """
        try:
            log.info("Loading agent configuration for agent_id: %s", agent_id)
            # From agent cores, load the agent configuration for the selected agent_id
            config = self.agent_cores.load_agent_core(agent_id)
            # set id in config
            config["agent_core"]["agent_id"] = agent_id 
            
            # Remove the redundant agent_id key if it exists
            if 'agent_id' in config:
                del config['agent_id']
            
            if not config:
                raise ValueError(f"Agent {agent_id} not found")

            # Ensure models key is present
            if "models" not in config:
                config["agent_core"]["models"] = {
                    "largeLanguageModel": {"names": [], "instances": [], "model_config_template": {}},
                    "embedding": {"names": [], "instances": [], "model_config_template": {}},
                    "largeLanguageAndVisionAssistant": {"names": [], "instances": [], "model_config_template": {}},
                    "yoloVision": {"names": [], "instances": [], "model_config_template": {}},
                    "speechRecognitionSTT": {"names": [], "instances": [], "model_config_template": {}},
                    "voiceGenerationTTS": {"names": [], "instances": [], "model_config_template": {}}
                }
                log.info("Added missing 'models' key to agent_config")

            # Ensure prompts key is present
            if "prompts" not in config:
                config["agent_core"]["prompts"] = {
                    "userInput": "",
                    "llmSystem": "",
                    "llmBooster": "",
                    "visionSystem": "",
                    "visionBooster": "",
                    "primeDirective": ""
                }
                log.info("Added missing 'prompts' key to agent_config")

            # Ensure modalityFlags key is present
            if "modalityFlags" not in config:
                config["agent_core"]["modalityFlags"] = {
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
                log.info("Added missing 'modalityFlags' key to agent_config")

            # Ensure conversation key is present
            if "conversation" not in config:
                config["agent_core"]["conversation"] = {
                    "save_name": "",
                    "load_name": ""
                }
                log.info("Added missing 'conversation' key to agent_config")

            # Update agent state from configuration
            self.agent_id = config["agent_core"]["agent_id"]
            
            # models
            self.llm = config["agent_core"]["models"]["largeLanguageModel"]["names"][0] if config["agent_core"]["models"]["largeLanguageModel"]["names"] else None
            self.embedding_model = config["agent_core"]["models"]["embedding"]["names"][0] if config["agent_core"]["models"]["embedding"]["names"] else None
            self.lvm = config["agent_core"]["models"]["largeLanguageAndVisionAssistant"]["names"][0] if config["agent_core"]["models"]["largeLanguageAndVisionAssistant"]["names"] else None
            self.vision = config["agent_core"]["models"]["yoloVision"]["names"][0] if config["agent_core"]["models"]["yoloVision"]["names"] else None
            self.sst = config["agent_core"]["models"]["speechRecognitionSTT"]["names"][0] if config["agent_core"]["models"]["speechRecognitionSTT"]["names"] else None
            self.voice_name = config["agent_core"]["models"]["voiceGenerationTTS"]["names"][0] if config["agent_core"]["models"]["voiceGenerationTTS"]["names"] else None
            self.voice_type = config["agent_core"]["models"]["voiceGenerationTTS"]["model_config_template"].get("voice_type", None)
            
            # prompts
            self.user_input_prompt = config["agent_core"]["prompts"]["userInput"]
            self.llm_system_prompt = config["agent_core"]["prompts"]["llmSystem"]
            self.llm_booster_prompt = config["agent_core"]["prompts"]["llmBooster"]
            self.vision_system_prompt = config["agent_core"]["prompts"]["visionSystem"]
            self.vision_booster_prompt = config["agent_core"]["prompts"]["visionBooster"]
            
            # flags
            self.LLM_SYSTEM_PROMPT_FLAG = config["agent_core"]["modalityFlags"]["LLM_SYSTEM_PROMPT_FLAG"]
            self.LLM_BOOSTER_PROMPT_FLAG = config["agent_core"]["modalityFlags"]["LLM_BOOSTER_PROMPT_FLAG"]
            self.VISION_SYSTEM_PROMPT_FLAG = config["agent_core"]["modalityFlags"]["VISION_SYSTEM_PROMPT_FLAG"]
            self.VISION_BOOSTER_PROMPT_FLAG = config["agent_core"]["modalityFlags"]["VISION_BOOSTER_PROMPT_FLAG"]
            self.TTS_FLAG = config["agent_core"]["modalityFlags"]["TTS_FLAG"]
            self.STT_FLAG = config["agent_core"]["modalityFlags"]["STT_FLAG"]
            self.CHUNK_FLAG = config["agent_core"]["modalityFlags"]["CHUNK_AUDIO_FLAG"]
            self.AUTO_SPEECH_FLAG = config["agent_core"]["modalityFlags"]["AUTO_SPEECH_FLAG"]
            self.LLAVA_FLAG = config["agent_core"]["modalityFlags"]["LLAVA_FLAG"]
            self.SCREEN_SHOT_FLAG = config["agent_core"]["modalityFlags"]["SCREEN_SHOT_FLAG"]
            self.SPLICE_FLAG = config["agent_core"]["modalityFlags"]["SPLICE_VIDEO_FLAG"]
            self.AGENT_FLAG = config["agent_core"]["modalityFlags"]["ACTIVE_AGENT_FLAG"]
            self.MEMORY_CLEAR_FLAG = config["agent_core"]["modalityFlags"]["CLEAR_MEMORY_FLAG"]
            self.EMBEDDING_FLAG = config["agent_core"]["modalityFlags"]["EMBEDDING_FLAG"]
            
            # conversation metadata
            self.save_name = config["agent_core"]["conversation"]["save_name"]
            self.load_name = config["agent_core"]["conversation"]["load_name"]

            # Update paths
            self.update_conversation_paths()
            log.info(f"Agent {agent_id} loaded successfully:\n%s", pformat(config, indent=2, width=80))

        except Exception as e:
            log.error(f"Error loading agent {agent_id}: {e}")
            raise


    async def list_available_agents(self) -> list:
        """
        Retrieve the list of available agents along with their detailed configuration.
        This asynchronous method fetches a list of agents using the agent_cores.listAgentCores() method.
        For each agent, it loads additional configuration details via agent_cores.loadAgentCore() and extracts
        information such as the agent's unique ID, the associated large language model (llm), vision assistant,
        voice generation model, and the version of the agent. If any part of the data retrieval fails, the
        method logs the error and returns an empty list.
        Returns:
            list: A list of dictionaries where each dictionary represents an agent with the following keys:
                - "id": The unique identifier of the agent.
                - "llm": The name of the large language model used by the agent.
                - "vision": The name of the large language and vision assistant.
                - "voice": The name of the voice generation text-to-speech (TTS) model.
                - "version": The version of the agent, or "Unknown" if not specified.
        Exceptions:
            Any exceptions during processing are logged and the function will return an empty list.
        """
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
            log.error(f"Error listing agents: {e}")
            raise RuntimeError(f"Error listing agents: {e}")


    def create_agent_from_template(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
        """
        Create a new agent using a specified template.
        
        This method generates a new agent configuration based on the provided template name and assigns it the given agent ID. 
        Optionally, a custom configuration dictionary can be supplied to override or extend the default settings defined in the template.
        
        Parameters:
            template_name (str): The name of the template to use for creating the agent.
            agent_id (str): The unique identifier to assign to the newly created agent.
            custom_config (Optional[Dict]): A dictionary containing custom configuration values to override the template defaults.
        
        Returns:
            None
        
        Raises:
            Exception: If an error occurs during the agent creation process.
        """
        try:
            # Use agentCores to create the agent from the template
            self.agent_cores.create_agent_from_template(template_name, agent_id)
            
            # Initialize the new agent
            self.set_agent(agent_id)
            log.info(f"Agent {agent_id} created successfully from template {template_name}")
        except Exception as e:
            log.error(f"Error creating agent from template: {e}")
            raise


    def save_agent_state(self):
        """
        Persist the current agent's configuration and state to the agent cores.

        This method captures the agent's current settings, including models, prompts, 
        modality flags, and other configurations, and stores them in the agent cores 
        for future retrieval or updates.

        Returns:
            bool: True if the agent state was successfully saved, False otherwise.

        Logs:
            Logs success or failure messages, including any encountered exceptions.
        """
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
                            "names": [self.llm] if self.llm else [],
                            "instances": []
                        },
                        "embedding": {
                            "names": [self.embedding_model] if self.embedding_model else [],
                            "instances": []
                        },
                        "largeLanguageAndVisionAssistant": {
                            "names": [self.lvm] if self.lvm else [],
                            "instances": []
                        },
                        "yoloVision": {
                            "names": [self.vision] if self.vision else [],
                            "instances": []
                        },
                        "speechRecognitionSTT": {
                            "names": [self.sst] if self.sst else [],
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
                            "llmSystem": self.llm_system_prompt,
                            "llmBooster": self.llm_booster_prompt,
                            "visionSystem": self.vision_system_prompt,
                            "visionBooster": self.vision_booster_prompt
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
            log.info(f"Saved agent state: {self.agent_id}")
            return True
        except Exception as e:
            log.error(f"Error saving agent state: {e}")
            return False
        

    def core_agent(self):
        """
        Constructs the core agent structure, which includes configurations for models, prompts, 
        databases, and modality flags. This structure serves as the foundational setup for the agent's 
        functionality.
        The method organizes the following components:
        1. **Model Configurations**:
           - Defines various model types such as large language models, embedding models, vision models, 
             speech recognition models, and text-to-speech models.
           - Each model type includes:
             - `names`: List of model names (if available).
             - `instances`: Placeholder for model instances.
             - `model_config_template`: Template for additional model-specific configurations.
        2. **Prompt Configurations**:
           - Specifies prompts for user input, system-level prompts, and booster prompts for both 
             language and vision systems.
           - Includes placeholders for prime directives and other customizable prompts.
        3. **Database Configurations**:
           - Defines database file paths for various agent functionalities such as conversation history, 
             knowledge storage, documentation, library, web search, web scraping, embeddings, and design 
             patterns.
           - TODO: Convert to a pandas-based database structure with JSON data for agents.
        4. **Modality Flags**:
           - Configures flags to enable or disable specific modalities such as text-to-speech (TTS), 
             speech-to-text (STT), audio chunking, automatic speech, vision systems, and more.
           - Includes flags for memory management, embedding usage, and active agent status.
        5. **Agent Core Structure**:
           - Combines all the above configurations into a single dictionary structure.
           - Includes agent identifiers such as `agent_id` and a placeholder for `uid`.
        Returns:
            None: The method initializes the `self.agent_core` attribute with the complete agent core 
            structure.
        """
        # Define model configurations
        models_config = {
            "largeLanguageModel": {
                "names": [self.llm] if self.llm else [],
                "instances": [],
                "model_config_template": {}
            },
            "embedding": {
                "names": [self.embedding_model] if self.embedding_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "largeLanguageAndVisionAssistant": {
                "names": [self.lvm] if self.lvm else [],
                "instances": [],
                "model_config_template": {}
            },
            "yoloVision": {
                "names": [self.vision] if self.vision else [],
                "instances": [],
                "model_config_template": {}
            },
            "speechRecognitionSTT": {
                "names": [self.sst] if self.sst else [],
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
        #TODO CONVERT TO PANDAS DB WITH JSON DATA FOR AGENTS
        db_config = {
            "agents": {
                "conversation": f"conversation_{self.agent_id}.db",
                "knowledge": f"knowledge_{self.agent_id}.db",
                "documentation": f"documentation_{self.agent_id}.db",
                "library": f"library_{self.agent_id}.db",
                "web_search": f"web_search_{self.agent_id}.db",
                "web_scrape": f"web_scrape_{self.agent_id}.db",
                "embeddings": f"embeddings_{self.agent_id}.db",
                "design_patterns": f"design_patterns_{self.agent_id}.db"
            },
        },

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

        # Create single complete agent core structure
        self.agent_core = {
            "agent_core": {
                "identifyers": {
                    "agent_id": self.agent_id,
                    "uid": None,
                },
                "models": models_config,
                "prompts": prompts_config,
                "databases": db_config,
                "modalityFlags": modality_flags
            }
        }
        

    def init_conversation(self):
        """
        Initializes a conversation with default values and file paths.
        This method sets up the conversation by generating a default save name 
        based on the agent ID and the current date. It also ensures that the 
        conversation paths are updated accordingly. Logs the success or failure 
        of the initialization process.
        Raises:
            Exception: If an error occurs during the initialization process, 
                       it logs the error and re-raises the exception.
        """
        try:
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            
            # Update conversation paths after initializing defaults
            self.update_conversation_paths()
            log.info("Conversation initialized successfully.")
        except Exception as e:
            log.error(f"Error initializing conversation: {e}")
            raise
        

    def update_conversation_paths(self):
        """
        Updates the file paths specific to the current conversation by creating 
        a directory for the agent's conversations if it does not already exist.

        This method ensures that the directory structure required for storing 
        conversation-related data is in place. If the directory creation fails, 
        an error is logged and the exception is re-raised.

        Raises:
            Exception: If there is an error while creating the conversation directory.
        """
        try:
            agent_conversation_dir = os.path.join(self.path_dict['conversation_library_dir'], self.agent_id)
            os.makedirs(agent_conversation_dir, exist_ok=True)
            log.info("Conversation paths updated successfully.")
        except Exception as e:
            log.error(f"Error updating conversation paths: {e}")
            raise


    def setup_conversation_storage(self):
        """
        Initializes the schema for conversation storage.
        This method defines and sets up a structured schema to store conversation 
        data, including messages and associated metadata. The metadata provides 
        contextual information such as the agent's identifier, the models involved 
        (e.g., language model, vision model, voice model), and additional flags 
        for custom configurations or states.
        Schema structure:
        - `messages`: A list to store conversation messages.
        - `metadata`: A dictionary containing:
            - `agent_id`: Identifier for the agent managing the conversation.
            - `models`: A nested dictionary specifying:
                - `llm`: Language model details or identifier.
                - `vision`: Vision model details or identifier.
                - `voice`: Voice model details or identifier.
            - `flags`: A dictionary for storing additional custom flags or states.
        """
        self.schema = {
            'messages': [],
            'metadata': {
                'agent_id': None,
                'models': {
                    'llm': None,
                    'vision': None,
                    'voice': None
                },
                'flags': {}
            }
        }


    async def store_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Store a message with associated metadata and update the conversation history.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
            metadata (Optional[Dict], optional): Additional metadata associated with the message. 
                Defaults to None.
        Returns:
            bool: True if the message was successfully stored, False otherwise.
        This method performs the following actions:
        1. Creates a dictionary entry for the message, including a timestamp, role, content, 
           and serialized metadata (if provided).
        2. Appends the entry to the internal pandas DataFrame (`self.df`).
        3. Updates the conversation history schema (`self.conversation_schema['messages']`) 
           with the message and its metadata.
        4. Logs and handles any exceptions that occur during the process.
        """
        try:
            entry = {
                'timestamp': datetime.now(),
                'role': role,
                'content': content,
                'metadata': json.dumps(metadata) if metadata else None
            }
            
            # Add to DataFrame
            self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True)
            
            # Add to conversation history
            self.schema['messages'].append({
                'role': role,
                'content': content,
                **(metadata or {})
            })
            
            return True
        except Exception as e:
            log.error(f"Error storing message: {e}")
            return False


    def export_conversation(self, format: str = "json") -> str:
        """
        Export the conversation history in a specified format, including multimodal data such as images, audio, and vision metadata.
        
        Parameters:
            format (str): The format in which to export the conversation. Currently supports "json". Defaults to "json".
        
        Returns:
            str: A string representation of the exported conversation in the specified format.
        
        Notes:
            - The exported conversation includes messages with roles, content, and any associated metadata (e.g., images, audio, vision).
            - Metadata is extracted and included in the output if available.
            - If an error occurs during the export process, an empty JSON object is returned.
        """
        try:
            messages = []
            for _, row in self.df.iterrows():
                message = {
                    "role": row["role"],
                    "content": row["content"]
                }
                
                if pd.notna(row["metadata"]):
                    metadata = json.loads(row["metadata"])
                    if metadata.get("images"):
                        message["images"] = metadata["images"]
                    if metadata.get("audio"):
                        message["audio"] = metadata["audio"]
                    if metadata.get("vision"):
                        message["vision"] = metadata["vision"]
                        
                messages.append(message)
            
            return json.dumps({
                "messages": messages,
                "metadata": self.schema["metadata"]
            }, indent=2)
        except Exception as e:
            log.error(f"Error exporting conversation: {e}")
            return "{}"


    async def get_conversation_history(self, session_id=None, limit=100):
        """
        Retrieve the conversation history for a specific session.
        
        Parameters:
            session_id (str, optional): The unique identifier for the session whose history is to be retrieved. 
                                        If None, retrieves the default session's history.
            limit (int, optional): The maximum number of messages to retrieve. Defaults to 100.
        
        Returns:
            list: A list of conversation messages, each containing details such as role, content, and metadata.
        
        Raises:
            Exception: If an error occurs while retrieving the conversation history, it logs the error and re-raises the exception.
        """
        try:
            history = await self.handler.get_conversation_history(session_id, limit)
            log.info(f"Retrieved conversation history: {history}")
            return history
        except Exception as e:
            log.error(f"Error retrieving conversation history: {e}")
            raise


    async def save_conversation(self, save_name, session_id):
        """
        Asynchronously save the current conversation to a specified file or database.

        Args:
            save_name (str): The name under which the conversation will be saved.
            session_id (str): The unique identifier for the current session.

        Raises:
            Exception: If an error occurs during the save operation, it will be logged and re-raised.

        Logs:
            - Logs an informational message upon successful save.
            - Logs an error message if the save operation fails.
        """
        try:
            await self.handler.save_conversation(save_name, session_id)
            log.info(f"Saved conversation with name: {save_name}")
        except Exception as e:
            log.error(f"Error saving conversation: {e}")
            raise


    async def load_conversation(self, save_name):
        """
        Asynchronously loads a conversation from a specified file or database.

        Args:
            save_name (str): The name or identifier of the conversation to load.

        Returns:
            object: The loaded conversation object.

        Raises:
            Exception: If an error occurs while loading the conversation.

        Logs:
            - Info: Logs the successful loading of the conversation with its name.
            - Error: Logs any errors encountered during the loading process.
        """
        try:
            conversation = await self.handler.load_conversation(save_name)
            log.info(f"Loaded conversation with name: {save_name}")
            return conversation
        except Exception as e:
            log.error(f"Error loading conversation: {e}")
            raise


    async def clear_history(self, session_id=None):
        """
        Clears the conversation history for a specific session.

        Args:
            session_id (str, optional): The unique identifier for the session. 
                If not provided, the default behavior of the conversation handler is applied.

        Raises:
            Exception: If an error occurs while clearing the conversation history.

        Logs:
            - Info: Logs a message indicating successful clearing of the conversation history.
            - Error: Logs an error message if an exception is raised.
        """
        try:
            await self.handler.clear_history(session_id)
            log.info(f"Cleared conversation history for session_id: {session_id}")
        except Exception as e:
            log.error(f"Error clearing conversation history: {e}")
            raise


    # TODO the following API code should be moved into its own pandas db api script
    # -----------------------------------------------------
    # API routes for managing agents and their configurations
    # -----------------------------------------------------//
    def setup_routes(self):
        """
        Set up API routes for managing agents and their configurations.
        
        This method defines the endpoints for creating, listing, retrieving, updating, 
        and deleting agents, as well as managing their models, flags, and commands. 
        It provides a RESTful interface for interacting with the agent storage and 
        configuration system.
        """
        @self.router.post("/api/agent/create")
        async def create_agent(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
            """
            Create a new agent using a specified template.

            This endpoint allows the creation of a new agent by specifying a template name, 
            a unique agent ID, and optionally a custom configuration to override or extend 
            the default settings defined in the template.

            Parameters:
            template_name (str): The name of the template to use for creating the agent.
            agent_id (str): The unique identifier to assign to the newly created agent.
            custom_config (Optional[Dict], optional): A dictionary containing custom 
                configuration values to override the template defaults. Defaults to None.

            Returns:
            dict: A dictionary containing the status of the operation and the created agent details.

            Raises:
            HTTPException: If an error occurs during the agent creation process, 
            an HTTP 500 error is raised with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent = agent_storage.create_agent_from_template(template_name, agent_id, custom_config)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/list")
        async def list_agents(self):
            """
            Retrieve a list of all available agents.

            This endpoint fetches and returns a list of agents currently stored in the system, 
            including their configurations and metadata. It provides an overview of all agents 
            available for interaction or management.
            """
            try:
                agent_storage = AgentStorage()
                agents = await agent_storage.list_available_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/{agent_id}")
        async def get_agent(self, agent_id: str):
            """
            Retrieve the configuration for a specific agent.

            This endpoint fetches the configuration details of an agent 
            identified by the provided `agent_id`. If the agent does not 
            exist, a 404 HTTP exception is raised. In case of any other 
            unexpected errors, a 500 HTTP exception is raised.

            Args:
                agent_id (str): The unique identifier of the agent.

            Returns:
                dict: The configuration details of the agent.

            Raises:
                HTTPException: If the agent is not found (404) or if an 
                unexpected error occurs (500).
            """
            try:
                agent_storage = AgentStorage()
                agent_config = agent_storage.get_agent_config(agent_id)
                if not agent_config:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                return agent_config
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.post("/api/agent/load")
        async def load_agent(self, agent_id: str):
            """
            Load an existing agent by its unique identifier.

            This endpoint retrieves the configuration and state of an agent identified by the provided `agent_id`.
            If the agent does not exist or an error occurs during the loading process, an appropriate HTTP exception
            is raised.

            Args:
            agent_id (str): The unique identifier of the agent to be loaded.

            Returns:
            dict: A dictionary containing the status of the operation and the loaded agent details.

            Raises:
            HTTPException: If an error occurs during the agent loading process, an HTTP 500 error is raised
            with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent = agent_storage.load_agent(agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.delete("/api/agent/{agent_id}")
        async def delete_agent(self, agent_id: str):
            """
            Delete an existing agent by its unique identifier.

            This endpoint removes the agent identified by the provided `agent_id` from the system.
            If the agent does not exist or an error occurs during the deletion process, an appropriate
            HTTP exception is raised.

            Args:
            agent_id (str): The unique identifier of the agent to be deleted.

            Returns:
            dict: A dictionary containing the status of the operation and a success message.

            Raises:
            HTTPException: If an error occurs during the deletion process, an HTTP 500 error is raised
            with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.purge_agent(agent_id)
                return {"status": "success", "message": f"Agent {agent_id} deleted"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.post("/api/agent/reset")
        async def reset_agents(self):
            """
            Reset all agents to their default templates.

            This endpoint removes all existing agents and reinitializes them using the default templates.
            It ensures that the system is restored to its original state with the default agent configurations.

            Returns:
            dict: A dictionary containing the status of the operation and a success message.

            Raises:
            HTTPException: If an error occurs during the reset process, an HTTP 500 error is raised
            with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.purge_agents()
                agent_storage.setup_default_agents()
                agent_storage.reload_templates()
                return {"status": "success", "message": "All agents reset to defaults"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/models")
        async def get_available_models(self):
            """
            Retrieve a list of available models.

            This endpoint fetches and returns a list of models that are currently available 
            for use by the agents. The models may include large language models, vision models, 
            embedding models, and others, depending on the system's configuration.
            
            Returns:
            dict: A dictionary containing the list of available models under the key "models".
            
            Raises:
            HTTPException: If an error occurs during the retrieval process, an HTTP 500 error 
            is raised with the error details.
            """
            try:
                agent_storage = AgentStorage()
                models = await agent_storage.get_available_models()
                return {"models": models}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.get("/api/agent/commands")
        async def get_command_library(self):
            """
            Handles GET requests to retrieve the available command library for agents.

            This endpoint interacts with the AgentStorage to fetch the list of commands
            available in the system. If an error occurs during the process, an HTTP 500
            exception is raised with the error details.

            Returns:
                dict: A dictionary containing the list of available commands under the key "commands".

            Raises:
                HTTPException: If an error occurs while fetching the command library, 
                an HTTP 500 response is returned with the error details.
            """
            try:
                agent_storage = AgentStorage()
                commands = await agent_storage.get_command_library()
                return {"commands": commands}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.put("/api/agent/{agent_id}/flags")
        async def update_agent_flags(self, agent_id: str, flags: Dict[str, bool]):
            """
            Update the modality flags for a specific agent.

            This endpoint allows updating the modality flags (e.g., TTS_FLAG, STT_FLAG) 
            for an agent identified by the provided `agent_id`. The flags determine 
            the agent's behavior and enabled functionalities.

            Args:
            agent_id (str): The unique identifier of the agent whose flags are to be updated.
            flags (Dict[str, bool]): A dictionary containing the flag names as keys and their 
                         corresponding boolean values to update.

            Returns:
            dict: A dictionary containing the status of the operation and a success message.

            Raises:
            HTTPException: If an error occurs during the update process, an HTTP 500 error is 
                       raised with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.initialize_agent_storage(agent_id)
                
                # Update flags
                for flag_name, flag_value in flags.items():
                    if hasattr(agent_storage, flag_name):
                        setattr(agent_storage, flag_name, flag_value)
                
                agent_storage.save_agent_state()
                return {"status": "success", "message": f"Agent {agent_id} flags updated"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.put("/api/agent/{agent_id}/models")
        async def update_agent_models(self, agent_id: str, models: Dict[str, str]):
            """
            Update the model configurations for a specific agent.

            This endpoint allows updating the models associated with an agent identified by the provided `agent_id`. 
            The models can include large language models, embedding models, vision models, and others. 
            The updated configurations are saved to the agent's state.

            Args:
            agent_id (str): The unique identifier of the agent whose models are to be updated.
            models (Dict[str, str]): A dictionary containing the model types as keys (e.g., "largeLanguageModel", 
                         "embeddingModel") and their corresponding model names as values.

            Returns:
            dict: A dictionary containing the status of the operation and a success message.

            Raises:
            HTTPException: If an error occurs during the update process, an HTTP 500 error is raised with the error details.
            """
            try:
                agent_storage = AgentStorage()
                agent_storage.initialize_agent_storage(agent_id)
                
                # Update models
                if "largeLanguageModel" in models:
                    agent_storage.large_language_model = models["largeLanguageModel"]
                if "embeddingModel" in models:
                    agent_storage.embedding_model = models["embeddingModel"]
                if "visionModel" in models:
                    agent_storage.language_and_vision_model = models["visionModel"]
                if "yoloModel" in models:
                    agent_storage.yolo_model = models["yoloModel"]
                if "whisperModel" in models:
                    agent_storage.whisper_model = models["whisperModel"]
                if "voiceName" in models:
                    agent_storage.voice_name = models["voiceName"]
                
                agent_storage.save_agent_state()
                return {"status": "success", "message": f"Agent {agent_id} models updated"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
        # --------------------------------------            