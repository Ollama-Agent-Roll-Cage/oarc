"""pandasDB.py

please help me create a pandas db using the following colab code snippet for the rest of oarc: ollamaAgentRollCage
Pandas Query Engine
This guide shows you how to use our PandasQueryEngine: convert natural language to Pandas python code using LLMs.
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

from oarc.decorators.log import log
from oarc.database.agentStorage import AgentStorage
from oarc.database.prompt_template import PromptTemplate

@log
class PandasDB:
    def __init__(self):
        """Initialize PandasDB with necessary attributes"""
        self.df = pd.DataFrame(columns=[
            'timestamp',
            'role',
            'content',
            'metadata'
        ])
        self.setup_conversation_storage()
        self.query_engine = None
        self.conversation_handler = None
        self.agent_cores = None
        self.current_date = datetime.now().strftime("%Y%m%d")
        
        # Set up path library
        self.pathLibrary = {
            'conversation_library_dir': 'conversations',
            'default_conversation_path': None
        }

    def setup_query_engine(self, df: pd.DataFrame, verbose: bool = True, synthesize_response: bool = True):
        """Set up the Pandas Query Engine"""
        try:
            self.df = df
            self.query_engine = PandasQueryEngine(
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
        """Update query engine prompts with custom templates"""
        try:
            new_prompt = PromptTemplate(
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
            self.query_engine.update_prompts({"pandas_prompt": new_prompt})
            log.info("Query engine prompts updated")
        except Exception as e:
            log.error(f"Error updating query engine prompts: {e}")

    async def query_data(self, query_str: str) -> str:
        """Execute a natural language query against the dataframe"""
        try:
            if not self.query_engine:
                raise ValueError("Query engine not initialized. Call setup_query_engine first.")
                
            response = self.query_engine.query(query_str)
            log.info(f"Query executed: {query_str}")
            return str(response)
        except Exception as e:
            log.error(f"Error executing query: {e}")
            return f"Error: {str(e)}"
    
    def chatbotPandasDB(self, query_str: str):
        """Process a natural language query for the chatbot"""
        
        try:
            # Ensure we have a dataframe loaded
            if self.df is None:
                raise ValueError("No dataframe loaded. Please load data first.")

            # Execute the query
            response = asyncio.run(self.query_data(query_str))
            
            # Add to conversation history if needed
            if self.conversation_handler:
                asyncio.run(self.store_message("user", query_str))
                asyncio.run(self.store_message("assistant", str(response)))
            
            return response
        except Exception as e:
            log.error(f"Error in chatbot query: {e}")
            return f"Error processing query: {str(e)}"
    
    def storeAgent(self):
        """A method to store the current agent json config in the pandas db
        
        Allows users to store the models associated with the agent
        """
        try:
            # Create agent configuration entry
            agent_entry = {
                'timestamp': datetime.now(),
                'role': 'system',
                'content': 'agent_configuration',
                'metadata': json.dumps({
                    'agent_id': self.agent_id,
                    'models': {
                        'largeLanguageModel': {
                            'name': self.large_language_model,
                            'type': 'llm'
                        },
                        'largeLanguageAndVisionAssistant': {
                            'name': self.language_and_vision_model,
                            'type': 'vision'
                        },
                        'yoloVision': {
                            'name': self.yolo_model,
                            'type': 'detection'
                        },
                        'speechRecognitionSTT': {
                            'name': self.whisper_model,
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
                        'llmSystem': self.llmSystemPrompt,
                        'llmBooster': self.llmBoosterPrompt,
                        'visionSystem': self.visionSystemPrompt,
                        'visionBooster': self.visionBoosterPrompt
                    }
                })
            }

            # Add to DataFrame
            self.df = pd.concat([self.df, pd.DataFrame([agent_entry])], ignore_index=True)
            
            # Save to agent cores if available
            if self.agent_cores:
                self.save_agent_state()
                
            log.info(f"Agent {self.agent_id} configuration stored successfully")
            return True
            
        except Exception as e:
            log.error(f"Error storing agent configuration: {e}")
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
            
            # Load from PandasDB
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
            
            # Save to PandasDB
            asyncio.run(self.conversation_handler.save_conversation(self.save_name, self.agent_id))
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
        
    def file_name_conversation_history_filter(self, input):
        # Use regex to replace all spaces with underscores and convert to lowercase
        output = re.sub(' ', '_', input).lower()
        return output
    
    def setAgent(self, agent_id):
        """Load agent configuration from agentCores and update current state"""
        try:
            log.info("Loading agent configuration for agent_id: %s", agent_id)
            # From agent cores, load the agent configuration for the selected agent_id
            agent_config = self.agent_cores.loadAgentCore(agent_id)
            # set id in config
            agent_config["agentCore"]["agent_id"] = agent_id 
            
            # Remove the redundant agent_id key if it exists
            if 'agent_id' in agent_config:
                del agent_config['agent_id']
            
            if not agent_config:
                raise ValueError(f"Agent {agent_id} not found")

            # Ensure models key is present
            if "models" not in agent_config:
                agent_config["agentCore"]["models"] = {
                    "largeLanguageModel": {"names": [], "instances": [], "model_config_template": {}},
                    "embedding": {"names": [], "instances": [], "model_config_template": {}},
                    "largeLanguageAndVisionAssistant": {"names": [], "instances": [], "model_config_template": {}},
                    "yoloVision": {"names": [], "instances": [], "model_config_template": {}},
                    "speechRecognitionSTT": {"names": [], "instances": [], "model_config_template": {}},
                    "voiceGenerationTTS": {"names": [], "instances": [], "model_config_template": {}}
                }
                log.info("Added missing 'models' key to agent_config")

            # Ensure prompts key is present
            if "prompts" not in agent_config:
                agent_config["agentCore"]["prompts"] = {
                    "userInput": "",
                    "llmSystem": "",
                    "llmBooster": "",
                    "visionSystem": "",
                    "visionBooster": "",
                    "primeDirective": ""
                }
                log.info("Added missing 'prompts' key to agent_config")

            # Ensure modalityFlags key is present
            if "modalityFlags" not in agent_config:
                agent_config["agentCore"]["modalityFlags"] = {
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
            if "conversation" not in agent_config:
                agent_config["agentCore"]["conversation"] = {
                    "save_name": "",
                    "load_name": ""
                }
                log.info("Added missing 'conversation' key to agent_config")

            # Update agent state from configuration
            self.agent_id = agent_config["agentCore"]["agent_id"]
            # models
            self.large_language_model = agent_config["agentCore"]["models"]["largeLanguageModel"]["names"][0] if agent_config["agentCore"]["models"]["largeLanguageModel"]["names"] else None
            self.embedding_model = agent_config["agentCore"]["models"]["embedding"]["names"][0] if agent_config["agentCore"]["models"]["embedding"]["names"] else None
            self.language_and_vision_model = agent_config["agentCore"]["models"]["largeLanguageAndVisionAssistant"]["names"][0] if agent_config["agentCore"]["models"]["largeLanguageAndVisionAssistant"]["names"] else None
            self.yolo_model = agent_config["agentCore"]["models"]["yoloVision"]["names"][0] if agent_config["agentCore"]["models"]["yoloVision"]["names"] else None
            self.whisper_model = agent_config["agentCore"]["models"]["speechRecognitionSTT"]["names"][0] if agent_config["agentCore"]["models"]["speechRecognitionSTT"]["names"] else None
            self.voice_name = agent_config["agentCore"]["models"]["voiceGenerationTTS"]["names"][0] if agent_config["agentCore"]["models"]["voiceGenerationTTS"]["names"] else None
            self.voice_type = agent_config["agentCore"]["models"]["voiceGenerationTTS"]["model_config_template"].get("voice_type", None)
            # prompts
            self.user_input_prompt = agent_config["agentCore"]["prompts"]["userInput"]
            self.llmSystemPrompt = agent_config["agentCore"]["prompts"]["llmSystem"]
            self.llmBoosterPrompt = agent_config["agentCore"]["prompts"]["llmBooster"]
            self.visionSystemPrompt = agent_config["agentCore"]["prompts"]["visionSystem"]
            self.visionBoosterPrompt = agent_config["agentCore"]["prompts"]["visionBooster"]
            # flags
            self.LLM_SYSTEM_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["LLM_SYSTEM_PROMPT_FLAG"]
            self.LLM_BOOSTER_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["LLM_BOOSTER_PROMPT_FLAG"]
            self.VISION_SYSTEM_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["VISION_SYSTEM_PROMPT_FLAG"]
            self.VISION_BOOSTER_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["VISION_BOOSTER_PROMPT_FLAG"]
            self.TTS_FLAG = agent_config["agentCore"]["modalityFlags"]["TTS_FLAG"]
            self.STT_FLAG = agent_config["agentCore"]["modalityFlags"]["STT_FLAG"]
            self.CHUNK_FLAG = agent_config["agentCore"]["modalityFlags"]["CHUNK_AUDIO_FLAG"]
            self.AUTO_SPEECH_FLAG = agent_config["agentCore"]["modalityFlags"]["AUTO_SPEECH_FLAG"]
            self.LLAVA_FLAG = agent_config["agentCore"]["modalityFlags"]["LLAVA_FLAG"]
            self.SCREEN_SHOT_FLAG = agent_config["agentCore"]["modalityFlags"]["SCREEN_SHOT_FLAG"]
            self.SPLICE_FLAG = agent_config["agentCore"]["modalityFlags"]["SPLICE_VIDEO_FLAG"]
            self.AGENT_FLAG = agent_config["agentCore"]["modalityFlags"]["ACTIVE_AGENT_FLAG"]
            self.MEMORY_CLEAR_FLAG = agent_config["agentCore"]["modalityFlags"]["CLEAR_MEMORY_FLAG"]
            self.EMBEDDING_FLAG = agent_config["agentCore"]["modalityFlags"]["EMBEDDING_FLAG"]
            # conversation metadata
            self.save_name = agent_config["agentCore"]["conversation"]["save_name"]
            self.load_name = agent_config["agentCore"]["conversation"]["load_name"]

            # Update paths
            self.updateConversationPaths()
            log.info(f"Agent {agent_id} loaded successfully:\n%s", pformat(agent_config, indent=2, width=80))

        except Exception as e:
            log.error(f"Error loading agent {agent_id}: {e}")
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
            log.error(f"Error listing agents: {e}")
            return []

    def create_agent_from_template(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
        """Create a new agent from a template"""
        try:
            # Use agentCores to create the agent from the template
            self.agent_cores.create_agent_from_template(template_name, agent_id)
            
            # Initialize the new agent
            self.setAgent(agent_id)
            log.info(f"Agent {agent_id} created successfully from template {template_name}")
        except Exception as e:
            log.error(f"Error creating agent from template: {e}")
            raise

    def save_agent_state(self):
        try:
            # Create current state configuration matching agentCore structure
            current_state = {
                "agentCore": {
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
            log.info(f"Saved agent state: {self.agent_id}")
            return True
        except Exception as e:
            log.error(f"Error saving agent state: {e}")
            return False
        
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
        #TODO CONVERT TO PANDAS DB WITH JSON DATA FOR AGENTS
        database_config = {
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

        # Create the complete agent core structure
        self.agentCore = {
            "agentCore": {
                "identifyers": {
                    "agent_id": self.agent_id,
                    "uid": None,
                },
                "models": models_config,
                "prompts": prompts_config,
                "databases": database_config,
                "modalityFlags": modality_flags
            }
        }
        
    def initializeConversation(self):
        try:
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            
            # Update conversation paths after initializing defaults
            self.updateConversationPaths()
            log.info("Conversation initialized successfully.")
        except Exception as e:
            log.error(f"Error initializing conversation: {e}")
            raise
        
    def updateConversationPaths(self):
        """Update conversation-specific paths"""
        try:
            agent_conversation_dir = os.path.join(self.pathLibrary['conversation_library_dir'], self.agent_id)
            os.makedirs(agent_conversation_dir, exist_ok=True)
            log.info("Conversation paths updated successfully.")
        except Exception as e:
            log.error(f"Error updating conversation paths: {e}")
            raise

    def setup_conversation_storage(self):
        """Setup conversation storage schema"""
        self.conversation_schema = {
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
        """Store a message with multimodal data"""
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
            self.conversation_schema['messages'].append({
                'role': role,
                'content': content,
                **(metadata or {})
            })
            
            return True
        except Exception as e:
            log.error(f"Error storing message: {e}")
            return False

    def export_conversation(self, format: str = "json") -> str:
        """Export conversation in standard format with multimodal data"""
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
                "metadata": self.conversation_schema["metadata"]
            }, indent=2)
        except Exception as e:
            log.error(f"Error exporting conversation: {e}")
            return "{}"

    async def get_conversation_history(self, session_id=None, limit=100):
        try:
            history = await self.conversation_handler.get_conversation_history(session_id, limit)
            log.info(f"Retrieved conversation history: {history}")
            return history
        except Exception as e:
            log.error(f"Error retrieving conversation history: {e}")
            raise

    async def save_conversation(self, save_name, session_id):
        try:
            await self.conversation_handler.save_conversation(save_name, session_id)
            log.info(f"Saved conversation with name: {save_name}")
        except Exception as e:
            log.error(f"Error saving conversation: {e}")
            raise

    async def load_conversation(self, save_name):
        try:
            conversation = await self.conversation_handler.load_conversation(save_name)
            log.info(f"Loaded conversation with name: {save_name}")
            return conversation
        except Exception as e:
            log.error(f"Error loading conversation: {e}")
            raise

    async def clear_history(self, session_id=None):
        try:
            await self.conversation_handler.clear_history(session_id)
            log.info(f"Cleared conversation history for session_id: {session_id}")
        except Exception as e:
            log.error(f"Error clearing conversation history: {e}")
            raise

    def setup_routes(self):
        @self.router.post("/api/agent/create")
        async def create_agent(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
            """Create a new agent from a template"""
            try:
                agent_storage = AgentStorage()
                agent = agent_storage.create_agent_from_template(template_name, agent_id, custom_config)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/api/agent/list")
        async def list_agents(self):
            """Get list of available agents"""
            try:
                agent_storage = AgentStorage()
                agents = await agent_storage.list_available_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/api/agent/{agent_id}")
        async def get_agent(self, agent_id: str):
            """Get agent configuration"""
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
            """Load an existing agent"""
            try:
                agent_storage = AgentStorage()
                agent = agent_storage.load_agent(agent_id)
                return {"status": "success", "agent": agent}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.delete("/api/agent/{agent_id}")
        async def delete_agent(self, agent_id: str):
            """Delete an existing agent"""
            try:
                agent_storage = AgentStorage()
                agent_storage.purge_agent(agent_id)
                return {"status": "success", "message": f"Agent {agent_id} deleted"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/agent/reset")
        async def reset_agents(self):
            """Reset all agents to default templates"""
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
            """Get list of available models"""
            try:
                agent_storage = AgentStorage()
                models = await agent_storage.get_available_models()
                return {"models": models}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/api/agent/commands")
        async def get_command_library(self):
            """Get available command library"""
            try:
                agent_storage = AgentStorage()
                commands = await agent_storage.get_command_library()
                return {"commands": commands}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.put("/api/agent/{agent_id}/flags")
        async def update_agent_flags(self, agent_id: str, flags: Dict[str, bool]):
            """Update agent flags"""
            try:
                agent_storage = AgentStorage()
                agent_storage.initAgentStorage(agent_id)
                
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
            """Update agent models"""
            try:
                agent_storage = AgentStorage()
                agent_storage.initAgentStorage(agent_id)
                
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