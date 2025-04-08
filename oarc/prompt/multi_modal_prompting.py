"""
Core messaging functionality for the OARC chatbot.

This module provides the foundational messaging and interaction capabilities
for the OARC chatbot, enabling multimodal communication, model management,
and advanced prompt handling.
"""

# TODO this should implement some sort of factor method which can spawn instances

import base64
import json
import multiprocessing
import time
from datetime import datetime
from pprint import pformat
from typing import Optional

import ollama

from oarc.database import PandasDB
from oarc.utils.log import log
from oarc.utils.paths import Paths


class MultiModalPrompting:


    def __init__(self):
        # Initialize standard flags
        self.AGENT_FLAG = True
        self.SYSTEM_SELECT_FLAG = False
        self.TTS_FLAG = True  # TTS integration flag
        
        # Add missing flag that's causing the error
        self.LLAVA_FLAG = False  # Flag for LLAVA (vision) processing
        
        # Add other flags that might be needed 
        self.EMBEDDING_FLAG = False
        self.MEMORY_CLEAR_FLAG = False
        self.LLM_BOOSTER_PROMPT_FLAG = False
        self.VISION_SYSTEM_PROMPT = False
        
        # Initialize other attributes
        self.agent = None
        self.history = None
        self.handler = None
        self.database = PandasDB()
        self.paths = Paths()  # Get singleton instance
        

    def set_model(self, model_name: str) -> bool:
        """Set the model for the agent."""
        try:
            if "core" not in self.agent:
                raise ValueError("Agent core not found in loaded agent configuration")
            
            self.agent["agent_core"]["models"]["largeLanguageModel"]["names"] = [model_name]
            log.info(f"Model set to {model_name} for agent {self.agent_id}")
            return True
        except Exception as e:
            log.error(f"Error setting model for agent {self.agent_id}: {e}")
            return False
        

    def swap(self, model_name):
        try:
            log.info(f"Swapping model to: {model_name}")
            self.llm = model_name
            log.info(f"Model swapped to {model_name}, inheriting the previous chat history")
        except Exception as e:
            log.error(f"Error swapping model: {e}")
            
            
    def swapClear(self, swap_model_selection):
        try:
            log.info(f"Swapping model to: {swap_model_selection} and clearing chat history")
            self.llm = swap_model_selection
            self.history = []  # Clear chat history
            log.info(f"Model swapped to {swap_model_selection}, with new chat history")
            return True
        except Exception as e:
            log.error(f"Error swapping model and clearing chat history: {e}")
            return False
        

    def initializeChat(self):
        """ a method to initilize the chatbot agent conversation
        """
        # initialize chat history
        self.database.history = []
        self.database.llava_history = []
        
        # loaded agent
        self.agent = {}
        
        # # TODO -> Direct ollama api access, currently unused 
        # self.url = "http://localhost:11434/api/chat"

        # # Setup chat_history
        # self.headers = {'Content-Type': 'application/json'}
    

    async def commandPromptCheck(self, input_prompt):
        try:
            if not input_prompt:
                raise ValueError("User input prompt is None or empty")
            
            command_payload = None
            if input_prompt.startswith("/") or input_prompt.startswith("activate "):
                for command, command_details in self.command_library.items():
                    if input_prompt.startswith(command):
                        args = input_prompt[len(command):].strip().split()
                        command_payload = {
                            "command": command,
                            "method": command_details["method"],
                            "is_async": command_details.get("is_async", False),
                            "argsList": args
                        }
                        break
            else:
                # Not a command, send to multi modal prompting
                log.info(f"⌛Loading Agent {self.agent_id} in progress⌛:\n%s", pformat(self.agent, indent=2, width=80))
                response = await self.multi_modal_prompting.send_prompt(
                    self.agent,
                    self.handler,
                    self.history
                )
                return {"status": "success", "response": response}

            if command_payload:
                command = command_payload["command"]
                method = command_payload["method"]
                args = command_payload["argsList"]
                is_async = command_payload["is_async"]

                if is_async:
                    await method(*args)
                else:
                    method(*args)

                return {"status": "success", "command": command}

            return {"status": "error", "message": "Command not found"}

        except Exception as e:
            log.error(f"Error in commandPromptCheck: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        

    async def promptSetManager(self, sys_prompt_select):
        """ a method for direct system prompt, and booster prompt selection & modification
            via the agent_matrix.db, allowing for any agent prompt to be selected and loaded
            into the current core
        """
        # TODO add booster prompt, vision system prompt, and vision booster prompt selection.
        if sys_prompt_select in self.prompts:
            self.history.append({"role": "system", "content": self.prompts[sys_prompt_select]})
        else:
            log.info("Invalid choice. Please select a valid prompt.")
        return sys_prompt_select
    

    def get_system_prompt(self):
        """Extract system prompt from loaded agent configuration."""
        try:
            if not self.agent:
                return None

            # Navigate through the agent configuration structure
            if "agent_core" in self.agent:
                if "prompts" in self.agent["agent_core"]:
                    if "agent" in self.agent["agent_core"]["prompts"]:
                        system_prompt = self.agent["agent_core"]["prompts"]["agent"].get("llmSystem")
                        if system_prompt:
                            return system_prompt

            log.warning("No system prompt found in agent configuration")
            return None
        except Exception as e:
            log.error(f"Error retrieving system prompt: {e}")
            return None
        

    async def send_prompt(self, agent, handler, history):
        """Send prompt with multimodal data handling"""
        try:
            self.agent = agent
            self.history = history

            user_prompt = self.agent["agent_core"]["prompts"]["userInput"]
            model = self.agent["agent_core"]["models"]["largeLanguageModel"]["names"][0]
            conversation_name = self.agent["agent_core"]["conversation"]["load_name"]

            log.info(f"Loading agent {conversation_name[:50]} with model {model[:50]} and prompt {user_prompt[:50]}")

            # Set flags from agent configuration
            EMBEDDING_FLAG = self.agent["agent_core"]["modalityFlags"]["EMBEDDING_FLAG"]
            MEMORY_CLEAR_FLAG = self.agent["agent_core"]["modalityFlags"]["MEMORY_CLEAR_FLAG"]
            STREAMING_FLAG = self.agent["agent_core"]["modalityFlags"].get("STREAMING_FLAG", False)
            
            # Keep LLAVA and other flags
            self.LLAVA_FLAG = self.agent["agent_core"]["modalityFlags"].get("LLAVA_FLAG", False)
            self.LLM_BOOSTER_PROMPT_FLAG = self.agent["agent_core"]["modalityFlags"].get("LLM_BOOSTER_PROMPT_FLAG", False)
            self.TTS_FLAG = self.agent["agent_core"]["modalityFlags"].get("TTS_FLAG", False)

            self.handler = handler

            if MEMORY_CLEAR_FLAG is True:
                history.clear()
                log.info("Chat history cleared due to MEMORY_CLEAR_FLAG")

            # Get system prompt
            system_prompt = self.get_system_prompt()
            if not system_prompt:
                log.warning("No system prompt found in agent configuration")
                log.warning("Using default system prompt")
                system_prompt = "You are a helpful assistant. Please help the user with their task."

            # Set system prompt in history
            if "llmSystem" in self.agent["agent_core"]["prompts"]:
                self.history.append({"role": "system", "content": self.agent["agent_core"]["prompts"]["llmSystem"]})
                log.info(f"System prompt set: {self.agent['agent_core']['prompts']['llmSystem']}")

            # Handle LLAVA and booster prompts as before
            if not self.LLAVA_FLAG and not self.LLM_BOOSTER_PROMPT_FLAG:
                self.history.append({"role": "user", "content": user_prompt})
                log.info(f"User prompt added: {user_prompt}")

            if self.LLM_BOOSTER_PROMPT_FLAG and not self.LLAVA_FLAG:
                self.fused_prompt = self.agent["agent_core"]["prompts"]["llmBooster"] + f"{user_prompt}"
                self.history.append({"role": "user", "content": self.fused_prompt})
                log.info(f"Fused prompt added: {self.fused_prompt}")

            # Handle LLAVA vision processing if enabled
            if self.LLAVA_FLAG:
                # Your existing LLAVA code here
                pass

            # Get language from agent config
            language = agent["agent_core"].get("language", "en") if "agent_core" in agent else "en"

            # Generate response
            model_response = ''
            if EMBEDDING_FLAG is True:
                response = await self.embedding_ollama_prompt(self.agent_id, user_prompt)
                model_response = response
            else:
                try:
                    # Import ollama here to ensure it's available
                    from oarc.ollama.utils import ollama
                    
                    if STREAMING_FLAG:
                        log.info(f"Using streaming mode with model {model}")
                        
                        # Initialize an empty response
                        model_response = ""
                        current_sentence = ""
                        sentence_buffer = []
                        
                        # Get the streaming response
                        try:
                            response = await ollama.chat(
                                model=model,
                                messages=self.history,
                                stream=True
                            )
                            
                            # Process the streaming response properly
                            if hasattr(response, '__aiter__'):
                                # Modern interface with async for
                                async for chunk in response:
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        content = chunk['message']['content']
                                        model_response += content
                                        current_sentence += content
                                        
                                        # Check for sentence boundaries to process in chunks
                                        if content in '.!?\n' and len(current_sentence.strip()) > 0:
                                            sentence_buffer.append(current_sentence)
                                            current_sentence = ""
                                            
                                            # If we have a complete sentence, process it
                                            if hasattr(handler, 'handle_stream'):
                                                for sentence in sentence_buffer:
                                                    await handler.handle_stream(sentence)
                                                sentence_buffer = []
                                        
                                        # For longer chunks without sentence breaks
                                        if len(current_sentence) > 120:
                                            sentence_buffer.append(current_sentence)
                                            current_sentence = ""
                                            
                                            # Process buffered sentences
                                            if hasattr(handler, 'handle_stream'):
                                                for sentence in sentence_buffer:
                                                    await handler.handle_stream(sentence)
                                                sentence_buffer = []
                            else:
                                # For older non-async interface
                                for chunk in response:
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        content = chunk['message']['content']
                                        model_response += content
                                        
                                        # Call handler's stream handler if it exists
                                        if hasattr(handler, 'handle_stream'):
                                            await handler.handle_stream(content)
                            
                            # Process any remaining content
                            if current_sentence or sentence_buffer:
                                if current_sentence:
                                    sentence_buffer.append(current_sentence)
                                
                                if hasattr(handler, 'handle_stream'):
                                    for sentence in sentence_buffer:
                                        await handler.handle_stream(sentence)
                                        
                            log.info(f"Streaming response complete: {model_response[:100]}...")
                        except Exception as e:
                            log.error(f"Error processing stream: {e}")
                            raise
                    else:
                        # Regular non-streaming response
                        log.info(f"Using standard response from model {model}")
                        response = await ollama.chat(
                            model=model,
                            messages=self.history
                        )
                        model_response = response['message']['content']
                        log.info(f"Response generation complete: {model_response[:100]}...")
                except Exception as e:
                    log.error(f"Error during Ollama chat: {e}")
                    return f"Error: {e}"
            
            # Store in conversation history
            self.history.append({"role": "assistant", "content": model_response})
            
            # Add metadata as before
            metadata = {
                "model_info": {
                    "name": model,
                    "type": "llm"
                }
            }

            # Add vision data to metadata if available
            if self.LLAVA_FLAG and 'user_screenshot_raw2' in locals():
                metadata["vision"] = {
                    "llava": [{
                        "image": user_screenshot_raw2,
                        "timestamp": datetime.now().isoformat()
                    }]
                }

            # Handle TTS processing if enabled
            if self.TTS_FLAG and hasattr(self, 'tts_processor_instance'):
                try:
                    # Process TTS with the correct voice reference
                    await self.tts_processor_instance.process_tts_audio(model_response, language=language)
                except Exception as e:
                    log.error(f"TTS processing error: {str(e)}")

            return model_response

        except Exception as e:
            log.error(f"Error in send_prompt: {e}")
            return f"Error: {e}"
        

    def ollama_concurrent_prompts(self, model, prompt_list):
        """ a method for sending prompts in multiprocessing to the model, this will be recorded 
        to the conversation.
            args: model, promptList
        """
        for prompt in prompt_list:
            with multiprocessing.Pool(processes=1) as pool:
                #TODO TEST AND IMPLEMENT
                pool.apply_async(self.send_prompt, args=(model,))
        pass
    

    def ollama_concurrent_models(self, modelList, prompList, multiprocessEachModel):
        """ a method for sending prompts in multiprocessing to multiple model, this will be recorded
        to the conversation. This can be used to prompt multiple models with one prompt, multiple
        prompts, or can be used to prompt multiple modelse each with multiple prompts in multiprocessing.
            args: modelList, promptList, multiprocessEachModel"""
        for model in modelList:
            with multiprocessing.Pool(processes=1) as pool:
                #TODO TEST AND IMPLEMENT
                pool.apply_async(self.send_prompt, args=(model,))
        pass

    
    async def llava_prompt(self, input_prompt, screenshot_raw2, llava_input_prompt, language_and_vision_model="llava"):
        """ a method for prompting the vision model
            args: user_screenshot_raw2, llava_user_input_prompt, language_and_vision_model="llava"
            returns: none

            #TODO add modelfile, system prompt get feature and modelfile manager library
            #TODO /sys prompt select, /booster prompt select, ---> leverage into function calling ai 
            for modular auto prompting chatbot
        """ 
        # setup history & prompt
        self.llava_input_prompt = llava_input_prompt
        self.llava_history = []

        # if agent selected, set up system prompts for vision model

        if self.VISION_SYSTEM_PROMPT is True:
            self.vision_system_constructor = f"{self.agent['agent_core']['prompts']['visionSystem']} " + f"{input_prompt}"
            self.llava_history.append({"role": "system", "content": f"{self.vision_system_constructor}"})
        else:
            self.navigator_default()
            self.vision_system_constructor = f"{self.general_navigator_agent['agent_core']['prompts']['visionSystem']} " + f"{input_prompt}"
            self.llava_history.append({"role": "system", "content": f"{self.vision_system_constructor}"})

        if self.AGENT_FLAG == True:
            self.vision_booster_constructor = f"{self.agent['agent_core']['prompts']['visionBooster']}" + f"{input_prompt}"
            message = {"role": "user", "content": f"{self.vision_booster_constructor}"}
        else:
            self.vision_booster_constructor = f"{self.general_navigator_agent['agent_core']['prompts']['visionBooster']}" + f"{input_prompt}"
            message = {"role": "user", "content": f"{self.vision_booster_constructor}"}

        #TODO ADD LLM PROMPT REFINEMENT (example: stable diffusion llm to prompt diffusion model) 
        # AS A PREPROCESS COMBINED WITH THE CURRENT AGENTS PRIME DIRECTIVE
        # or llava model to annotate the image on screen and and store data, or pipe it stable diffusion or other llm
        
        if screenshot_raw2 is not None:
            # Assuming user_input_image is a base64 encoded image
            message["images"] = [screenshot_raw2]
            
        try:
            
            # Prompt vision model with compiled chat history data
            response_llava = await ollama.chat(
                model=language_and_vision_model, 
                messages=self.llava_history + [message], 
                stream=True
            )
            
            # process model response chunks
            model_response = ''
            async for chunk in response_llava:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    model_response += content
                    log.info(content, end='', flush=True)
            
            # Append the full response to llava_history
            self.llava_history.append({"role": "assistant", "content": model_response})
            
            # Keep only the last 2 responses in llava_history

            self.llava_history = self.llava_history[-2:]

            # Store the conversation in the conversation handler
            await self.handler.store_message({"role": "assistant", "content": model_response})
            
            return model_response
        except Exception as e:
            return f"Error: {e}"


    async def embedding_ollama_prompt(self, agent_id: str, message: str, stream: bool = True) -> Optional[str]:
        """Enhanced Ollama chat using agent's database configuration with MongoDB vector search."""
        # Load agent configuration
        agent = self.loadAgentCore(agent_id)
        if not agent:
            return f"Agent '{agent_id}' not found"
                
        # Get model configurations
        llm_model = agent["agent_core"]["models"]["largeLanguageModel"]["names"][0]
        if not llm_model:
            return "No language model configured for this agent"
        
        embedding_model = agent["agent_core"]["models"]["embedding"][0]
        
        try:
            # Get collections
            collection_name = f"conversations_{agent_id}"
            conv_collection = self.database[collection_name]
            
            # Ensure vector search index exists
            try:
                conv_collection.create_index([("vector", "vectorSearch")], {
                    "numDimensions": 768,  # for nomic-embed-text
                    "similarity": "cosine"
                })
            except Exception as e:
                log.info(f"Note: Vector index may already exist: {e}")
            
            # Create session ID
            session_id = f"{agent_id}_{int(time.time())}"
            
            # Build system prompt
            system_prompt = (
                f"{agent['agent_core']['prompts']['userInput']} "
                f"{agent['agent_core']['prompts']['agent']['llmSystem']} "
                f"{agent['agent_core']['prompts']['agent']['llmBooster']}"
            )
            
            # Get message embedding
            embedding_response = ollama.embeddings(
                model=embedding_model,
                prompt=message
            )
            message_embedding = embedding_response['embedding']
            
            # Query similar conversations using vector search
            similar_convs = conv_collection.aggregate([
                {
                    "$vectorSearch": {
                        "queryVector": message_embedding,
                        "path": "vector",
                        "numCandidates": 100,
                        "limit": 2,
                        "index": "vector_index",
                    }
                }
            ])
            
            # Build context from similar conversations
            context = ""
            for conv in similar_convs:
                context += f"Previous conversation:\nUser: {conv['content']}\n"
                if 'response' in conv:
                    context += f"Assistant: {conv['response']}\n\n"
            
            # Store user message
            conv_collection.insert_one({
                "timestamp": time.time(),
                "role": "user",
                "content": message,
                "session_id": session_id,
                "vector": message_embedding
            })
            
            # Chat with model and handle response
            response_text = ""
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"{context}\nCurrent message: {message}"}
            ]
            
            if stream:
                # Stream response
                log.info("\nAssistant: ", end="", flush=True)
                stream = ollama.chat(
                    model=llm_model,
                    messages=messages,
                    stream=True
                )
                
                for chunk in stream:
                    chunk_text = chunk['message']['content']
                    print(chunk_text, end='', flush=True)
                    response_text += chunk_text
                print()  # New line after response
                
            else:
                # Single response
                response = ollama.chat(
                    model=llm_model,
                    messages=messages
                )
                response_text = response['message']['content']
            
            # Get embedding for response
            response_embedding = ollama.embeddings(
                model=embedding_model,
                prompt=response_text
            )['embedding']
            
            # Store assistant response with embedding
            conv_collection.insert_one({
                "timestamp": time.time(),
                "role": "assistant", 
                "content": response_text,
                "session_id": session_id,
                "vector": response_embedding,
                "response": response_text  # Store response text
            })
                    
            return response_text
                
        except Exception as e:
            return f"Error in chat: {str(e)}"
        

    async def shot_prompt(self, prompt, modelSelect="none"):
        """ a method to perform a shot prompt with the selected model, this will not be recorded to
        the conversation, history and can be used to extract direct data from a model
        
            args:
                prompt - user input shot prompt data
                modelSelect - user input model selection
                
            returns: 
                model_response - model response data
        """
        if modelSelect == "none":
            modelSelect = self.llm
        
        # Clear chat history
        self.shot_history = []
        # Append user prompt
        self.shot_history.append({"role": "user", "content": prompt})

        try:
            response = await ollama.chat(model=modelSelect, prompt=prompt, stream=True)
            
            model_response = ''
            async for chunk in response:
                if 'response' in chunk:
                    content = chunk['response']
                    model_response += content
                    print(content, end='', flush=True)
            
            print('\n')
            
            # Append the full response to shot_history
            self.shot_history.append({"role": "assistant", "content": model_response})
            
            return model_response
        except Exception as e:
            return f"Error: {e}"
    

    async def mod_prompt(self, prompt, modelSelect="none", appendHistory="new"):
        """ a method to perform a shot prompt with the selected model, this will not be recorded to
        the conversation, history and can be used to extract direct data from a model
        
            args:
                prompt - user input shot prompt data
                modelSelect - user input model selection
                
            returns: 
                model_response - model response data
        """
        if modelSelect == "none":
            modelSelect = self.llm
        
        if appendHistory == "new":
            # Clear chat history
            self.shot_history = []
            # Append user prompt
            self.shot_history.append({"role": "user", "content": prompt})

            try:
                response = ollama.generate(model=modelSelect, prompt=prompt, stream=True)
                
                model_response = ''
                for chunk in response:
                    if 'response' in chunk:
                        content = chunk['response']
                        model_response += content
                        print(content, end='', flush=True)
                
                print('\n')
                
                # Append the full response to shot_history
                self.shot_history.append({"role": "assistant", "content": model_response})
                
                return model_response
            except Exception as e:
                return f"Error: {e}"
        else:
            # Append user prompt
            self.history.append({"role": "user", "content": prompt})

            try:
                response = ollama.chat(model=self.llm, messages=self.history, stream=True)
                
                model_response = ''
                for chunk in response:
                    if 'response' in chunk:
                        content = chunk['response']
                        model_response += content
                        print(content, end='', flush=True)
                
                print('\n')
                
                # Append the full response to shot_history
                self.history.append({"role": "assistant", "content": model_response})
                
                return model_response
            except Exception as e:
                return f"Error: {e}"
        

    async def design_prompt(self, prompt, modelSelect="none", contextChat="new", appendChat="new", ):
        """ a method to perform a shot prompt with the selected model, this will not be recorded to
        the conversation, history and can be used to extract direct data from a model
        
            args:
                prompt - user input shot prompt data
                modelSelect - user input model selection
                
                #TODO Add LLaVA Arg, and Img are to allow for modular input, if speech recognition
                is active take a screen shot and pipe in automatically, this allows for instant
                llava shot prompts, in a text to text agent conversation where the agent itself does
                not want to load up a llava model for every prompt, and instead can be used to seed
                the conversation history with different models
                
                #TODO add model name tag to conversation history,
                
                #TODO conversation history arg -> select which chat history the shot prompt should
                read from, and where it should be saved to. This can be saved to the main agent 
                conversation history for shot prompt agent data references.
                
            returns: 
                model_response - model response data
        """
        if modelSelect == "none":
            modelSelect = self.llm
        
        #TODO if contextArg is not "new", and is instead;
        #       agentCoreConversation; spin up shot prompt selected conversation
        #       
        # if conversationOut is not base, store prompt to specified conversation
        #       else append shot prompt and response to agentCoreConversation base conversation
        #       add model name tags to allow the agent to infer when llms are being
        #       swapped in and out of the conversation.
        
        # Clear chat history
        self.shot_history = []
        # Append user prompt
        self.shot_history.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(model=modelSelect, prompt=prompt, stream=True)
            
            model_response = ''
            for chunk in response:
                if 'response' in chunk:
                    content = chunk['response']
                    model_response += content
                    print(content, end='', flush=True)
            
            print('\n')
            
            # Append the full response to shot_history
            self.shot_history.append({"role": "assistant", "content": model_response})
            
            return model_response
        except Exception as e:
            return f"Error: {e}"


    def chain_of_thought(self, prompt, reasoningModules):
        """ a method to chain together a series of prompts, and responses to create a chain of thought
            for the agent to follow, this can be used to create a more human like conversation flow
            and can be used to create a more human like conversation flow and can be used to create
            a more human like conversation flow and can be used to create a more human like conversation 
            flow - gpt4o
            
            todo, reasoning, step by step, monte carlo, schroers path finding,
            RAG: DB, document retrivel, intelligent search -> RAG
        """
        # from prompt, and selected reasoning modules
        # for prompt, device step by step solution action plan
        # attempt to build the solution
        # test the solution
        # if failure, return to the step by step action plan and verify which step failed
        # and attempt to fix the failed steps, reusing the verified steps
        # if success, store the solution in the knowledge base
        # if failure, device new step by step solution action plan
        # and attempt to build the solution
        # if success, store the solution in the knowledge base
        # if failure, return to the step by step action plan and verify which step failed
        # and attempt to fix the failed steps, reusing the verified steps
        # if success, store the solution in the knowledge base
        # if failure, device new step by step solution action plan
        # and attempt to build the solution
        # if success, store the solution in the knowledge base
        # ... loop untill success, loop until thought chain number becomes too large, 
        # or loop until failure happens for all action plans
        # if failure for all action plants for selected reasoning modules, return to the
        # original prompt, and attempt to build a new chain of thought with a new selected 
        # reasoning module, or combination of reasoning modules, to solve to problem.
        pass
    
    def select_reasoning_module(self):
        """ a method to select the reasoning algorithm for the chainOfThought method"""
        #TODO SELECT MODE
        # codingReasoningModule, searchReasoningModule, deepResearch
        pass
    
    def coding_reasoning_module(self):
        """ a method defining the coding reasoning module for the chainOfThought method"""
        #TODO SELECT MODE
        # LOCAL MODES:
        # CODE FROM SCRATCH, LOCAL TEMPLATE FILES, EXISTING PROJECT
        # REFINE WITH WEB SEARCH AND REASONING MODES:
        # WEB EXAMPLE, GITHUB, STACKOVERFLOW, ARXIV, WIKIPEDIA, DUCKDUCKGO
        pass
    
    def search_reasoning_module(self):
        """ a method defining the search reasoning module for the chainOfThought method"""
        #TODO DUCKDUCKGO
        #TODO ARXIV
        #TODO GITHUB REPO CLONER
        #TODO CRAWL 4 AI
        # ----> WIKIPEDIA - AUTO CRAWL FOR DUCKDUCKGO WIKIPEDIA PAGES
        # ----> OTHER AUTO CRAWL FOR DUCKDUCKGO PAGES, GITHUB, ARXIV, ETC
        #TODO !!NEW!! Web crawlers for the most part will be moved to agent chef modules
        pass
    
    def deep_research(self, prompt):
        """ a method to allow the agent to perform deep reseach on the specified topic using the
        provided search apis and the chainOfThought method. Providing deep thinking loops, for
        long term memory storage and retrieval, and the ability to perform deep research on a
        topic, and provide a detailed documentation in the knowledge base database. This can then
        be retreived by the agent for future reference when coding, writing, or performing other
        tasks.
        """
        pass
    
    def swarm_prompt(self, multimodalStream):
        """ a method to allow the agent to perform a swarm prompt on 
        the specified topic using multiple models in swarm configuration with the agent_core metadata from the pandaDB"""
        pass