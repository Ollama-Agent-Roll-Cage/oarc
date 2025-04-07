"""
TestAgent: Multimodal AI agent that integrates text, speech, and vision capabilities.
Provides a unified interface for processing and responding to various input types.
"""
import os
import platform
import sys

import gradio as gr
from fastapi import Request

from oarc.database import PandasDB
from oarc.prompt import MultiModalPrompting
from oarc.server.gradio import GradioServer, GradioServerAPI
from oarc.speech import SpeechManager, SpeechToText
from oarc.speech.speech_errors import TTSInitializationError
from oarc.utils.const import FAILURE, SUCCESS
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.yolo.processor import YoloProcessor

# Test agent constants
TEST_AGENT_ID = "test_agent"
TEST_VOICE_NAME = "C3PO"  # Derived from HF_VOICE_REF_PACK_C3PO
TEST_VOICE_TYPE = "xtts_v2"
TEST_DEFAULT_MODEL = "llama2"

# Global instances that will be shared across the application
api_instance = None
speech_manager_instance = None
agent_instance = None

class TestAgentGradioServer(GradioServer):
    """
    Gradio server implementation for TestAgent.
    
    Creates and manages the Gradio web UI for the TestAgent, handling
    the interface layout and event registrations.
    """
    
    def __init__(self, test_agent, host="localhost", port=7860):
        """Initialize TestAgent's Gradio server."""        
        super().__init__(server_name="OARC Test Agent", host=host, port=port)
        self.test_agent = test_agent
    
    def _setup_layout(self):
        """Set up the TestAgent's Gradio interface layout."""
        log.info("Setting up TestAgent Gradio interface layout")
        
        with self.demo:
            gr.Markdown("# 🤖 OARC Multimodal Agent Demo")
            
            with gr.Row():
                with gr.Column():
                    # Input components
                    log.info("Setting up Gradio input components")
                    text_input = self.add_component("text_input", "Textbox", 
                                                  label="Text Input", 
                                                  placeholder="Type your question here...")
                    
                    audio_input = self.add_component("audio_input", "Audio", 
                                                   label="Speech Input",
                                                   type="filepath")
                    
                    image_input = self.add_component("image_input", "Image",
                                                   label="Vision Input")
                    
                    submit_btn = self.add_component("submit_btn", "Button",
                                                  value="Submit",
                                                  variant="primary")
                    
                with gr.Column():
                    # Output components
                    log.info("Setting up Gradio output components")
                    text_output = self.add_component("text_output", "Textbox",
                                                   label="Agent Response")
                    
                    audio_output = self.add_component("audio_output", "Audio",
                                                    label="Speech Output")
                    
                    vision_output = self.add_component("vision_output", "JSON",
                                                     label="Vision Analysis")
            
            # Register event handler for the submit button
            self.add_event_handler(
                trigger_component=submit_btn,
                fn=self.test_agent.process_input,
                inputs=[text_input, audio_input, image_input],
                outputs=[text_output, audio_output, vision_output]
            )


class TestAgentAPI(GradioServerAPI):
    """
    API server implementation for TestAgent.
    
    Provides REST API endpoints for programmatic access to TestAgent's
    functionality, including processing multimodal inputs.
    """
    
    def __init__(self, test_agent, host="localhost", port=7861):
        """Initialize TestAgent's API server."""        
        super().__init__(server_name="OARC Test Agent API", host=host, port=port)
        self.test_agent = test_agent
    
    def setup_routes(self):
        """Set up API routes for TestAgent."""
        super().setup_routes()
        
        # Add API endpoints specific to TestAgent
        @self.app.post("/api/process")
        async def process_input(request: Request):
            """Process multimodal inputs via API."""
            data = await request.json()
            result = await self.test_agent.process_input(
                text_input=data.get("text"),
                audio_input=data.get("audio"),
                image_input=data.get("image")
            )
            return result
        
        @self.app.get("/api/config")
        async def get_config():
            """Get the agent's current configuration."""
            return self.test_agent.agent_config


class TestAgent:
    """TestAgent: Multimodal AI agent that integrates text, speech, and vision capabilities.
    Provides a unified interface for processing and responding to various input types.
    """

    def __init__(self, api_instance=None):
        log.info("Initializing TestAgent components")
        
        # Log system info to help with debugging
        log.info(f"System: {platform.system()} {platform.release()} ({platform.platform()})")
        log.info(f"Python: {platform.python_version()}")
        
        # Get paths and configure output directory
        paths = Paths() # Get singleton instance
        paths.log_paths() # Log all paths for debugging

        # Set up output directory for test results
        output_dir = paths.get_output_dir()
        log.info(f"Using global output directory: {output_dir}")
        
        # Create test-specific output directory
        test_output_dir = os.path.join(output_dir, "test_agent")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Core components
        log.info("Initializing API components")
        self.api = api_instance  # Use the globally initialized API instance
        if self.api is None:
            log.warning("API instance not provided. Some functionality may be limited.")
            
        log.info("Initializing database")
        self.db = PandasDB()
        log.info("Initializing language model")
        self.llm = MultiModalPrompting()
        
        # Speech components initialization
        log.info("Setting up speech components")
        self.stt = SpeechToText()
        log.info("Speech-to-text component initialized")

        # Use the global speech manager instance for TTS
        log.info("Setting up text-to-speech component")
        global speech_manager_instance
        if speech_manager_instance is None:
            # Create a new instance if not provided
            voice_type = TEST_VOICE_TYPE
            voice_name = TEST_VOICE_NAME
            log.info(f"Configuring TTS with voice type '{voice_type}', voice '{voice_name}'")
            self.tts = SpeechManager(voice_name=voice_name, voice_type=voice_type)
            # Update global instance
            speech_manager_instance = self.tts
        else:
            # Use the existing global instance
            self.tts = speech_manager_instance
        log.info("Text-to-speech component initialized")
        
        # Vision component initialization
        log.info("Setting up vision component")
        self.yolo = YoloProcessor.get_instance()
        log.info("Vision component initialized")
        
        # Agent configuration
        log.info("Setting up agent configuration")
        self.agent_config = {
            "agent_id": TEST_AGENT_ID,
            "models": {
                "largeLanguageModel": TEST_DEFAULT_MODEL,
                "largeLanguageAndVisionAssistant": "llava",
                "yoloVision": "yolov8n"
            },
            "prompts": {
                "llmSystem": "You are a helpful assistant that can see, hear and speak.",
                "visionSystem": "Describe what you see in the image in detail."
            },
            "flags": {
                "TTS_FLAG": True,
                "STT_FLAG": True,
                "LLAVA_FLAG": True,
                "YOLO_FLAG": True
            }
        }
        log.info(f"Agent configuration complete: {self.agent_config['agent_id']}")
        
        try:
            log.info(f"Storing agent configuration for {self.agent_config['agent_id']}")
            self.db.store_agent(self.agent_config)
            log.info("Agent configuration stored successfully")
        except Exception as e:
            log.error(f"Failed to store agent configuration: {e}", exc_info=True)
        
        # Initialize servers
        self.gradio_server = None
        self.api_server = None
        
        log.info("TestAgent initialization complete")

    async def process_input(self, text_input=None, audio_input=None, image_input=None):
        """
        Process multimodal inputs and generate appropriate responses.
        
        Args:
            text_input: Text query from user
            audio_input: Audio file or recording
            image_input: Image for visual analysis
            
        Returns:
            Dictionary containing response data with text, audio, and vision results
        """
        log.info("Processing new multimodal input request")
        log.info(f"Input types - Text: {'Present' if text_input else 'None'}, "
                 f"Audio: {'Present' if audio_input else 'None'}, "
                 f"Image: {'Present' if image_input else 'None'}")
        
        response_data = {"text": None, "audio": None, "vision": None}
        
        try:
            # Speech-to-text processing    
            if audio_input is not None:
                log.info("Processing speech input")
                try:
                    text_input = await self.stt.recognizer(audio_input)
                    response_data["text"] = text_input
                    log.info(f"Speech recognition result: {text_input}")
                except Exception as e:
                    log.error(f"Speech recognition failed: {e}", exc_info=True)
                    if text_input is None:
                        text_input = "I couldn't understand the audio clearly."
            
            # Vision processing
            if image_input is not None:
                log.info("Processing image input")
                
                # YOLO object detection
                if self.agent_config["flags"]["YOLO_FLAG"]:
                    log.info("Running YOLO detection")
                    try:
                        detections = await self.yolo.process_frame(image_input)
                        log.info(f"YOLO detection complete with {len(detections) if detections else 0} objects")
                    except Exception as e:
                        log.error(f"YOLO detection failed: {e}", exc_info=True)
                        detections = None
                else:
                    detections = None
                    log.info("YOLO detection skipped (disabled in config)")
                
                # LLaVA vision-language processing
                if self.agent_config["flags"]["LLAVA_FLAG"]:
                    log.info("Running LLaVA vision-language model")
                    prompt = text_input if text_input else "What do you see in this image?"
                    try:
                        vision_response = await self.llm.llava_prompt(
                            prompt,
                            image_input,
                            self.agent_config["prompts"]["visionSystem"]
                        )
                        log.info("LLaVA processing complete")
                    except Exception as e:
                        log.error(f"LLaVA processing failed: {e}", exc_info=True)
                        vision_response = "Error analyzing the image."
                else:
                    vision_response = None
                    log.info("LLaVA processing skipped (disabled in config)")
                    
                response_data["vision"] = {
                    "llava": vision_response,
                    "yolo": detections
                }
                log.info("Vision processing complete")
            
            # Language model response generation
            log.info("Generating LLM response")
            try:
                llm_response = await self.llm.send_prompt(
                    self.agent_config,
                    self.db.handler,
                    {
                        "text": text_input,
                        "vision": response_data["vision"]
                    }
                )
                response_data["text"] = llm_response
                log.info("LLM response generated successfully")
            except Exception as e:
                log.error(f"LLM response generation failed: {e}", exc_info=True)
                response_data["text"] = "I'm sorry, I encountered an error generating a response."
            
            # Text-to-speech processing
            if self.agent_config["flags"]["TTS_FLAG"] and response_data["text"] != "I'm sorry, I encountered an error generating a response.":
                log.info("Converting text response to speech")
                try:
                    audio = await self.tts.process_tts_responses(response_data["text"])
                    response_data["audio"] = audio
                    log.info("Speech synthesis complete")
                except Exception as e:
                    log.error(f"Speech synthesis failed: {e}", exc_info=True)
            
            log.info("Multimodal processing complete, returning response")
            return response_data
            
        except Exception as e:
            log.critical(f"Error in process_input: {str(e)}", exc_info=True)
            return {"error": str(e), "text": "Sorry, an error occurred while processing your request."}


    def launch_gradio(self, web_port=7860, api_port=7861):
        """Launch interactive Gradio UI and API servers for the agent"""
        log.info("Starting Gradio interface and API servers")
        
        try:
            # Create and initialize the servers
            self.gradio_server = TestAgentGradioServer(self, port=web_port)
            self.api_server = TestAgentAPI(self, port=api_port)
            
            # Connect them for seamless integration
            self.api_server.connect_gradio_server(self.gradio_server)
            
            # Initialize both servers
            self.gradio_server.initialize()
            self.api_server.initialize()
            
            # Start the servers
            api_success = self.api_server.start()
            if api_success:
                log.info(f"API server running at {self.api_server.get_url()}")
            
            # The Gradio server will block as it runs in the main thread
            web_success = self.gradio_server.start()
            log.info(f"Web interface running at {self.gradio_server.get_url()}")
            
            return web_success and api_success
            
        except Exception as e:
            log.critical(f"Failed to launch servers: {e}", exc_info=True)
            raise

    def cleanup(self):
        """Clean up resources when shutting down"""
        log.info("Cleaning up TestAgent resources")
        
        # Stop servers if running
        if hasattr(self, 'api_server') and self.api_server and self.api_server.is_running:
            self.api_server.stop()
            
        if hasattr(self, 'gradio_server') and self.gradio_server and self.gradio_server.is_running:
            self.gradio_server.stop()
            
        # Clean up component resources
        if hasattr(self, 'stt'):
            self.stt.cleanup()
            
        log.info("TestAgent cleanup complete")


def main():
    """Run the TestAgent application"""
    log.info("TestAgent script running...")
    
    # Declare globals we'll use
    global api_instance, speech_manager_instance, agent_instance
    
    try:
        # Step 1: Initialize the API
        from oarc.api import API  # Defer import to avoid circular dependency issues
        api_instance = API()
        log.info("API initialized successfully")
        
        # Step 2: Initialize speech manager - this must succeed before we proceed 
        # SpeechManager will handle voice pack verification and download as needed
        voice_name = TEST_VOICE_NAME
        speech_manager_instance = SpeechManager(voice_name=voice_name)
        log.info(f"SpeechManager initialized with voice: {voice_name}")
        
        # Step 3: Initialize the TestAgent with API
        agent_instance = TestAgent(api_instance=api_instance)
        log.info("TestAgent initialized successfully")
        
        # Step 4: Launch the Gradio interface
        agent_instance.launch_gradio()
        log.info("Gradio interface launched")
        
        return SUCCESS
        
    except ImportError as e:
        log.critical(f"Failed to import required module: {e}", exc_info=True)
        return FAILURE
    except TTSInitializationError as e:
        log.critical(f"Failed to initialize SpeechManager: {e}")
        return FAILURE
    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
        return FAILURE
    finally:
        # Clean up resources in reverse order
        if agent_instance is not None and hasattr(agent_instance, 'cleanup'):
            log.info("Cleaning up agent resources...")
            agent_instance.cleanup()
        
        # Only clean up SpeechManager if it wasn't passed to TestAgent
        # (TestAgent will clean up its own copy)
        if speech_manager_instance is not None and hasattr(speech_manager_instance, 'cleanup'):
            if agent_instance is None or agent_instance.tts != speech_manager_instance:
                log.info("Cleaning up speech manager resources...")
                speech_manager_instance.cleanup()

if __name__ == "__main__":
    sys.exit(main())