"""
TestAgent: Multimodal AI agent that integrates text, speech, and vision capabilities.
Provides a unified interface for processing and responding to various input types.
"""

import sys
import os
import pytest
import platform

# Add proper logging
from oarc.utils.log import log

# Add voice pack URL
VOICE_PACK_URL = "https://huggingface.co/Borcherding/XTTS-v2_C3PO/tree/main"

try:
    import gradio as gr
    from oarc.api import API
except ImportError as e:
    log.error(f"Failed to import API: {e}")
    # Provide a more graceful fallback
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)

from oarc.database import PandasDB
from oarc.promptModel import MultiModalPrompting
from oarc.speech import TextToSpeech, SpeechToText, SpeechManager
from oarc.yolo import YoloProcessor
from oarc.utils.paths import Paths
from oarc.speech.speech_errors import TTSInitializationError
from oarc.utils.speech_utils import SpeechUtils

from oarc.server.gradio import GradioServer, GradioServerAPI
from fastapi import Request

from oarc.utils.const import (
    SUCCESS, FAILURE
)

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

    def __init__(self):
        log.info("Initializing TestAgent components")
        
        # Log system info to help with debugging
        log.info(f"System: {platform.system()} {platform.release()} ({platform.platform()})")
        log.info(f"Python: {platform.python_version()}")
        
        # Log all configured paths for transparency and debugging
        # FIX: Use the correct method to get the Paths singleton instance
        paths = Paths()  # The singleton decorator will return the instance
        paths.log_paths()
        
        # Core components
        log.info("Initializing API components")
        if API is not None:
            self.api = API()
        else:
            self.api = None
            log.warning("API was not imported. Some functionality may be limited.")
        log.info("Initializing database")
        self.db = PandasDB()
        log.info("Initializing language model")
        self.llm = MultiModalPrompting()
        
        # Speech components initialization
        log.info("Setting up speech components")
        self.stt = SpeechToText()
        log.info("Speech-to-text component initialized")

        # Get TTS-related directories and verify voice reference file exists
        log.info("Setting up text-to-speech component")
        
        try:
            # Get TTS paths dictionary from our singleton Paths class - correctly access singleton
            tts_paths_dict = paths.get_tts_paths_dict()
            
            # Log the voice configuration
            voice_type = "xtts_v2"
            voice_name = "c3po"
            log.info(f"Configuring TTS with voice type '{voice_type}', voice '{voice_name}'")
            
            # Use the SpeechManager singleton
            from oarc.speech.speech_manager import SpeechManager
            self.tts = SpeechManager()
            log.info("Text-to-speech component initialized")
        except Exception as e:
            log.critical(f"Critical TTS initialization error: {str(e)}")
            log.critical("Cannot continue without proper TTS initialization. Exiting.")
            raise TTSInitializationError(str(e)) from e
        
        # Vision component initialization
        log.info("Setting up vision component")
        self.yolo = YoloProcessor.get_instance()
        log.info("Vision component initialized")
        
        # Agent configuration
        log.info("Setting up agent configuration")
        self.agent_config = {
            "agent_id": "test_agent",
            "models": {
                "largeLanguageModel": "llama2",
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
            self.db.storeAgent(self.agent_config)
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
        
        # Speech-to-text processing
        try:
            
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
                    self.db.conversation_handler,
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
        if self.api_server and self.api_server.is_running:
            self.api_server.stop()
            
        if self.gradio_server and self.gradio_server.is_running:
            self.gradio_server.stop()
            
        # Clean up component resources
        if hasattr(self, 'stt'):
            self.stt.cleanup()
            
        if hasattr(self, 'tts'):
            self.tts.cleanup()
            
        log.info("TestAgent cleanup complete")


def main():
    """Run the TestAgent application"""
    log.info("TestAgent script running...")
    
    try:
        # Initialize speech manager - this must succeed before we proceed 
        try:
            manager = SpeechManager()
        except (TTSInitializationError, FileNotFoundError) as e:
            log.critical(f"Failed to initialize SpeechManager: {e}")
            # Exit immediately on any SpeechManager error
            return FAILURE
        
        # Create agent normally with API enabled
        try:
            agent = TestAgent()
            api = API()
        except Exception as e:
            log.critical(f"Failed to initialize TestAgent: {e}", exc_info=True)
            return FAILURE
            
        agent.launch_gradio()
        return SUCCESS
        
    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
        return FAILURE
        
    finally:
        # Only cleanup if we have fully initialized objects
        if 'agent' in locals() and agent is not None and hasattr(agent, 'cleanup'):
            log.info("Cleaning up agent resources...")
            agent.cleanup()
            
        if 'manager' in locals() and manager is not None and hasattr(manager, 'cleanup'):
            log.info("Cleaning up speech manager resources...")
            manager.cleanup()

if __name__ == "__main__":
    sys.exit(main())