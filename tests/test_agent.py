"""
TestAgent: Multimodal AI agent that integrates text, speech, and vision capabilities.
Provides a unified interface for processing and responding to various input types.
"""

import gradio as gr
import logging
import traceback

from oarc.api import API
from oarc.database import PandasDB
from oarc.promptModel import MultiModalPrompting
from oarc.speech import TextToSpeech, SpeechToText
from oarc.yoloProcessor import YoloProcessor
from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class TestAgent:
    """TestAgent: Multimodal AI agent that integrates text, speech, and vision capabilities.
    Provides a unified interface for processing and responding to various input types.
    """

    def __init__(self):
        log.info("TestAgent script started")
        log.info("Initializing TestAgent components")
        
        # Initialize paths utility first
        self.paths = Paths()
        log.info(f"Using model directory: {self.paths.get_model_dir()}")
        
        # Core components
        log.info("Initializing API components")
        self.api = API()
        log.info("Initializing database")
        self.db = PandasDB()
        log.info("Initializing language model")
        self.llm = MultiModalPrompting()
        
        # Speech components initialization
        log.info("Setting up speech components")
        self.stt = SpeechToText()
        log.info("Speech-to-text component initialized")

        # Get TTS-related directories using the Paths utility
        log.info("Setting up text-to-speech component")
        self.tts = TextToSpeech(
            developer_tools_dict=self.paths.get_tts_paths_dict(),
            voice_type="xtts_v2",
            voice_name="c3po"
        )
        log.info("Text-to-speech component initialized")
        
        # Vision component initialization
        log.info("Setting up vision component")
        self.yolo = YoloProcessor()
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
            if self.agent_config["flags"]["TTS_FLAG"] and response_data["text"]:
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


    def launch_gradio(self):
        """Launch interactive Gradio UI for the agent"""
        log.info("Starting Gradio interface")
        
        try:
            with gr.Blocks(title="OARC Test Agent") as demo:
                gr.Markdown("# 🤖 OARC Multimodal Agent Demo")
                log.info("Gradio interface title and header set")
                
                with gr.Row():
                    with gr.Column():
                        # Input components
                        log.info("Setting up Gradio input components")
                        text_input = gr.Textbox(label="Text Input", placeholder="Type your question here...")
                        audio_input = gr.Audio(label="Speech Input", source="microphone")
                        image_input = gr.Image(label="Vision Input", source="webcam")
                        
                        submit_btn = gr.Button("Submit", variant="primary")
                    
                    with gr.Column():
                        # Output components
                        log.info("Setting up Gradio output components")
                        text_output = gr.Textbox(label="Agent Response")
                        audio_output = gr.Audio(label="Speech Output")
                        vision_output = gr.JSON(label="Vision Analysis")
                
                # Handle submission
                log.info("Setting up Gradio event handlers")
                submit_btn.click(
                    fn=self.process_input,
                    inputs=[text_input, audio_input, image_input],
                    outputs=[text_output, audio_output, vision_output]
                )
            
            log.info("Launching Gradio server on port 7860")
            demo.launch(server_port=7860)
            log.info("Gradio server started successfully")
            
        except Exception as e:
            log.critical(f"Failed to launch Gradio interface: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    try:
        agent = TestAgent()
        agent.launch_gradio()
    except Exception as e:
        log.critical(f"Fatal error: {e}\nTrace: {traceback.format_exc()}", exc_info=True)
        raise