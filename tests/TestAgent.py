"""
TestAgent: Multimodal AI agent that integrates text, speech, and vision capabilities.
Provides a unified interface for processing and responding to various input types.
"""

import os
import gradio as gr
import logging  # Add standard logging

from oarc.api import API
from oarc.database import PandasDB
from oarc.promptModel import MultiModalPrompting
from oarc.speech import TextToSpeech, SpeechToText
from oarc.yoloProcessor import YoloProcessor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class TestAgent:


    def __init__(self):
        log.info("Initializing TestAgent components")
        
        # Core components
        self.api = API()
        self.db = PandasDB()
        self.llm = MultiModalPrompting()
        
        # Speech components initialization
        log.debug("Setting up speech components")
        self.stt = SpeechToText()

        # Define default base directory if OARC_MODEL_GIT is not set
        base_dir = os.getenv('OARC_MODEL_GIT')
        if base_dir is None:
            # Default to a "models" directory in the project root
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'models')
            os.makedirs(base_dir, exist_ok=True)
            log.warning(f"OARC_MODEL_GIT environment variable not set. Using default: {base_dir}")

        # Create subdirectories if they don't exist
        coqui_dir = os.path.join(base_dir, 'coqui')
        whisper_dir = os.path.join(base_dir, 'whisper')
        generated_dir = os.path.join(base_dir, 'generated')
        voice_ref_dir = os.path.join(coqui_dir, 'voice_reference_pack')

        for directory in [coqui_dir, whisper_dir, generated_dir, voice_ref_dir]:
            os.makedirs(directory, exist_ok=True)

        self.tts = TextToSpeech(
            developer_tools_dict={
                'current_dir': os.getcwd(),
                'parent_dir': os.path.dirname(os.getcwd()),
                'speech_dir': coqui_dir,
                'recognize_speech_dir': whisper_dir,
                'generate_speech_dir': generated_dir,
                'tts_voice_ref_wav_pack_path_dir': voice_ref_dir
            },
            voice_type="xtts_v2",
            voice_name="c3po"
        )
        
        # Vision component initialization
        log.debug("Setting up vision component")
        self.yolo = YoloProcessor()
        
        # Agent configuration
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
        
        # Store configuration in database
        log.info(f"Storing agent configuration for {self.agent_config['agent_id']}")
        self.db.storeAgent(self.agent_config)
        

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
        response_data = {"text": None, "audio": None, "vision": None}
        
        try:
            log.info("Processing multimodal input")
            
            # Speech-to-text processing
            if audio_input is not None:
                log.debug("Processing speech input")
                text_input = await self.stt.recognizer(audio_input)
                response_data["text"] = text_input
                log.info(f"Speech recognition result: {text_input}")
            
            # Vision processing
            if image_input is not None:
                log.debug("Processing image input")
                
                # YOLO object detection
                if self.agent_config["flags"]["YOLO_FLAG"]:
                    log.debug("Running YOLO detection")
                    detections = await self.yolo.process_frame(image_input)
                else:
                    detections = None
                
                # LLaVA vision-language processing
                if self.agent_config["flags"]["LLAVA_FLAG"]:
                    log.debug("Running LLaVA vision-language model")
                    prompt = text_input if text_input else "What do you see in this image?"
                    vision_response = await self.llm.llava_prompt(
                        prompt,
                        image_input,
                        self.agent_config["prompts"]["visionSystem"]
                    )
                else:
                    vision_response = None
                    
                response_data["vision"] = {
                    "llava": vision_response,
                    "yolo": detections
                }
                log.info("Vision processing complete")
            
            # Language model response generation
            log.debug("Generating LLM response")
            llm_response = await self.llm.send_prompt(
                self.agent_config,
                self.db.conversation_handler,
                {
                    "text": text_input,
                    "vision": response_data["vision"]
                }
            )
            
            response_data["text"] = llm_response
            log.info("LLM response generated")
            
            # Text-to-speech processing
            if self.agent_config["flags"]["TTS_FLAG"] and llm_response:
                log.debug("Converting text response to speech")
                audio = await self.tts.process_tts_responses(llm_response)
                response_data["audio"] = audio
                log.info("Speech synthesis complete")
            
            return response_data
            
        except Exception as e:
            log.error(f"Error processing input: {str(e)}", exc_info=True)
            return {"error": str(e), "text": "Sorry, an error occurred while processing your request."}


    def launch_gradio(self):
        """Launch interactive Gradio UI for the agent"""
        log.info("Starting Gradio interface")
        
        with gr.Blocks(title="OARC Test Agent") as demo:
            gr.Markdown("# 🤖 OARC Multimodal Agent Demo")
            
            with gr.Row():
                with gr.Column():
                    # Input components
                    text_input = gr.Textbox(label="Text Input", placeholder="Type your question here...")
                    audio_input = gr.Audio(label="Speech Input", source="microphone")
                    image_input = gr.Image(label="Vision Input", source="webcam")
                    
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column():
                    # Output components
                    text_output = gr.Textbox(label="Agent Response")
                    audio_output = gr.Audio(label="Speech Output")
                    vision_output = gr.JSON(label="Vision Analysis")
            
            # Handle submission
            submit_btn.click(
                fn=self.process_input,
                inputs=[text_input, audio_input, image_input],
                outputs=[text_output, audio_output, vision_output]
            )
        
        log.info("Launching Gradio server on port 7860")
        demo.launch(server_port=7860)

if __name__ == "__main__":
    agent = TestAgent()
    agent.launch_gradio()