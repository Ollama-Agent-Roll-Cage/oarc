"""TestAgent.py

    This script is used to test the OARC multimodal agent components.

created on: 3/5/2025
by @LeoBorcherding    
"""

from oarc.oarc_api import oarcAPI
from oarc.speechToSpeech import textToSpeech, speechToText
from oarc.promptModel import multiModalPrompting
from oarc.yoloProcessor import YoloProcessor
from oarc.pandasDB import PandasDB
import gradio as gr
import numpy as np
import sys
import os

class TestAgent:
    def __init__(self):
        # Initialize components
        self.api = oarcAPI()
        self.db = PandasDB()
        self.llm = multiModalPrompting()
        
        # Initialize speech components
        self.stt = speechToText()
        self.tts = textToSpeech(
            developer_tools_dict={
                'current_dir': os.getcwd(),
                'parent_dir': os.path.dirname(os.getcwd()),
                'speech_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'coqui'),
                'recognize_speech_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'whisper'),
                'generate_speech_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'generated'),
                'tts_voice_ref_wav_pack_path_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'coqui/voice_reference_pack')
            },
            voice_type="xtts_v2",
            voice_name="c3po"
        )
        
        # Initialize vision components
        self.yolo = YoloProcessor()
        
        # Set up agent configuration
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
        
        # Store agent config
        self.db.storeAgent(self.agent_config)
        
    async def process_input(self, text_input=None, audio_input=None, image_input=None):
        """Process multimodal input and return response"""
        try:
            response_data = {"text": None, "audio": None, "vision": None}
            
            # Process speech input
            if audio_input is not None:
                text_input = await self.stt.recognizer(audio_input)
                response_data["text"] = text_input
            
            # Process vision input
            vision_data = None
            if image_input is not None:
                # Get YOLO detections
                detections = await self.yolo.process_frame(image_input)
                # Get LLaVA description
                vision_response = await self.llm.llava_prompt(
                    text_input if text_input else "What do you see in this image?",
                    image_input,
                    self.agent_config["prompts"]["visionSystem"]
                )
                response_data["vision"] = {
                    "llava": vision_response,
                    "yolo": detections
                }
            
            # Generate LLM response
            llm_response = await self.llm.send_prompt(
                self.agent_config,
                self.db.conversation_handler,
                {
                    "text": text_input,
                    "vision": response_data["vision"]
                }
            )
            
            # Generate speech response
            if self.agent_config["flags"]["TTS_FLAG"]:
                audio = await self.tts.process_tts_responses(llm_response)
                response_data["audio"] = audio
            
            return response_data
            
        except Exception as e:
            return {"error": str(e)}

    def launch_gradio(self):
        """Launch Gradio interface"""
        with gr.Blocks(title="OARC Test Agent") as demo:
            gr.Markdown("# 🤖 OARC Multimodal Agent Demo")
            
            with gr.Row():
                with gr.Column():
                    # Input components
                    text_input = gr.Textbox(label="Text Input")
                    audio_input = gr.Audio(label="Speech Input", source="microphone")
                    image_input = gr.Image(label="Vision Input", source="webcam")
                    
                    # Submit button
                    submit_btn = gr.Button("Submit")
                
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
        
        # Launch interface
        demo.launch(server_port=7860)

if __name__ == "__main__":
    agent = TestAgent()
    agent.launch_gradio()