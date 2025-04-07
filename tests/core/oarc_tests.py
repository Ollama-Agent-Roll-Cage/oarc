"""
Comprehensive OARC Integration Test

This script tests the integration of various OARC components including:
- Text-to-speech and speech-to-text functionality
- Multimodal prompting with LLaVA vision capabilities
- Agent storage and state management
- Command library execution

The test creates and configures an agent with speech and vision capabilities,
demonstrates voice interaction, image analysis, and command execution.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from oarc
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.const import SUCCESS, FAILURE

# Speech components
from oarc.speech.text_to_speech import TextToSpeech
from oarc.speech.speech_to_text import SpeechToText
from oarc.speech.speech_manager import SpeechManager

# Agent and database components
from oarc.database.agent_storage import AgentStorage
from oarc.database.pandas_db import PandasDB

# Prompting and command components
from oarc.promptModel.multi_modal_prompting import MultiModalPrompting
from oarc.wizards.flag_manger import FlagManager
from oarc.wizards.commandLibrary import commandLibrary

# Vision components
from oarc.yolo.processor import YoloProcessor

# Import the test harness
from tests.async_harness import AsyncTestHarness

# Test constants
TEST_AGENT_ID = "test_assistant"
TEST_VOICE_NAME = "c3po"  # Derived from HF_VOICE_REF_PACK_C3PO
TEST_VOICE_TYPE = "xtts_v2"
TEST_MODEL_NAME = "llava:8b-v1.6-mistral-7b-q5_K_M"

class OARCTests(AsyncTestHarness):
    """Integration test implementation for OARC components."""

    def __init__(self):
        """Initialize the OARC integration test harness."""
        super().__init__("OARC Integration")
    
    async def setup(self) -> bool:
        """Set up test environment and initialize components."""
        try:
            log.info(f"Setting up {self.test_name} test environment")
            
            # Get paths
            self.paths = Paths()
            self.paths.log_paths()
            
            # Set up test parameters
            self.agent_id = TEST_AGENT_ID
            self.voice_name = TEST_VOICE_NAME
            self.voice_type = TEST_VOICE_TYPE
            self.model_name = TEST_MODEL_NAME
            
            # Set up test directories
            self.output_dir = Path(self.paths.get_output_subdir("test_oarc"))
            self.output_dir.mkdir(exist_ok=True)
            self.test_image = await self.find_test_image()
            
            # Initialize components
            await self.setup_components()
            
            log.info("OARC Integration Test initialization complete")
            return True
            
        except Exception as e:
            log.error(f"Error in test setup: {e}", exc_info=True)
            return False
    
    async def find_test_image(self):
        """Find a suitable test image for vision testing."""
        # Look for a sample image in the project
        project_root = self.paths._paths['base']['project_root']
        
        # Try several common locations and image types
        possible_image_paths = [
            Path(project_root) / "tests" / "test_data" / "sample_image.jpg",
            Path(project_root) / "tests" / "test_data" / "sample_image.png",
            Path(project_root) / "tests" / "data" / "sample_image.jpg",
            Path(project_root) / "data" / "sample_image.jpg",
        ]
        
        # Return the first existing image
        for img_path in possible_image_paths:
            if img_path.exists():
                log.info(f"Found test image: {img_path}")
                return img_path
        
        # If no image found, return None
        log.warning("No test image found. Vision tests will be skipped.")
        return None
    
    async def setup_components(self):
        """Set up all OARC components for testing."""
        log.info("Setting up OARC components")
        
        # Get tool paths from Paths singleton
        tts_paths = self.paths.get_tts_paths_dict()
        
        # Initialize SpeechManager singleton
        self.speech_manager = SpeechManager(voice_name=self.voice_name, voice_type=self.voice_type)
        self.speech_manager.initialize_tts_model()
        
        # Initialize TextToSpeech
        self.tts = TextToSpeech(tts_paths, self.voice_type, self.voice_name)
        
        # Initialize SpeechToText
        self.stt = SpeechToText()
        
        # Initialize agent storage
        self.agent_storage = AgentStorage()
        self.pandas_db = PandasDB()
        
        # Initialize MultiModalPrompting
        self.multi_modal = MultiModalPrompting()
        
        # Initialize command library
        self.cmd_library = commandLibrary()
        self.cmd_library.updateCommandLibrary()
        
        # Initialize YoloProcessor for vision
        self.yolo = YoloProcessor()
        
        # Create FlagManager with all components
        self.flag_manager = FlagManager(
            agent_id=self.agent_id,
            command_library=self.cmd_library.command_library,
            new_state={"flags": {
                "TTS_FLAG": True,
                "STT_FLAG": True,
                "LLAVA_FLAG": True,
                "AUTO_SPEECH_FLAG": False
            }},
            tts_processor_instance=self.tts,
            speech_recognizer_instance=self.stt,
            yolo_processor_instance=self.yolo,
            large_language_model=self.model_name,
            current_dir=tts_paths['current_path'],
            parent_dir=tts_paths['parent_path'],
            speech_dir=tts_paths['speech_dir'],
            recognize_speech_dir=tts_paths['recognize_speech_dir'],
            generate_speech_dir=tts_paths['generate_speech_dir'],
            tts_voice_ref_wav_pack_path=tts_paths['tts_voice_ref_wav_pack_path_dir']
        )
        
        log.info("OARC components setup complete")

    async def teardown(self) -> None:
        """Clean up resources after tests."""
        log.info(f"Cleaning up {self.test_name} test environment")
        
        # Close TTS resources
        if hasattr(self, 'tts') and self.tts:
            self.tts.cleanup()
        
        # Close STT resources
        if hasattr(self, 'stt') and self.stt:
            self.stt.cleanup()
        
        # Close YOLO resources
        if hasattr(self, 'yolo') and self.yolo:
            self.yolo.cleanup()
        
        # Close speech manager resources
        if hasattr(self, 'speech_manager') and self.speech_manager:
            self.speech_manager.cleanup()

    async def test_agent_creation(self) -> bool:
        """Test creating and configuring an agent."""
        log.info("Testing agent creation")
        
        try:
            # Create or load test agent
            available_agents = await self.agent_storage.list_available_agents()
            
            if self.agent_id in available_agents:
                log.info(f"Using existing agent: {self.agent_id}")
                agent_config = self.agent_storage.load_agent(self.agent_id)
            else:
                log.info(f"Creating new agent: {self.agent_id}")
                self.agent_storage.create_agent_from_template(
                    template_name="assistant",
                    agent_id=self.agent_id,
                    custom_config={
                        "voice": {
                            "type": self.voice_type,
                            "name": self.voice_name
                        },
                        "model": self.model_name,
                        "flags": {
                            "TTS_FLAG": True,
                            "STT_FLAG": True,
                            "LLAVA_FLAG": True,
                            "AUTO_SPEECH_FLAG": False
                        }
                    }
                )
                agent_config = self.agent_storage.load_agent(self.agent_id)
            
            log.info(f"Agent configuration: {agent_config}")
            return agent_config is not None
            
        except Exception as e:
            log.error(f"Agent creation test failed: {e}", exc_info=True)
            return False

    async def test_text_to_speech(self) -> bool:
        """Test text-to-speech functionality."""
        log.info("Testing text-to-speech")
        
        try:
            test_text = "Hello! I am the OARC test assistant. I am demonstrating text to speech capabilities."
            
            # Test TTS speech generation
            output_file = self.output_dir / "tts_test_output.wav"
            
            log.info(f"Generating speech for: '{test_text}'")
            result = self.speech_manager.generate_speech_to_file(
                text=test_text,
                output_file=str(output_file),
                speed=1.0,
                language="en"
            )
            
            if result:
                log.info(f"Speech successfully generated to: {output_file}")
            else:
                log.error("Speech generation failed!")
            
            # Test TTS sentence splitting
            sentences = self.tts.split_into_sentences(test_text)
            log.info(f"Text split into {len(sentences)} sentences: {sentences}")
            
            return result
            
        except Exception as e:
            log.error(f"Text-to-speech test failed: {e}", exc_info=True)
            return False

    async def test_command_library(self) -> bool:
        """Test command library functionality."""
        log.info("Testing command library")
        
        try:
            # Get all available commands
            commands = self.cmd_library.command_library.keys()
            log.info(f"Available commands: {', '.join(commands)}")
            
            # Test agent state retrieval via flag manager
            agent_state = self.flag_manager.get_agent_state()
            log.info(f"Current agent state: {agent_state}")
            
            return len(commands) > 0
            
        except Exception as e:
            log.error(f"Command library test failed: {e}", exc_info=True)
            return False

    async def test_multimodal_prompting(self) -> bool:
        """Test multimodal prompting with text and vision."""
        log.info("Testing multimodal prompting")
        
        try:
            # Initialize chat
            self.multi_modal.initializeChat()
            
            # Prepare a sample text prompt
            text_prompt = "Hello, can you introduce yourself as an OARC assistant?"
            
            # Test text-only prompt
            log.info(f"Sending text prompt: '{text_prompt}'")
            
            # Set model first
            self.multi_modal.set_model(self.model_name)
            
            # Send simple text prompt
            response = await self.multi_modal.mod_prompt(text_prompt)
            
            log.info(f"Received response: '{response[:100]}...'")
            success = response and len(response) > 0
            
            # Test vision prompt if we have a test image
            if self.test_image and success:
                vision_prompt = "What do you see in this image?"
                log.info(f"Sending vision prompt with image: {self.test_image}")
                
                try:
                    # Load image file
                    with open(self.test_image, "rb") as f:
                        image_data = f.read()
                    
                    # Send multimodal prompt
                    vision_response = await self.multi_modal.llava_prompt(
                        user_input_prompt=vision_prompt,
                        user_screenshot_raw2=image_data,
                        llava_user_input_prompt=vision_prompt
                    )
                    
                    log.info(f"Received vision response: '{vision_response[:100]}...'")
                    success = success and vision_response and len(vision_response) > 0
                    
                except Exception as e:
                    log.error(f"Vision prompting failed: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            log.error(f"Multimodal prompting test failed: {e}")
            return False

    async def run_tests(self) -> bool:
        """Run all integration tests."""
        log.info("Starting OARC integration tests")
        
        try:
            # Step 1: Test agent creation
            self.results["Agent Creation"] = await self.test_agent_creation()
            if not self.results["Agent Creation"]:
                log.error("Agent creation test failed!")
                return False
            
            # Step 2: Test command library 
            self.results["Command Library"] = await self.test_command_library()
            if not self.results["Command Library"]:
                log.error("Command library test failed!")
                return False
            
            # Step 3: Test text-to-speech
            self.results["Text-to-Speech"] = await self.test_text_to_speech()
            if not self.results["Text-to-Speech"]:
                log.error("Text-to-speech test failed!")
                return False
            
            # Step 4: Test multimodal prompting
            self.results["Multimodal Prompting"] = await self.test_multimodal_prompting()
            if not self.results["Multimodal Prompting"]:
                log.error("Multimodal prompting test failed!")
                return False
            
            # All tests passed!
            log.info("All integration tests passed successfully!")
            return True
            
        except Exception as e:
            log.error(f"Integration tests failed with error: {e}")
            return False

# Use the async harness runner
if __name__ == "__main__":
    AsyncTestHarness.run(OARCTests)
