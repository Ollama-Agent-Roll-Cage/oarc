"""
Text-to-Speech API Module

This module provides a FastAPI-based interface for text-to-speech conversion using various voice models.
It supports:
- Basic speech synthesis from text with configurable voice selection
- Advanced synthesis options including speed and language preferences
- Voice listing capabilities to discover available synthetic voices
- Interrupt and cleanup mechanisms to manage ongoing speech generation

The API leverages the Paths utility for consistent directory management across the application
and follows best practices for error handling and resource management.
"""

import os

from oarc.base_api import BaseToolAPI

from fastapi import HTTPException
from oarc.base_api import BaseToolAPI
from oarc.utils.paths import Paths
from oarc.speech import TextToSpeech
from oarc.speech.tts_request import TTSRequest
from oarc.utils.log import log


class TextToSpeechAPI(BaseToolAPI):
    """FastAPI wrapper for text-to-speech functionality.
    
    Provides endpoints for speech synthesis, voice management, and
    resource control operations, with proper error handling and
    environment-aware resource management.
    """

    def __init__(self):
        """Initialize the TTS API with paths configuration and directory setup."""
        super().__init__(prefix="/tts", tags=["text-to-speech"])
        
        # Initialize core attributes
        self.tts_instance = None
        self.paths = Paths
        self.model_dir = self.paths.get_model_dir()
        
        # Get TTS paths and ensure they exist
        self.developer_tools_dict = self.paths.get_tts_paths_dict()
        self.paths.ensure_paths(self.developer_tools_dict)
        
        log.info(f"TextToSpeechAPI initialized with model directory: {self.model_dir}")

    def setup_routes(self):
        """Configure API endpoints for text-to-speech functionality."""
        
        @self.router.post("/synthesize")
        async def synthesize_speech(self, text: str, voice_name: str = "c3po"):
            """Generate speech audio from input text with specified voice.
            
            Args:
                text: The text to convert to speech
                voice_name: Name of the voice to use (default: c3po)
                
            Returns:
                dict: Audio data with sample rate
                
            Raises:
                HTTPException: If speech synthesis fails
            """
            try:
                if not self.tts_instance:
                    log.info(f"Creating TTS instance with voice: {voice_name}")
                    self.tts_instance = TextToSpeech(
                        developer_tools_dict=self.developer_tools_dict,
                        voice_type="xtts_v2", 
                        voice_name=voice_name
                    )
                
                log.info(f"Synthesizing speech for text: '{text[:30]}...'")
                audio_data = self.tts_instance.process_tts_responses(text, voice_name)
                if audio_data is not None:
                    log.info("Speech synthesis successful")
                    return {"audio_data": audio_data.tolist(), "sample_rate": self.tts_instance.sample_rate}
                else:
                    log.error("Failed to generate audio")
                    raise HTTPException(status_code=500, detail="Failed to generate audio")
                    
            except Exception as e:
                log.error(f"TTS synthesis failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))


        @self.router.post("/advanced-synthesize")
        async def advanced_synthesize(self, request: TTSRequest):
            """Generate speech with advanced configuration options.
            
            Args:
                request: TTSRequest object containing text, voice, and other parameters
                
            Returns:
                dict: Audio data with sample rate
                
            Raises:
                HTTPException: If advanced speech synthesis fails
            """
            try:
                if not self.tts_instance:
                    log.info(f"Creating TTS instance with voice: {request.voice_name}")
                    self.tts_instance = TextToSpeech(
                        developer_tools_dict=self.developer_tools_dict,
                        voice_type="xtts_v2", 
                        voice_name=request.voice_name
                    )
                
                log.info(f"Synthesizing speech with advanced options: '{request.text[:30]}...'")
                audio_data = self.tts_instance.process_tts_responses(request.text, request.voice_name)
                if audio_data is not None:
                    log.info("Advanced speech synthesis successful")
                    return {"audio_data": audio_data.tolist(), "sample_rate": self.tts_instance.sample_rate}
                else:
                    log.error("Failed to generate audio")
                    raise HTTPException(status_code=500, detail="Failed to generate audio")
                    
            except Exception as e:
                log.error(f"Advanced TTS synthesis failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))


        @self.router.get("/voices")
        async def list_voices(self):
            """Retrieve list of available voices for text-to-speech.
            
            Returns:
                dict: List of available voice names
                
            Raises:
                HTTPException: If voice listing fails
            """
            try:
                coqui_dir = self.paths.get_coqui_dir()
                
                voices = [d.replace('XTTS-v2_', '') for d in os.listdir(coqui_dir) 
                         if d.startswith('XTTS-v2_') and os.path.isdir(os.path.join(coqui_dir, d))]
                
                log.info(f"Found {len(voices)} voices: {', '.join(voices) if voices else 'None'}")
                return {"voices": voices}
                
            except Exception as e:
                log.error(f"Error listing voices: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        

        @self.router.post("/interrupt")
        async def interrupt_speech(self):
            """Interrupt ongoing speech synthesis.
            
            Returns:
                dict: Status message indicating success or warning
                
            Raises:
                HTTPException: If interruption fails
            """
            try:
                if self.tts_instance:
                    log.info("Interrupting speech synthesis")
                    self.tts_instance.interrupt_generation()
                    return {"status": "success", "message": "Speech interrupted"}
                else:
                    log.warning("No active TTS instance to interrupt")
                    return {"status": "warning", "message": "No active TTS instance to interrupt"}
                    
            except Exception as e:
                log.error(f"Error interrupting speech: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        
        @self.router.post("/cleanup")
        async def cleanup(self):
            """Clean up TTS resources and free memory.
            
            Returns:
                dict: Status message indicating success or warning
                
            Raises:
                HTTPException: If cleanup fails
            """
            try:
                if self.tts_instance:
                    log.info("Cleaning up TTS resources")
                    self.tts_instance.cleanup()
                    self.tts_instance = None
                    return {"status": "success", "message": "TTS resources cleaned up"}
                else:
                    log.warning("No active TTS instance to clean up")
                    return {"status": "warning", "message": "No active TTS instance to clean up"}
                    
            except Exception as e:
                log.error(f"Error cleaning up TTS: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

