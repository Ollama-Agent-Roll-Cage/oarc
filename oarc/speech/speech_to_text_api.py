# speechToText.py

import logging
import os
import tempfile

from fastapi import HTTPException, UploadFile, BackgroundTasks

from oarc.speech import SpeechToText
from oarc.base_api import BaseToolAPI

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class SpeechToTextAPI(BaseToolAPI):


    def __init__(self):
        log.info("Initializing SpeechToTextAPI")
        super().__init__(prefix="/stt", tags=["speech-to-text"])
        self.stt_instance = None
        log.info("SpeechToTextAPI initialized")
    

    def setup_routes(self):
        """Set up API routes for speech-to-text functionality"""
        log.info("Setting up SpeechToTextAPI routes")
        

        @self.router.post("/recognize")
        async def recognize_speech(self, audio: UploadFile):
            log.info(f"Speech recognition endpoint called with file: {audio.filename}")
            try:
                # Initialize speech recognizer if not already created
                if not self.stt_instance:
                    log.info("Creating new SpeechToText instance")
                    self.stt_instance = SpeechToText()
                
                # Create temporary file to store the uploaded audio
                temp_file = None
                try:
                    # Create a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_file_path = temp_file.name
                    log.info(f"Created temporary file for uploaded audio: {temp_file_path}")
                    
                    # Write uploaded audio to the temporary file
                    audio_data = await audio.read()
                    log.info(f"Read {len(audio_data)} bytes from uploaded audio")
                    
                    with open(temp_file_path, 'wb') as f:
                        f.write(audio_data)
                    
                    # Send audio to frontend for visualization (optional)
                    await self.stt_instance.send_audio_to_frontend(temp_file_path, "stt")
                    
                    # Recognize speech from the file
                    if self.stt_instance.use_whisper:
                        log.info("Using Whisper for speech recognition")
                        text = await self.stt_instance.whisperSTT(temp_file_path)
                    else:
                        log.info("Using Google for speech recognition")
                        text = await self.stt_instance.googleSTT(temp_file_path)
                    
                    log.info(f"Speech recognition result: '{text}'")
                    return {"text": text}
                    
                finally:
                    # Clean up the temporary file
                    if temp_file and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        log.info(f"Removed temporary file: {temp_file.name}")
                        
            except Exception as e:
                log.error(f"Speech recognition failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")
        

        @self.router.post("/listen")
        async def listen_for_speech(self, threshold: int = 605, silence_duration: float = 0.25):
            log.info(f"Listen endpoint called with threshold={threshold}, silence_duration={silence_duration}")
            try:
                if not self.stt_instance:
                    log.info("Creating new SpeechToText instance")
                    self.stt_instance = SpeechToText()
                
                # Listen for speech
                audio_file = await self.stt_instance.listen(threshold, silence_duration)
                if not audio_file:
                    log.info("No speech detected")
                    return {"status": "error", "message": "No speech detected"}
                
                log.info(f"Audio captured in file: {audio_file}")
                
                # Recognize the speech
                text = await self.stt_instance.recognizer(audio_file)
                log.info(f"Recognized text: '{text}'")
                
                # Clean up the audio file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                    log.info(f"Removed audio file: {audio_file}")
                
                return {"status": "success", "text": text}
                
            except Exception as e:
                log.error(f"Listening failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Listening failed: {str(e)}")
        

        @self.router.post("/wake-word")
        async def set_wake_word(self, wake_word: str):
            log.info(f"Set wake word endpoint called with wake_word='{wake_word}'")
            try:
                if not self.stt_instance:
                    log.info("Creating new SpeechToText instance")
                    self.stt_instance = SpeechToText()
                
                self.stt_instance.set_wake_word(wake_word)
                log.info(f"Wake word set to: '{wake_word}'")
                return {"status": "success", "message": f"Wake word set to '{wake_word}'"}
                
            except Exception as e:
                log.error(f"Setting wake word failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Setting wake word failed: {str(e)}")
        

        @self.router.post("/wait-for-wake-word")
        async def wait_for_wake_word(self):
            log.info("Wait for wake word endpoint called")
            try:
                if not self.stt_instance:
                    log.info("Creating new SpeechToText instance")
                    self.stt_instance = SpeechToText()
                
                # This should be handled as a background task since it's blocking
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self.stt_instance.wait_for_wake_word)
                log.info(f"Started background task to wait for wake word '{self.stt_instance.wake_word}'")
                
                return {"status": "listening", "message": f"Listening for wake word '{self.stt_instance.wake_word}'"}
                
            except Exception as e:
                log.error(f"Wake word detection failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Wake word detection failed: {str(e)}")
        
        
        @self.router.get("/models")
        async def get_available_models(self):
            log.info("Get available models endpoint called")
            try:
                models = [
                    {"id": "google", "name": "Google Speech Recognition", "description": "Cloud-based speech recognition"},
                    {"id": "whisper-tiny", "name": "Whisper Tiny", "description": "Lightweight local speech recognition"},
                    {"id": "whisper-base", "name": "Whisper Base", "description": "Basic local speech recognition"},
                    {"id": "whisper-small", "name": "Whisper Small", "description": "More accurate local speech recognition"}
                ]
                log.info(f"Returning {len(models)} available models")
                return {"models": models}
            except Exception as e:
                log.error(f"Failed to get available models: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        log.info("SpeechToTextAPI routes setup complete")
