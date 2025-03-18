#TODO combine speechToText and textToSpeech apis into one class

from fastapi import FastAPI, APIRouter

class SpeechToSpeechAPI:
    def __init__(self):
        self.router = APIRouter()
        self.setup_routes()
    
    def setup_routes(self):
        @self.router.post("/recognize")
        async def recognize_speech(self, audio: UploadFile):
            pass
            
        @self.router.post("/synthesize") 
        async def synthesize_speech(self, text: str):
            pass