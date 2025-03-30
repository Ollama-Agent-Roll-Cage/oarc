# speechToText.py
import pyaudio
import speech_recognition as sr
import numpy as np
# import whisper
from whisper import load_model
import queue
import audioop
import wave
import tempfile
import os
import torch
import keyboard
import re
import json
import websockets
import asyncio
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, BackgroundTasks
from oarc.base_api import BaseToolAPI

class speechToText:
    def __init__(self):
        """Initialize speech recognizer with queues and improved wake word detection"""
        # Basic flags and settings
        self.is_listening = True
        self.is_active = True
        
        # Voice settings
        # self.wake_word = "Yo Jamie"  # Configurable wake word
        # self.wake_word = "Hey Rex"  # Configurable wake word
        self.wake_word = "Hey Echo"  # Configurable wake word
        
        self.recognizer = sr.Recognizer()
        self.use_wake_commands = False
        
        # Default to Google Speech Recognition
        self.use_whisper = False
        self.whisper_model = None
        
        # Queue system for audio processing
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Audio settings
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 2
        self.SILENCE_THRESHOLD = 605
        self.SILENCE_DURATION = 0.25
        
        # Audio buffers
        self.audio_data = np.array([])
        self.audio_buffer = []
        self.frames = []
        self.is_recording = False

    def listen(self, threshold=605, silence_duration=0.25):
        """Enhanced listen method with silence detection"""
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except IOError:
            print("Error: Could not access the microphone.")
            audio.terminate()
            return None

        frames = []
        silent_frames = 0
        sound_detected = False

        while True:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                rms = audioop.rms(data, 2)

                if rms > threshold:
                    silent_frames = 0
                    sound_detected = True
                else:
                    silent_frames += 1

                if sound_detected and (silent_frames * (self.CHUNK / self.RATE) > self.SILENCE_DURATION):
                    return

            except Exception as e:
                print(f"Error during recording: {e}")
                return

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if sound_detected:
            # Save to temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            # Send audio data to frontend for visualization
            asyncio.run(self.send_audio_to_frontend(temp_file.name, "stt"))
            
            return temp_file.name
        return None

    def recognizer(self, audio_input, model="google"):
        """Enhanced speech recognition with multiple backends"""
        if isinstance(audio_input, str):
            # Input is a file path
            if model == "whisper":
                if self.whisper_model is None:
                    try:
                        self.whisperSTT()
                    except Exception as e:
                        print(f"error loading Whisper: {e}")
                        speech_str = self.googleSTT(audio_input)      
                try:
                    transcript = self.whisper_model.transcribe(audio_input)
                    speech_str = transcript["text"]
                except Exception as e:
                    print(f"Whisper error: {e}")
                    speech_str = self.googleSTT(audio_input)
            else:
                speech_str = self.googleSTT(audio_input)
            
        return speech_str

    def whisperSTT(self, model_size="tiny"):
        """Optional method to enable and load Whisper with minimal model"""
        try:
            self.use_whisper = True
            self.whisper_model = load_model(model_size)
            print(f"Whisper {model_size} model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.use_whisper = False
            self.whisper_model = None
            return False
        
    def googleSTT(self, audio_input):
        """Use Google Speech Recognition"""
        try:
            if isinstance(audio_input, str):
                # Convert file to AudioData
                with sr.AudioFile(audio_input) as source:
                    audio_data = self.recognizer.record(source)
            else:
                audio_data = audio_input
                
            return self.recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition; {e}"

    def voiceArgSpaceFilter(self, input):
        # Use regex to replace all spaces with underscores and convert to lowercase for arg name consitency
        output = re.sub(' ', '_', input).lower()
        return output
    
    def set_wake_word(self, wake_word: str):
        """Set new wake word"""
        self.wake_word = wake_word.lower()
        
    def wait_for_wake_word(self):
        """Wait for wake word activation"""
        while True:
            temp_file = self.listen()
            if temp_file:
                try:
                    speech_text = self.recognize_speech(temp_file).lower()
                    
                    if self.wake_word in speech_text:
                        print("Wake word detected! Starting to listen...")
                        self.is_listening = True
                        return True
                        
                finally:
                    # Cleanup temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def start_continuous_listening(self):
        """Start continuous listening process"""
        self.is_listening = True
        while self.is_listening:
            temp_file = self.listen()
            if temp_file:
                try:
                    speech_text = self.recognize_speech(temp_file)
                    self.text_queue.put(speech_text)
                    
                    # Check for sleep commands
                    if any(phrase in speech_text.lower() for phrase in [
                        "what do you think echo", "please explain echo"
                    ]):
                        self.is_listening = False
                        break
                        
                finally:
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def silenceFiltering(self, audio_input, threshold):
        frames = []
        silent_frames = 0
        sound_detected = False

        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                rms = audioop.rms(data, 2)

                if rms > self.SILENCE_THRESHOLD:
                    silent_frames = 0
                    sound_detected = True
                else:
                    silent_frames += 1

                if sound_detected and (silent_frames * (self.CHUNK / self.RATE) > self.SILENCE_DURATION):
                    break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

            rms = audioop.rms(data, 2)

            if rms > threshold:
                silent_frames = 0
                sound_detected = True
            else:
                silent_frames += 1

            if sound_detected and (silent_frames * (self.CHUNK / self.RATE) > self.SILENCE_DURATION):
                return
            
    def cleanup(self):
        """Clean up resources"""
        self.is_listening = False
        self.is_active = False
        if self.whisper_model is not None:
            self.whisper_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def hotkeyRecognitionLoop(self):
        keyboard.add_hotkey('ctrl+shift', self.speech_recognizer_instance.auto_speech_set, args=(True, self.listen_flag))
        keyboard.add_hotkey('ctrl+alt', self.speech_recognizer_instance.chunk_speech, args=(True,))
        keyboard.add_hotkey('shift+alt', self.interrupt_speech)
        keyboard.add_hotkey('tab+ctrl', self.speech_recognizer_instance.toggle_wake_commands)
        
        while True:
            speech_done = False
            cmd_run_flag = False
            
            # check for speech recognition
            if self.listen_flag:
                # Wait for the key press to start speech recognition
                keyboard.wait('ctrl+shift')
                
                # Start speech recognition
                self.speech_recognizer_instance.auto_speech_flag = True
                while self.speech_recognizer_instance.auto_speech_flag:
                    try:
                        # Record audio from microphone
                        if self.listen_flag:
                            # Recognize speech to text from audio
                            if self.speech_recognizer_instance.use_wake_commands:
                                # using wake commands
                                user_input_prompt = self.speech_recognizer_instance.wake_words(audio)
                            else:
                                audio = self.speech_recognizer_instance.get_audio()
                                # using push to talk
                                user_input_prompt = self.speech_recognizer_instance.recognize_speech(audio)
                                
                            # print recognized speech
                            if user_input_prompt:
                                self.speech_recognizer_instance.chunk_flag = False
                                self.speech_recognizer_instance.auto_speech_flag = False
                                
                                #TODO: SEND RECOGNIZED SPEECH BACK TO CHATBOT WIZARD
                                # Filter voice commands and execute them if necessary
                                user_input_prompt = self.voice_command_select_filter(user_input_prompt)
                                cmd_run_flag = self.command_select(user_input_prompt)
                                
                                # Check if the listen flag is still on before sending the prompt to the model
                                if self.listen_flag and not cmd_run_flag:
                                    # Send the recognized speech to the model
                                    response = self.send_prompt(user_input_prompt)
                                    # Process the response with the text-to-speech processor
                                    response_processed = False
                                    if self.listen_flag is False and self.leap_flag is not None and isinstance(self.leap_flag, bool):
                                        if not self.leap_flag and not response_processed:
                                            self.tts_processor_instance.process_tts_responses(response, self.voice_name)
                                            response_processed = True
                                            if self.speech_interrupted:
                                                print("Speech was interrupted. Ready for next input.")
                                                self.speech_interrupted = False
                                return  # Exit the function after recognizing speech
                                
                    # google speech recognition error exception: inaudible sample
                    except sr.UnknownValueError:
                        print(self.colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + self.colors["OKCYAN"])
                    
                    # google speech recognition error exception: no connection
                    except sr.RequestError as e:
                        print(self.colors["OKCYAN"] + "Could not request results from Google Speech Recognition service; {0}".format(e) + self.colors["OKCYAN"])
                        
    async def send_audio_to_frontend(self, audio_file, audio_type):
        async with websockets.connect('ws://localhost:2020/audio-stream') as websocket:
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            await websocket.send(json.dumps({
                'audio_type': audio_type,
                'audio_data': list(audio_data)
            }))
     
class SpeechToTextAPI(BaseToolAPI):
    def __init__(self):
        super().__init__(prefix="/stt", tags=["speech-to-text"])
        self.stt_instance = None
    
    def setup_routes(self):
        @self.router.post("/recognize")
        async def recognize_speech(self, audio: UploadFile):
            try:
                # Initialize speech recognizer if not already created
                if not self.stt_instance:
                    self.stt_instance = speechToText()
                
                # Create temporary file to store the uploaded audio
                temp_file = None
                try:
                    # Create a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_file_path = temp_file.name
                    
                    # Write uploaded audio to the temporary file
                    audio_data = await audio.read()
                    with open(temp_file_path, 'wb') as f:
                        f.write(audio_data)
                    
                    # Send audio to frontend for visualization (optional)
                    await self.stt_instance.send_audio_to_frontend(temp_file_path, "stt")
                    
                    # Recognize speech from the file
                    if self.stt_instance.use_whisper:
                        text = await self.stt_instance.whisperSTT(temp_file_path)
                    else:
                        text = await self.stt_instance.googleSTT(temp_file_path)
                    
                    return {"text": text}
                    
                finally:
                    # Clean up the temporary file
                    if temp_file and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")
        
        @self.router.post("/listen")
        async def listen_for_speech(self, threshold: int = 605, silence_duration: float = 0.25):
            try:
                if not self.stt_instance:
                    self.stt_instance = speechToText()
                
                # Listen for speech
                audio_file = await self.stt_instance.listen(threshold, silence_duration)
                if not audio_file:
                    return {"status": "error", "message": "No speech detected"}
                
                # Recognize the speech
                text = await self.stt_instance.recognizer(audio_file)
                
                # Clean up the audio file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                
                return {"status": "success", "text": text}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Listening failed: {str(e)}")
        
        @self.router.post("/wake-word")
        async def set_wake_word(self, wake_word: str):
            try:
                if not self.stt_instance:
                    self.stt_instance = speechToText()
                
                self.stt_instance.set_wake_word(wake_word)
                return {"status": "success", "message": f"Wake word set to '{wake_word}'"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Setting wake word failed: {str(e)}")
        
        @self.router.post("/wait-for-wake-word")
        async def wait_for_wake_word(self):
            try:
                if not self.stt_instance:
                    self.stt_instance = speechToText()
                
                # This should be handled as a background task since it's blocking
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self.stt_instance.wait_for_wake_word)
                
                return {"status": "listening", "message": f"Listening for wake word '{self.stt_instance.wake_word}'"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Wake word detection failed: {str(e)}")
        
        @self.router.get("/models")
        async def get_available_models(self):
            try:
                return {
                    "models": [
                        {"id": "google", "name": "Google Speech Recognition", "description": "Cloud-based speech recognition"},
                        {"id": "whisper-tiny", "name": "Whisper Tiny", "description": "Lightweight local speech recognition"},
                        {"id": "whisper-base", "name": "Whisper Base", "description": "Basic local speech recognition"},
                        {"id": "whisper-small", "name": "Whisper Small", "description": "More accurate local speech recognition"}
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
