"""
This module provides a comprehensive suite for capturing, processing, and recognizing speech. 
It handles audio capture via microphone using PyAudio, applies silence filtering to detect active speech segments, 
and supports multiple speech recognition backends, including Google Speech Recognition and Whisper. 
Additional functionalities include wake word detection, hotkey controls for speech recognition, 
and integration with a frontend via websockets for audio visualization. 
The module also includes resource cleanup and utilities for seamless voice-command workflows.
"""

import asyncio
import audioop
import json
import logging
import os
import queue
import re
import tempfile
import wave

import keyboard
import numpy as np
import pyaudio
import speech_recognition as sr
import torch
import websockets
from whisper import load_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class SpeechToText:


    def __init__(self):
        """Initialize speech recognizer with queues and improved wake word detection"""
        log.info("Initializing SpeechToText")
        
        # Basic flags and settings
        self.is_listening = True
        self.is_active = True
        
        # Voice settings
        # self.wake_word = "Yo Jamie"  # Configurable wake word
        # self.wake_word = "Hey Rex"  # Configurable wake word
        self.wake_word = "Hey Echo"  # Configurable wake word
        log.info(f"Default wake word set to: '{self.wake_word}'")
        
        self.recognizer = sr.Recognizer()
        self.use_wake_commands = False
        
        # Default to Google Speech Recognition
        self.use_whisper = False
        self.whisper_model = None
        
        # Queue system for audio processing
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        log.info("Audio processing queues initialized")
        
        # Audio settings
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 2
        self.SILENCE_THRESHOLD = 605
        self.SILENCE_DURATION = 0.25
        log.info(f"Audio settings: RATE={self.RATE}, CHANNELS={self.CHANNELS}, SILENCE_THRESHOLD={self.SILENCE_THRESHOLD}")
        
        # Audio buffers
        self.audio_data = np.array([])
        self.audio_buffer = []
        self.frames = []
        self.is_recording = False
        
        log.info("SpeechToText initialization complete")


    def listen(self, threshold=605, silence_duration=0.25):
        """Enhanced listen method with silence detection"""
        log.info(f"Starting audio listening with threshold={threshold}, silence_duration={silence_duration}s")
        
        audio = pyaudio.PyAudio()
        try:
            log.info("Opening audio stream")
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except IOError as e:
            log.error(f"Failed to access microphone: {e}", exc_info=True)
            audio.terminate()
            return None

        frames = []
        silent_frames = 0
        sound_detected = False
        start_time = None
        
        log.info("Started audio capture loop")
        while True:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                rms = audioop.rms(data, 2)

                if rms > threshold:
                    if not sound_detected:
                        log.info(f"Sound detected (RMS: {rms} > threshold: {threshold})")
                        start_time = asyncio.get_event_loop().time()
                    silent_frames = 0
                    sound_detected = True
                else:
                    silent_frames += 1

                if sound_detected and (silent_frames * (self.CHUNK / self.RATE) > self.SILENCE_DURATION):
                    duration = asyncio.get_event_loop().time() - start_time if start_time else 0
                    log.info(f"Audio capture complete - duration: {duration:.2f}s")
                    break

            except Exception as e:
                log.error(f"Error during recording: {e}", exc_info=True)
                stream.stop_stream()
                stream.close()
                audio.terminate()
                return None

        stream.stop_stream()
        stream.close()
        audio.terminate()
        log.info("Audio stream closed")

        if sound_detected:
            # Save to temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            log.info(f"Saving audio to temporary file: {temp_file.name}")
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            # Send audio data to frontend for visualization
            try:
                asyncio.run(self.send_audio_to_frontend(temp_file.name, "stt"))
                log.info("Audio sent to frontend for visualization")
            except Exception as e:
                log.warning(f"Failed to send audio to frontend: {e}")
            
            return temp_file.name
        
        log.info("No sound detected above threshold")
        return None


    async def recognizer(self, audio_input, model="google"):
        """Enhanced speech recognition with multiple backends"""
        log.info(f"Starting speech recognition using model: {model}")
        
        if isinstance(audio_input, str):
            log.info(f"Input is file path: {audio_input}")
            # Input is a file path
            if model == "whisper":
                if self.whisper_model is None:
                    log.info("Whisper model not loaded, attempting to load")
                    try:
                        await self.whisperSTT()
                    except Exception as e:
                        log.error(f"Failed to load Whisper model: {e}", exc_info=True)
                        log.info("Falling back to Google STT")
                        speech_str = await self.googleSTT(audio_input)      
                try:
                    log.info("Processing audio with Whisper model")
                    transcript = self.whisper_model.transcribe(audio_input)
                    speech_str = transcript["text"]
                    log.info(f"Whisper recognition result: '{speech_str}'")
                except Exception as e:
                    log.error(f"Whisper recognition error: {e}", exc_info=True)
                    log.info("Falling back to Google STT")
                    speech_str = await self.googleSTT(audio_input)
            else:
                log.info("Using Google Speech Recognition")
                speech_str = await self.googleSTT(audio_input)
            
        else:
            log.info("Input is audio data object")
            if model == "whisper":
                # For Whisper we need a file, so save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                log.info(f"Saving audio data to temporary file for Whisper: {temp_file.name}")
                
                with wave.open(temp_file.name, 'wb') as wf:
                    # Write audio data to file
                    # This is simplified - in practice you'd need to extract the raw audio bytes properly
                    wf.write(audio_input.read())
                
                try:
                    if self.whisper_model is None:
                        await self.whisperSTT()
                    transcript = self.whisper_model.transcribe(temp_file.name)
                    speech_str = transcript["text"]
                    log.info(f"Whisper recognition result: '{speech_str}'")
                except Exception as e:
                    log.error(f"Whisper recognition error: {e}", exc_info=True)
                    speech_str = await self.googleSTT(audio_input)
                
                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                    log.info(f"Temporary file removed: {temp_file.name}")
                except Exception as e:
                    log.warning(f"Failed to remove temporary file: {e}")
            else:
                speech_str = await self.googleSTT(audio_input)
        
        log.info(f"Speech recognition complete: '{speech_str}'")
        return speech_str


    async def whisperSTT(self, model_size="tiny"):
        """Optional method to enable and load Whisper with minimal model"""
        log.info(f"Loading Whisper model: {model_size}")
        try:
            # Check if GPU is available for faster processing
            if torch.cuda.is_available():
                log.info("CUDA is available, using GPU for Whisper")
                device = "cuda"
            else:
                log.info("CUDA not available, using CPU for Whisper")
                device = "cpu"
                
            self.use_whisper = True
            self.whisper_model = load_model(model_size, device=device)
            log.info(f"Whisper {model_size} model loaded successfully on {device}")
            return True
        except Exception as e:
            log.error(f"Error loading Whisper model: {e}", exc_info=True)
            self.use_whisper = False
            self.whisper_model = None
            return False
        

    async def googleSTT(self, audio_input):
        """Use Google Speech Recognition"""
        log.info("Starting Google speech recognition")
        try:
            if isinstance(audio_input, str):
                log.info(f"Converting file to AudioData: {audio_input}")
                # Convert file to AudioData
                with sr.AudioFile(audio_input) as source:
                    audio_data = self.recognizer.record(source)
            else:
                log.info("Using provided audio data")
                audio_data = audio_input
                
            log.info("Sending audio to Google Speech Recognition API")
            result = self.recognizer.recognize_google(audio_data)
            log.info(f"Google Speech Recognition result: '{result}'")
            return result
        except sr.UnknownValueError:
            log.warning("Google Speech Recognition could not understand audio")
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            log.error(f"Could not request results from Google Speech Recognition: {e}", exc_info=True)
            return f"Could not request results from Google Speech Recognition; {e}"


    def voiceArgSpaceFilter(self, input):
        """Replace spaces with underscores and convert to lowercase for arg name consistency"""
        log.info(f"Filtering input: '{input}'")
        output = re.sub(' ', '_', input).lower()
        log.info(f"Filtered output: '{output}'")
        return output
    

    def set_wake_word(self, wake_word: str):
        """Set new wake word"""
        log.info(f"Changing wake word from '{self.wake_word}' to '{wake_word}'")
        self.wake_word = wake_word.lower()
        

    def wait_for_wake_word(self):
        """Wait for wake word activation"""
        log.info(f"Waiting for wake word: '{self.wake_word}'")
        while True:
            temp_file = self.listen()
            if temp_file:
                try:
                    speech_text = self.recognize_speech(temp_file).lower()
                    log.info(f"Heard: '{speech_text}'")
                    
                    if self.wake_word in speech_text:
                        log.info(f"Wake word '{self.wake_word}' detected! Starting to listen...")
                        self.is_listening = True
                        return True
                    else:
                        log.info("Wake word not detected, continuing to listen")
                        
                finally:
                    # Cleanup temp file
                    try:
                        os.remove(temp_file)
                        log.info(f"Temporary file removed: {temp_file}")
                    except Exception as e:
                        log.warning(f"Failed to remove temporary file: {e}")


    def start_continuous_listening(self):
        """Start continuous listening process"""
        log.info("Starting continuous listening mode")
        self.is_listening = True
        while self.is_listening:
            temp_file = self.listen()
            if temp_file:
                try:
                    speech_text = self.recognize_speech(temp_file)
                    log.info(f"Recognized: '{speech_text}'")
                    self.text_queue.put(speech_text)
                    
                    # Check for sleep commands
                    if any(phrase in speech_text.lower() for phrase in [
                        "what do you think echo", "please explain echo"
                    ]):
                        log.info("Sleep command detected, stopping continuous listening")
                        self.is_listening = False
                        break
                        
                finally:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        log.warning(f"Failed to remove temporary file: {e}")


    def silenceFiltering(self, audio_input, threshold):
        """Filter out silence from audio"""
        log.info(f"Performing silence filtering with threshold {threshold}")
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

            log.info("Started silence filtering audio capture loop")
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                rms = audioop.rms(data, 2)

                if rms > self.SILENCE_THRESHOLD:
                    silent_frames = 0
                    sound_detected = True
                    log.info(f"Sound detected (RMS: {rms} > threshold: {self.SILENCE_THRESHOLD})")
                else:
                    silent_frames += 1

                if sound_detected and (silent_frames * (self.CHUNK / self.RATE) > self.SILENCE_DURATION):
                    log.info("Silence detected after sound, stopping capture")
                    break
        finally:
            log.info("Closing audio stream")
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
                log.info("Silence filtering complete")
                return
            

    def cleanup(self):
        """Clean up resources"""
        log.info("Cleaning up SpeechToText resources")
        self.is_listening = False
        self.is_active = False
        
        if self.whisper_model is not None:
            log.info("Freeing Whisper model")
            self.whisper_model = None
            if torch.cuda.is_available():
                log.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
        
        log.info("Cleanup complete")
    
    
    def hotkeyRecognitionLoop(self):
        """Set up hotkeys for speech recognition control"""
        log.info("Setting up hotkey recognition loop")
        keyboard.add_hotkey('ctrl+shift', self.speech_recognizer_instance.auto_speech_set, args=(True, self.listen_flag))
        keyboard.add_hotkey('ctrl+alt', self.speech_recognizer_instance.chunk_speech, args=(True,))
        keyboard.add_hotkey('shift+alt', self.interrupt_speech)
        keyboard.add_hotkey('tab+ctrl', self.speech_recognizer_instance.toggle_wake_commands)
        
        log.info("Hotkeys registered. Starting hotkey recognition loop")
        while True:
            speech_done = False
            cmd_run_flag = False
            
            # check for speech recognition
            if self.listen_flag:
                # Wait for the key press to start speech recognition
                log.info("Waiting for ctrl+shift hotkey to start speech recognition")
                keyboard.wait('ctrl+shift')
                log.info("Hotkey detected, starting speech recognition")
                
                # Start speech recognition
                self.speech_recognizer_instance.auto_speech_flag = True
                while self.speech_recognizer_instance.auto_speech_flag:
                    try:
                        # Record audio from microphone
                        if self.listen_flag:
                            # Recognize speech to text from audio
                            if self.speech_recognizer_instance.use_wake_commands:
                                # using wake commands
                                log.info("Using wake commands mode")
                                user_input_prompt = self.speech_recognizer_instance.wake_words(audio)
                            else:
                                log.info("Using push-to-talk mode")
                                audio = self.speech_recognizer_instance.get_audio()
                                # using push to talk
                                user_input_prompt = self.speech_recognizer_instance.recognize_speech(audio)
                                
                            # print recognized speech
                            if user_input_prompt:
                                log.info(f"Recognized speech: '{user_input_prompt}'")
                                self.speech_recognizer_instance.chunk_flag = False
                                self.speech_recognizer_instance.auto_speech_flag = False
                                
                                #TODO: SEND RECOGNIZED SPEECH BACK TO CHATBOT WIZARD
                                # Filter voice commands and execute them if necessary
                                user_input_prompt = self.voice_command_select_filter(user_input_prompt)
                                cmd_run_flag = self.command_select(user_input_prompt)
                                
                                # Check if the listen flag is still on before sending the prompt to the model
                                if self.listen_flag and not cmd_run_flag:
                                    # Send the recognized speech to the model
                                    log.info("Sending recognized speech to model")
                                    response = self.send_prompt(user_input_prompt)
                                    # Process the response with the text-to-speech processor
                                    response_processed = False
                                    if self.listen_flag is False and self.leap_flag is not None and isinstance(self.leap_flag, bool):
                                        if not self.leap_flag and not response_processed:
                                            log.info(f"Converting response to speech: '{response}'")
                                            self.tts_processor_instance.process_tts_responses(response, self.voice_name)
                                            response_processed = True
                                            if self.speech_interrupted:
                                                log.info("Speech was interrupted. Ready for next input.")
                                                self.speech_interrupted = False
                                return  # Exit the function after recognizing speech
                                
                    # google speech recognition error exception: inaudible sample
                    except sr.UnknownValueError:
                        log.warning("Google Speech Recognition could not understand audio")
                    
                    # google speech recognition error exception: no connection
                    except sr.RequestError as e:
                        log.error(f"Could not request results from Google Speech Recognition: {e}", exc_info=True)
                        

    async def send_audio_to_frontend(self, audio_file, audio_type):
        """Send captured audio to frontend via websocket"""
        log.info(f"Sending audio to frontend, type: {audio_type}, file: {audio_file}")
        try:
            async with websockets.connect('ws://localhost:2020/audio-stream') as websocket:
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                log.info(f"Audio file read, size: {len(audio_data)} bytes")
                
                await websocket.send(json.dumps({
                    'audio_type': audio_type,
                    'audio_data': list(audio_data)
                }))
                log.info("Audio data sent to frontend")
        except Exception as e:
            log.error(f"Error sending audio to frontend: {e}", exc_info=True)
     
