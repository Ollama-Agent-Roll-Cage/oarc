"""
This module encapsulates the functionality to convert text into speech using Coqui TTS models.
It handles audio processing, model initialization (either fine-tuned or base with speaker reference),
and manages the generation, playback, and streaming of synthesized audio. The module also provides 
a Gradio interface for testing, robust sentence splitting, user interruption capabilities during audio playback,
and resource cleanup routines to ensure smooth performance.
"""

import os
import re
import json
import time
import queue
import shutil
import threading
from typing import Optional

import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import keyboard
import websockets

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.speech.speech_manager import SpeechManager

WEBSOCKET_URL = 'ws://localhost:2020/audio-stream'

class TextToSpeech:
    """ a class for managing the text to speech conversion.
    This class utilizes the SpeechManager singleton for TTS model management.
    """

    def __init__(self, developer_tools_dict, voice_type, voice_name):
        """
        Initialize the TTS processor.
        
        Args:
            developer_tools_dict: Dictionary containing path configurations
            voice_type: Type of voice to use (e.g., "xtts_v2")
            voice_name: Name of the voice to use (e.g., "c3po")
        """
        # Store initialization parameters
        self.developer_tools_dict = developer_tools_dict
        self.voice_type = voice_type
        self.voice_name = voice_name
        
        # Initialize paths using the Paths singleton
        self.paths = Paths()  # The singleton decorator will handle returning the instance
        
        # Get paths directly from the paths singleton
        tts_paths = self.paths.get_tts_paths_dict()
        self.current_dir = tts_paths['current_dir']
        self.parent_dir = tts_paths['parent_dir']
        self.speech_dir = tts_paths['speech_dir']
        self.recognize_speech_dir = tts_paths['recognize_speech_dir']
        self.generate_speech_dir = tts_paths['generate_speech_dir']
        self.tts_voice_ref_wav_pack_path = tts_paths['tts_voice_ref_wav_pack_path_dir']
        
        # Initialize other properties
        self.audio_data = None
        self.is_generating = False
        self.should_interrupt = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_interrupted = False
        
        # Get the colors dictionary (for terminal output)
        self.colors = {
            "RED": "\033[31m",
            "GREEN": "\033[32m",
            "YELLOW": "\033[33m",
            "BLUE": "\033[34m",
            "PURPLE": "\033[35m",
            "CYAN": "\033[36m",
            "LIGHT_CYAN": "\033[96m",
            "BRIGHT_YELLOW": "\033[93m",
            "RESET": "\033[0m"
        }
        
        # Get device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use SpeechManager for TTS instead of duplicating initialization
        self.speech_manager = SpeechManager(voice_name=self.voice_name, voice_type=self.voice_type)
        
        # Get TTS model and properties from SpeechManager
        self.tts = self.speech_manager.tts
        self.is_multi_speaker = self.speech_manager.is_multi_speaker
        self.voice_name_reference_speech_path = self.speech_manager.voice_ref_path
        self.sample_rate = self.speech_manager.sample_rate
        
        log.info(f"TextToSpeech initialized using SpeechManager with voice: {self.voice_name}")

    def process_tts_responses(self, response, voice_name):
        """Process text response into audio data suitable for streaming"""
        try:
            # Use the SpeechManager for speech generation to avoid duplicating code
            return self.speech_manager.generate_speech(response, speed=1.0, language="en")
            
        except Exception as e:
            log.error(f"Error generating audio: {e}")
            return None

    def play_audio_from_file(self, filename):
        """A method for audio playback from file."""
        # Check if the file exists
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist.")
            return

        try:
            # Load the audio file
            audio_data, sample_rate = sf.read(filename)

            # Play the audio file
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Failed to play audio from file {filename}. Reason: {e}")


    def generate_audio(self, sentence, ticker):
        """ a method to generate the audio for the chatbot
            args: sentence, voice_name_path, ticker
            returns: none
        """
        print(self.colors["LIGHT_CYAN"] + "🔊 generating audio for current sentence chunk 🔊:" + self.colors["RED"])
        # if self.is_multi_speaker:
        tts_audio = self.tts.tts(text=sentence, speaker_wav=self.voice_name_reference_speech_path, language="en", speed=3)
        # else:
        #     tts_audio = self.tts.tts(text=sentence, language="en", speed=3)

        # Convert to NumPy array (adjust dtype as needed)
        tts_audio = np.array(tts_audio, dtype=np.float32)

        # Save the audio with a unique name
        filename = os.path.join(self.generate_speech_dir, f"audio_{ticker}.wav")
        sf.write(filename, tts_audio, 22050)
    

    def get_audio_data(self):
        """Get the current audio buffer in a format suitable for streaming"""
        if len(self.audio_buffer) > 0:
            return {
                "sample_rate": self.sample_rate,
                "data": self.audio_buffer
            }
        return None
    

    def generate_play_audio_loop(self, tts_response_sentences):
        """ a method to generate and play the audio for the chatbot
            args: tts_sentences
            returns: none
        """
        # TODO /interrupt "mode" - added Shut up Feature - during audio playback loop, interupt model and allow user to overwride 
        # chat at current audio out to stop the model from talking and input new speech. 
        # Should probably make it better though, the interrupt loop doesnt function in the nextjs frontend 
        # through the api, it instead functions in the api terminal with hotkeys.
        
        # TODO /input audio "mode" "discord" - add, if modes "spacebar pressed". or "microphone input on" or "smart whisper prompting" with 
        # speech recognized and microphone "silence prompting" all as selections. Also add 2nd arg for discord audio, transcription.
        
        # TODO /decompose "mode" - pipe in Yolo, OCR, screen/game etc data decomposition, 
        # managing what data should be sent through text to speech and what should not
        
        # TODO /cut off speech "mode" - pipe interrupt data to write conversation history and mark/explain 
        # which audio chunk the user cut the model off at with the following modalities;
        #
        # (mode1) have model explain from there (always assume they didnt hear you), or 
        # (mode2) prompt the model with the marked conversation showcasing to the llm model/agent what 
        # the user did not hear through tts, but may have read on the screen text. 
        #
        # These modalities, essentially explaining to the agent wether or not the user can read the text, 
        # or can only hear, for the application of the system. This will provide for more smooth transitions 
        # for conversation, depending on modes 1 & 2.
        
        #TODO /smart listen "feature" - smart whisper/listen podcast moderator and fact checker.
        #
        #   feature set:
        #       wake commands/name call,
        #       long form whisper and audio chunking, storing processing,
        #       presume wether or not you are being spoken to, or listening to others
        #       when others are speaking, do not interrupt them, listen and whisper rec,
        #       while building conversation history, context, links, research, then
        #       forumalate possible response or responses, and when the conversation seems like
        #       both parties have felt they have been respected in their ability
        #       to uphold their free speech, the agent can jump in and say "hey guys! on
        #       that thought maybe you should try this?" 
        #
        #   tools:
        #       feature1: "longform whisper unless namecalled" on/off 
        #       feature2: "combine research data from conversation with /decompose"
        #       feature3: "listen and jump in not based on namecall, but instead based on
        #       respecting the conversation participants, and giving equal opporunities to
        #       contribute to the evolving (podcast) conversation"

        audio_queue = queue.Queue(maxsize=2)  # Queue to hold generated audio
        interrupted = False
        generation_complete = False


        def generate_audio_thread():
            nonlocal generation_complete
            for i, sentence in enumerate(tts_response_sentences):
                if interrupted:
                    break
                self.generate_audio(sentence, i)
                audio_queue.put(i)
            generation_complete = True

        # Start the audio generation thread
        threading.Thread(target=generate_audio_thread, daemon=True).start()

        ticker = 0
        while not (generation_complete and audio_queue.empty()) and not interrupted:
            try:
                current_ticker = audio_queue.get(timeout=0.1)
                filename = os.path.join(self.generate_speech_dir, f"audio_{current_ticker}.wav")
                
                play_thread = threading.Thread(target=self.play_audio_from_file, args=(filename,))
                play_thread.start()

                while play_thread.is_alive():
                    if keyboard.is_pressed('shift+alt'):
                        sd.stop()  # Stop the currently playing audio
                        interrupted = True
                        break
                    time.sleep(0.1)  # Small sleep to prevent busy-waiting

                play_thread.join()
                ticker += 1

            except queue.Empty:
                continue  # If queue is empty, continue waiting

        if interrupted:
            print(self.colors["BRIGHT_YELLOW"] + "Speech interrupted by user." + self.colors["RED"])
            self.clear_remaining_audio_files(ticker, len(tts_response_sentences))


    def interrupt_generation(self):
        """Interrupt ongoing speech generation"""
        self.speech_interrupted = True
        self.audio_buffer = np.array([], dtype=np.float32)
        

    def cleanup(self):
        """Clean up resources"""
        self.speech_interrupted = True
        self.audio_buffer = np.array([], dtype=np.float32)
        torch.cuda.empty_cache()
        

    def clear_remaining_audio_files(self, start_ticker, total_sentences):
        """ a method to clear the audio cache from the current splicing session
        """
        for i in range(start_ticker, total_sentences):
            filename = os.path.join(self.generate_speech_dir, f"audio_{i}.wav")
            if os.path.exists(filename):
                os.remove(filename)


    def clear_directory(self, directory):
        """ a method for clearing the given directory
            args: directory
            returns: none
        """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


    def split_into_sentences(self, text: str) -> list[str]:
        """A method for splitting the LLAMA response into sentences.
        Args:
            text (str): The input text.
        Returns:
            list[str]: List of sentences.

            #TODO split by * * as well, for roleplay text such as *ahem*, *nervous laughter*, etc. when this
            is sent to a diffusion model such as bark or f5, the tts will process them as emotional sounds,
            as well as handling them as seperate chunks split from the rest which will increase speed.
            
            #TODO retrain c3po with shorter sentences and more even volume distribution

            #TODO maximum split must be less than 250 token
                - no endless sentences, -> blocks of 11 seconds, if more the model will speed up to fit it in 
                that space where you control multiple generations, instead split out chunks and handle properly.
        """
        # Add spaces around punctuation marks for consistent splitting
        text = " " + text + " "
        text = text.replace("\n", " ")

        # Handle common abbreviations and special cases
        text = re.sub(r"(Mr|Mrs|Ms|Dr|i\.e)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)

        # Split on period, question mark, exclamation mark, or colon followed by optional spaces
        sentences = re.split(r"(?<=\d\.)\s+|(?<=[.!?:])\s+", text)

        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Combine the number with its corresponding sentence
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if re.match(r"^\d+\.", sentences[i]):
                combined_sentences.append(f"{sentences[i]} {sentences[i + 1]}")
                i += 2
            else:
                combined_sentences.append(sentences[i])
                i += 1

        # Ensure sentences are no longer than 250 characters
        final_sentences = []
        for sentence in combined_sentences:
            while len(sentence) > 250:
                # Find the nearest space before the 250th character
                split_index = sentence.rfind(' ', 0, 249)
                if split_index == -1:  # No space found, force split at 249
                    split_index = 249
                final_sentences.append(sentence[:split_index].strip())
                sentence = sentence[split_index:].strip()
            final_sentences.append(sentence)

        return final_sentences
    

    def gradio_interface(self):
        """Create Gradio interface for the detector"""
        import gradio as gr
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        def tts(text: str):
            # Process text
            audio_data = self.process_tts_responses(text, self.voice_name)
            if audio_data is not None:
                # Play audio
                sd.play(audio_data, self.sample_rate)
                sd.wait()
            return "Audio playback complete"

        gr.Interface(fn=tts, inputs="text", outputs="text").launch()


    async def send_audio_to_frontend(self, audio_data, audio_type):
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            await websocket.send(json.dumps({
                'audio_type': audio_type,
                'audio_data': list(audio_data)
            }))

    async def send_tts(self, text: str, voice_name: Optional[str] = None) -> bool:
        """Send TTS audio to the frontend"""
        try:
            # Use SpeechManager to generate speech
            speech_manager = SpeechManager(voice_name=voice_name if voice_name else "C3PO")
            audio_data = speech_manager.generate_speech(text, speed=1.0, language="en")
            if audio_data is not None:
                await self.send_audio_to_frontend(audio_data, "tts")
                return True
            return False
        except Exception as e:
            log.error(f"Error sending TTS audio: {e}")
            return False
