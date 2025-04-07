"""
This script implements a real-time speech transcription and synthesis application with a graphical user interface.
It listens for a specific wake word ("alexa") to start processing audio input using Whisper for transcription,
and utilizes a language model via the ollama API to generate dynamic responses. Additionally, the application plays
synthesized speech responses using gTTS and pygame, while handling asynchronous tasks through multithreading.
The interface also features a search functionality to highlight transcribed text, making it a versatile tool
for interactive voice-based interactions.
"""
import os
import sys
import io
import queue
import threading
import tempfile
import wave
import audioop

# Set environment variable to hide pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# PyQt6 imports
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QTextCursor, QTextCharFormat, QTextDocument
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFrame, QVBoxLayout, QHBoxLayout, 
    QLineEdit, QPushButton, QTextEdit, QWidget
)

# External library imports
import whisper
import pyaudio
import ollama
from gtts import gTTS
import pygame

# Internal imports
from oarc.utils.log import log

os.environ["PATH"] += os.pathsep + r"ffmpeg\bin"

# Redirect print statements to our logging system
_original_print = print
def _logged_print(*args, **kwargs):
    message = " ".join(map(str, args))
    log.info(message)
    # Uncomment the following line if you still want the print to appear in the console
    # _original_print(*args, **kwargs)
print = _logged_print


# TODO THIS IS AN EXAMPLE OF A SILENCE WAKE WORD WE NEED TO MIGRATE ANYTHING THAT ISNT ALREADY INTO THE MAIN speechtoText.py and textToSpeech.py files
class TranscriptionApp(QMainWindow):

    text_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("bot1")
        self.setStyleSheet("background-color: black;")
        self.setWindowOpacity(0.7)
        self.setMinimumSize(100, 100)
        
        self.text_queue = queue.Queue()
        self.speech_queue = queue.Queue()
        self.is_listening = True
        self.is_active = True
        
        # Initialize pygame mixer for audio playback
        log.info("Initializing pygame mixer")
        pygame.mixer.init(frequency=24000)  # Higher frequency for faster playback
        
        # Set up the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Set up search frame
        self.search_frame = QFrame()
        self.search_frame.setStyleSheet("background-color: black;")
        self.search_layout = QHBoxLayout(self.search_frame)
        
        self.search_entry = QLineEdit()
        self.search_entry.setStyleSheet("background-color: black; color: white; font-size: 12pt;")
        self.search_entry.returnPressed.connect(self.search_text)
        
        self.search_button = QPushButton("Search")
        self.search_button.setStyleSheet("background-color: black; color: white; font-size: 12pt;")
        self.search_button.clicked.connect(self.search_text)
        
        self.search_layout.addWidget(self.search_entry)
        self.search_layout.addWidget(self.search_button)
        
        # Set up text display area
        self.text_box = QTextEdit()
        self.text_box.setStyleSheet("background-color: black; color: white; font-size: 17pt;")
        self.text_box.setReadOnly(True)
        
        # Add widgets to main layout
        self.main_layout.addWidget(self.search_frame)
        self.main_layout.addWidget(self.text_box)
        
        # Connect the signal to update text
        self.text_signal.connect(self.update_text_box)
        
        # Load whisper model
        self.model = whisper.load_model("tiny")
        
        # Start speech processing thread
        threading.Thread(target=self.process_speech_queue, daemon=True).start()
        
        # Start the recording and transcribing thread
        threading.Thread(target=self.wait_for_wake_word, daemon=True).start()
        
        # Start the UI update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(10)  # Check the queue every 10ms
        
        # Enable dragging the window
        self.dragging = False
        self.offset = None


    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # Updated enum
            self.dragging = True
            self.offset = event.position()  # Updated method


    def mouseMoveEvent(self, event):
        if self.dragging and self.offset:
            self.move(self.pos() + event.position().toPoint() - self.offset.toPoint())  # Updated position handling


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # Updated enum
            self.dragging = False


    def speak_text(self, text, lang='en'):
        if text.strip():
            log.info(f"Queuing speech: {text[:30]}...")
            self.speech_queue.put((text, lang))


    def process_speech_queue(self):
        while True:
            try:
                text, lang = self.speech_queue.get()
                try:
                    log.info(f"Converting text to speech: {text[:30]}...")
                    tts = gTTS(text=text, lang=lang, slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    
                    # Load and play the audio
                    pygame.mixer.music.unload()
                    pygame.mixer.music.load(fp)
                    pygame.mixer.music.play()
                    
                    log.info("Playing speech")
                    # Wait for the audio to finish playing
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(-1)
                    log.info("Speech playback complete")
                        
                except Exception as e:
                    log.error(f"Speech error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Queue processing error: {e}")


    def get_llama_response(self, text):
        log.info(f"Processing input: {text}")
        if "thanks alexa" in text.lower() or "thank you alexa" in text.lower() or "okay, thanks alexa" in text.lower():
            self.is_listening = False
            response_text = "It was my pleasure to assist you. If you need anything else, just say 'Alexa' to wake me up. Have a great day!"
            self.text_queue.put(response_text)
            self.speak_text(response_text)
        else:
            try:
                log.info(f"Getting response from LLM model llama3.2:1b")
                stream = ollama.chat(
                    model='llama3.2:1b',
                    messages=[{'role': 'user', 'content': text}],
                    stream=True,
                )

                sentence_buffer = ""
                for chunk in stream:
                    if 'message' in chunk and 'message' in chunk:
                        content = chunk['message']['content']
                        self.text_queue.put(content)
                        
                        sentence_buffer += content
                        
                        # Check for sentence endings
                        while any(end in sentence_buffer for end in '.!?'):
                            # Find the first sentence ending
                            end_indices = [sentence_buffer.find(end) for end in '.!?' if end in sentence_buffer]
                            first_end = min(i for i in end_indices if i != -1)
                            
                            # Extract the complete sentence
                            complete_sentence = sentence_buffer[:first_end + 1].strip()
                            if complete_sentence:
                                self.speak_text(complete_sentence)
                            
                            # Keep the remainder in the buffer
                            sentence_buffer = sentence_buffer[first_end + 1:].trip()
                
                # Speak any remaining text
                if sentence_buffer.strip():
                    self.speak_text(sentence_buffer)
                    
            except Exception as e:
                error_message = f"Error getting LLM response: {str(e)}\n"
                self.text_queue.put(error_message)
                return error_message


    def wait_for_wake_word(self):
        while True:
            temp_file = self.listen()
            if temp_file:
                transcript = self.transcribe(temp_file)
                text = transcript['text'].lower()
                
                if 'alexa' in text:
                    self.text_queue.put("Wake word detected! Starting to listen...\n")
                    self.speak_text("Hello, how can I help you?")
                    self.is_listening = True
                    self.record_and_transcribe()
                
                os.remove(temp_file)


    def search_text(self):
        search_term = self.search_entry.text().lower()
        if not search_term:
            return
            
        # Clear previous formatting
        cursor = self.text_box.textCursor()
        self.text_box.selectAll()
        format = self.text_box.currentCharFormat()
        format.setBackground(QColor("black"))
        format.setForeground(QColor("white"))
        cursor.setCharFormat(format)
        cursor.clearSelection()
        self.text_box.setTextCursor(cursor)
        
        # Search and highlight
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor("yellow"))
        highlight_format.setForeground(QColor("black"))
        
        cursor = self.text_box.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.text_box.setTextCursor(cursor)
        
        # Update find flags
        while self.text_box.find(search_term, QTextDocument.FindFlag.FindCaseSensitively):
            cursor = self.text_box.textCursor()
            cursor.mergeCharFormat(highlight_format)


    def listen(self, threshold=605, silence_duration=0.25):
        log.info(f"Starting audio listening with threshold={threshold}, silence_duration={silence_duration}")
        FORMAT = pyaudio.paInt32      
        CHANNELS = 1
        RATE = 44100
        CHUNK = 2

        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except IOError:
            log.error("Could not access the microphone.")
            audio.terminate()
            return None

        frames = []
        silent_frames = 0
        sound_detected = False

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            rms = audioop.rms(data, 2)

            if rms > threshold:
                silent_frames = 0
                sound_detected = True
            else:
                silent_frames += 1

            if sound_detected and (silent_frames * (CHUNK / RATE) > silence_duration):
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if sound_detected:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            return temp_file.name
        return None


    def transcribe(self, temp_file):
        if temp_file:
            transcript = self.model.transcribe(temp_file)
            try:
                return transcript
            except:
                pass
        return {"text": "", "language": "unknown"}


    def detectlang(self, transcript):
        detected_language = transcript.get('language', 'unknown')
        return detected_language


    @pyqtSlot(str)
    def update_text_box(self, text_data):
        self.text_box.moveCursor(QTextCursor.End)
        self.text_box.insertPlainText(text_data)
        self.text_box.ensureCursorVisible()


    def process_queue(self):
        try:
            while True:
                text_data = self.text_queue.get_nowait()
                self.text_signal.emit(text_data)
        except queue.Empty:
            pass


    def record_and_transcribe(self):
        while self.is_listening:
            temp_file = self.listen()
            if temp_file:
                transcript = self.transcribe(temp_file)
                lang = self.detectlang(transcript)

                output_text = f"{lang}\n{transcript['text']}\n-----------------------\n\n"
                self.text_queue.put(output_text)
                
                if "thanks alexa" in transcript['text'].lower() or "thank you alexa" in transcript['text'].lower() or "okay, thanks alexa" in transcript['text'].lower():
                    self.is_listening = False
                    self.text_queue.put("Sleep word detected! Waiting for wake word...\n")
                    break
                
                # Process LLM response in a separate thread
                def process_llm_response():
                    self.text_queue.put("LLM Response:\n")
                    self.get_llama_response(transcript['text'])
                    self.text_queue.put("\n" + "-"*50 + "\n")

                threading.Thread(target=process_llm_response, daemon=True).start()
                os.remove(temp_file)


def main():
    log.info("Starting TranscriptionApp")
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()