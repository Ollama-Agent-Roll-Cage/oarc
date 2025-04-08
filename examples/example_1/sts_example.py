#!/usr/bin/env python3
"""
OARC Speech-to-Speech Chat Interface

This application demonstrates a complete speech-to-speech workflow using the OARC package:
1. Speech recognition (STT) via OARC's SpeechToText
2. LLM processing via Ollama
3. Text-to-speech (TTS) synthesis using OARC's C3PO voice
4. Conversation history storage with PandasDB

The application provides both text and voice interfaces, allowing users to
interact with AI models through speech or keyboard input.
"""

import os
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QTextEdit, QLabel, QComboBox, QSlider, QProgressBar,
    QMessageBox, QSplitter, QFrame, QScrollArea, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QColor, QTextCursor, QIcon, QFont

# OARC imports
from oarc.speech import SpeechToText
from oarc.speech import TextToSpeech
from oarc.speech import SpeechManager
from oarc.ollama.utils.ollama_commands import OllamaCommands
from oarc.database.pandas_db import PandasDB
from oarc.prompt import MultiModalPrompting
from oarc.utils.log import log
from oarc.utils.paths import Paths


class MessageBubble(QFrame):
    """Widget to display a single message in chat-like bubble style."""
    
    def __init__(self, message, is_user=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        
        # Set styles based on message sender
        if is_user:
            self.setStyleSheet(
                "background-color: #DCF8C6; border-radius: 10px; padding: 10px; margin: 5px;"
            )
            alignment = Qt.AlignmentFlag.AlignRight
        else:
            self.setStyleSheet(
                "background-color: #FFFFFF; border-radius: 10px; padding: 10px; margin: 5px;"
            )
            alignment = Qt.AlignmentFlag.AlignLeft
        
        # Layout and label for message content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add timestamp 
        timestamp = QLabel(datetime.now().strftime("%H:%M"))
        timestamp.setStyleSheet("color: #888888; background-color: transparent;")
        timestamp.setAlignment(alignment)
        
        # Message text
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("background-color: transparent;")
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Add widgets to layout
        layout.addWidget(message_label)
        layout.addWidget(timestamp)


class OARCSpeechChat(QMainWindow):
    """Main application window for speech-to-speech interface."""
    
    # Define signals
    message_signal = pyqtSignal(str, bool)  # Message text, is_user flag
    status_signal = pyqtSignal(str)  # Status message
    progress_signal = pyqtSignal(int, str)  # Progress percentage, status text
    
    def __init__(self):
        super().__init__()
        
        # Initialize state
        self.is_listening = False
        self.is_speaking = False
        self.ollama_model = "llama3"  # Default model
        self.voice_name = "C3PO"  # Default voice
        self.setup_components()
        self.setup_ui()
        
        # Set up queues for threaded operations
        self.speech_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # Start background threads
        self.speech_thread = threading.Thread(target=self.speech_generation_thread, daemon=True)
        self.speech_thread.start()
        
        # Set up timer for regular UI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.process_queues)
        self.update_timer.start(100)  # 100ms interval
        
        # Process message for UI update from different threads
        self.message_signal.connect(self.add_message_bubble)
        self.status_signal.connect(self.update_status)
        self.progress_signal.connect(self.update_progress)
        
        log.info("OARC Speech Chat initialized")
    
    def setup_components(self):
        """Initialize OARC components."""
        try:
            # Initialize PandasDB for conversation storage
            log.info("Initializing PandasDB...")
            self.db = PandasDB()
            self.db.init_conversation()
            
            # Initialize STT component
            log.info("Initializing Speech-to-Text...")
            self.stt = SpeechToText()
            
            # Initialize TTS component with C3PO voice
            log.info("Initializing Text-to-Speech...")
            self.speech_manager = SpeechManager(voice_name=self.voice_name, voice_type="xtts_v2")
            
            # Initialize Ollama interface
            log.info("Initializing Ollama Commands...")
            self.ollama = OllamaCommands()
            
            # Get available models
            self.available_models = self.ollama.list_local_models()
            if not self.available_models:
                log.warning("No Ollama models found locally")
                self.available_models = ["llama3", "llama2", "mistral"]
                
            # Initialize MultiModalPrompting
            log.info("Initializing MultiModalPrompting...")
            self.prompt_manager = MultiModalPrompting()
            
            log.info("All components initialized successfully")
            
        except Exception as e:
            log.error(f"Error initializing components: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Initialization Error", 
                                f"Failed to initialize OARC components: {str(e)}")
    
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("OARC Speech-to-Speech Chat")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Title section
        title_label = QLabel("OARC Speech-to-Speech Chat")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for chat and controls
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter, 1)
        
        # Chat area with scroll
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        # Add spacer at the bottom to push content up
        self.scroll_layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        
        self.scroll_area.setWidget(self.scroll_content)
        chat_layout.addWidget(self.scroll_area)
        
        # Controls panel
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Ollama Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.available_models)
        model_layout.addWidget(self.model_combo)
        
        # Voice selection
        model_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["C3PO", "Darth Vader", "Yoda"])  # Add your available voices
        self.voice_combo.currentTextChanged.connect(self.change_voice)
        model_layout.addWidget(self.voice_combo)
        
        controls_layout.addLayout(model_layout)
        
        # Input area
        input_layout = QHBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your message here or use voice input...")
        self.text_input.setMaximumHeight(80)
        input_layout.addWidget(self.text_input, 1)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        self.listen_button = QPushButton("🎤 Listen")
        self.listen_button.setCheckable(True)
        self.listen_button.clicked.connect(self.toggle_listening)
        button_layout.addWidget(self.listen_button)
        
        self.send_button = QPushButton("📤 Send")
        self.send_button.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_button)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.clicked.connect(self.clear_conversation)
        button_layout.addWidget(self.clear_button)
        
        input_layout.addLayout(button_layout)
        controls_layout.addLayout(input_layout)
        
        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        controls_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("System ready")
        self.status_label.setStyleSheet("color: #555555;")
        controls_layout.addWidget(self.status_label)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(controls_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 200])
        
        # Add initial welcome message
        self.add_message_bubble(
            "Hello! I'm your OARC AI assistant. You can speak to me by clicking the Listen button, "
            "or type your message and click Send. How can I help you today?", 
            False
        )
    
    def toggle_listening(self, checked):
        """Toggle speech recognition on/off."""
        if checked:
            self.is_listening = True
            self.listen_button.setText("🛑 Stop")
            self.status_signal.emit("Listening for speech...")
            self.progress_signal.emit(0, "Listening")
            
            # Start listening in a separate thread
            threading.Thread(target=self.listen_for_speech, daemon=True).start()
        else:
            self.is_listening = False
            self.listen_button.setText("🎤 Listen")
            self.status_signal.emit("Listening stopped")
            self.progress_signal.emit(0, "Ready")
    
    def listen_for_speech(self):
        """Thread function to capture and process speech."""
        try:
            self.status_signal.emit("Listening for speech...")
            
            # Use OARC's SpeechToText to listen for speech
            audio_data = self.stt.listen(threshold=600, silence_duration=0.5)
            if not self.is_listening:
                return  # User stopped listening
            
            self.status_signal.emit("Processing speech...")
            self.progress_signal.emit(50, "Transcribing")
            
            # Transcribe the captured audio
            transcribed_text = self.stt.transcribe(audio_data)
            if transcribed_text:
                self.status_signal.emit("Speech recognized!")
                self.progress_signal.emit(100, "Recognized")
                
                # Update UI with transcribed text
                self.text_input.setText(transcribed_text)
                
                # Auto-send the message
                self.send_message()
            else:
                self.status_signal.emit("Could not recognize speech")
                self.progress_signal.emit(0, "Ready")
                
        except Exception as e:
            log.error(f"Error in speech recognition: {str(e)}", exc_info=True)
            self.status_signal.emit(f"Error: {str(e)}")
            self.progress_signal.emit(0, "Error")
        finally:
            # Reset state
            self.listen_button.setChecked(False)
            self.is_listening = False
            self.listen_button.setText("🎤 Listen")
    
    def send_message(self):
        """Send the current text input to the AI and process response."""
        message = self.text_input.toPlainText().strip()
        if not message:
            return
        
        # Add user message to UI
        self.message_signal.emit(message, True)
        
        # Store in database
        self.db.store_message("user", message)
        
        # Clear input field
        self.text_input.clear()
        
        # Process in background thread
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        """Process user message and generate AI response."""
        try:
            self.status_signal.emit("Thinking...")
            self.progress_signal.emit(30, "Processing")
            
            # Get selected model
            model = self.model_combo.currentText()
            
            # Get conversation history
            history = self.db.get_conversation_history(limit=10)
            
            # Format prompt with conversation history
            prompt = self.prompt_manager.format_chat_prompt(
                user_message=message,
                history=history,
                system_prompt="You are a helpful AI assistant."
            )
            
            # Get response from Ollama
            self.progress_signal.emit(50, "Getting response")
            response = self.ollama.generate(
                model_name=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=1024
            )
            
            if not response:
                self.status_signal.emit("Error: No response from Ollama")
                self.progress_signal.emit(0, "Error")
                return
            
            # Extract just the AI's message from the response
            if isinstance(response, dict) and 'response' in response:
                ai_message = response['response']
            else:
                ai_message = str(response)
            
            # Add AI message to UI
            self.message_signal.emit(ai_message, False)
            
            # Store in database
            self.db.store_message("assistant", ai_message)
            
            # Convert to speech
            self.status_signal.emit("Generating speech...")
            self.progress_signal.emit(75, "Generating speech")
            
            # Add to speech queue
            self.speech_queue.put(ai_message)
            
        except Exception as e:
            log.error(f"Error processing message: {str(e)}", exc_info=True)
            self.status_signal.emit(f"Error: {str(e)}")
            self.progress_signal.emit(0, "Error")
    
    def speech_generation_thread(self):
        """Background thread for generating speech audio."""
        while True:
            try:
                # Get next text from queue
                text = self.speech_queue.get()
                self.is_speaking = True
                
                # Generate speech using OARC's SpeechManager
                audio_data = self.speech_manager.generate_speech(
                    text=text,
                    speed=1.0,
                    language="en"
                )
                
                # Save to temporary file and play
                temp_dir = os.path.join(Paths().get_output_dir(), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f"speech_{int(time.time())}.wav")
                
                # Save audio data to file
                import soundfile as sf
                sf.write(temp_file, audio_data, self.speech_manager.sample_rate)
                
                # Play audio in appropriate platform-specific way
                if sys.platform == "win32":
                    os.startfile(temp_file)
                elif sys.platform == "darwin":  # macOS
                    from subprocess import call
                    call(["afplay", temp_file])
                else:  # Linux
                    from subprocess import call
                    call(["aplay", temp_file])
                
                # Update progress when done
                self.progress_signal.emit(100, "Speech complete")
                time.sleep(2)  # Keep progress visible briefly
                self.progress_signal.emit(0, "Ready")
                
            except Exception as e:
                log.error(f"Error in speech generation: {str(e)}", exc_info=True)
                self.status_signal.emit(f"Speech error: {str(e)}")
            
            finally:
                self.speech_queue.task_done()
                self.is_speaking = False
    
    def process_queues(self):
        """Process any pending queue messages."""
        # This is called by timer to process updates from background threads
        pass
    
    def change_voice(self, voice_name):
        """Change the TTS voice."""
        try:
            self.voice_name = voice_name
            # Reinitialize speech manager with new voice
            self.speech_manager = SpeechManager(voice_name=voice_name, voice_type="xtts_v2")
            self.status_signal.emit(f"Voice changed to {voice_name}")
        except Exception as e:
            log.error(f"Error changing voice: {str(e)}", exc_info=True)
            self.status_signal.emit(f"Voice change error: {str(e)}")
    
    def clear_conversation(self):
        """Clear the conversation history."""
        reply = QMessageBox.question(
            self, 
            "Clear Conversation", 
            "Are you sure you want to clear the entire conversation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear UI
            for i in reversed(range(self.scroll_layout.count())):
                item = self.scroll_layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), MessageBubble):
                    item.widget().deleteLater()
            
            # Add spacer back
            self.scroll_layout.addSpacerItem(
                QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
            )
            
            # Clear database
            self.db.clear_history()
            
            # Add welcome message
            self.add_message_bubble(
                "Conversation cleared. How can I help you?", 
                False
            )
            
            self.status_signal.emit("Conversation cleared")
    
    @pyqtSlot(str, bool)
    def add_message_bubble(self, message, is_user):
        """Add message bubble to the chat area."""
        bubble = MessageBubble(message, is_user)
        
        # Insert before the last item (which is the spacer)
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, bubble)
        
        # Scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
    
    @pyqtSlot(str)
    def update_status(self, message):
        """Update status label."""
        self.status_label.setText(message)
    
    @pyqtSlot(int, str)
    def update_progress(self, value, text):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(text)
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop any active processes
        self.is_listening = False
        
        # Clean up resources
        try:
            if hasattr(self, 'stt'):
                self.stt.cleanup()
            
            if hasattr(self, 'speech_manager'):
                self.speech_manager.cleanup()
                
            # Save conversation history
            if hasattr(self, 'db'):
                self.db.export_conversation("json")
                
        except Exception as e:
            log.error(f"Error during cleanup: {str(e)}", exc_info=True)
        
        # Accept the close event
        event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = OARCSpeechChat()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()