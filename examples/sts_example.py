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
import asyncio
import ollama
import base64
import json
import pandas as pd

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
        self.available_models = ["llama3", "llama2", "mistral"]  # Default models in case setup fails
        
        self.setup_components()  # This will properly set self.available_models
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
    
    # Updated setup_components method to use the correct method from OllamaCommands
    def setup_components(self):
        """Initialize OARC components."""
        try:
            # Initialize PandasDB for conversation storage
            log.info("Initializing PandasDB...")
            self.db = PandasDB()
            
            # Fix for the agent_id issue - create a custom conversation initialization method
            self.initialize_conversation_storage()
            
            # Initialize STT component
            log.info("Initializing Speech-to-Text...")
            self.stt = SpeechToText()
            
            # Initialize TTS component with TextToSpeech
            log.info("Initializing Text-to-Speech...")
            self.tts = TextToSpeech(voice_name=self.voice_name, voice_type="xtts_v2")
            
            # Initialize Ollama interface
            log.info("Initializing Ollama Commands...")
            self.ollama = OllamaCommands()
            
            # Get available models - use the class method instead of direct API call
            try:
                log.info("Fetching available Ollama models...")
                
                # Use the ollama_list method from the OllamaCommands class
                # which needs to be run in an async context
                models_list = asyncio.run(self.ollama.ollama_list())
                
                if models_list:
                    self.available_models = models_list
                    log.info(f"Found {len(self.available_models)} Ollama models: {self.available_models}")
                else:
                    log.warning("No Ollama models found")
                    self.available_models = []
                    
            except Exception as e:
                log.warning(f"Error getting Ollama models: {str(e)}")
                self.available_models = []
                
            # Initialize MultiModalPrompting
            log.info("Initializing MultiModalPrompting...")
            self.prompt_manager = MultiModalPrompting()
            # Set necessary attributes for proper functioning
            self.prompt_manager.AGENT_FLAG = True
            self.prompt_manager.SYSTEM_SELECT_FLAG = False
            self.prompt_manager.TTS_FLAG = True  # We want to use TTS integration

            # Initialize TTS processor for MultiModalPrompting if needed
            if not hasattr(self.prompt_manager, 'tts_processor_instance'):
                self.prompt_manager.tts_processor_instance = self.tts
            
            log.info("All components initialized successfully")
            
        except Exception as e:
            log.error(f"Error initializing components: {str(e)}", exc_info=True)
            # Set default available_models if they weren't set due to error
            if not hasattr(self, 'available_models'):
                self.available_models = ["llama3", "llama2", "mistral"]
            QMessageBox.critical(self, "Initialization Error", 
                                f"Failed to initialize OARC components: {str(e)}")
    
    def initialize_conversation_storage(self):
        """Initialize conversation storage without using agent_id."""
        try:
            # Create a dataframe with necessary columns if it doesn't exist
            if not hasattr(self.db, 'df') or self.db.df is None:
                self.db.df = pd.DataFrame(columns=['timestamp', 'role', 'content', 'metadata'])
            
            # Set up conversation filenames
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.db.save_name = f"conversation_user_{current_date}"
            self.db.load_name = self.db.save_name
            
            # Create output directory if it doesn't exist
            if not hasattr(self.db, 'output_dir'):
                from oarc.utils.paths import Paths
                paths = Paths()
                self.db.output_dir = os.path.join(paths.get_output_dir(), "conversations")
                os.makedirs(self.db.output_dir, exist_ok=True)
            
            log.info(f"Conversation storage initialized with save_name: {self.db.save_name}")
        except Exception as e:
            log.error(f"Error initializing conversation storage: {str(e)}", exc_info=True)
            raise
    
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
        
        # FLUID SPEECH-TO-SPEECH BUTTON
        self.fluid_button = QPushButton("🎙️ START FLUID CONVERSATION")
        self.fluid_button.setCheckable(True)
        self.fluid_button.setStyleSheet("""
            QPushButton {
                font-size: 16pt;
                padding: 20px;
                margin-bottom: 15px;
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.fluid_button.clicked.connect(self.toggle_fluid_conversation)
        controls_layout.addWidget(self.fluid_button)
        
        # Settings section
        settings_group = QWidget()
        settings_layout = QVBoxLayout(settings_group)
        
        # Add a section title
        settings_title = QLabel("Settings")
        settings_title.setStyleSheet("font-weight: bold; margin-top: 5px;")
        settings_layout.addWidget(settings_title)
        
        # Model and voice selection
        selection_layout = QHBoxLayout()
        
        # Model selection
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Ollama Model:"))
        self.model_combo = QComboBox()
        if self.available_models:
            self.model_combo.addItems(self.available_models)
        else:
            self.model_combo.addItem("No models available")
            self.model_combo.setEnabled(False)
        model_layout.addWidget(self.model_combo)
        selection_layout.addLayout(model_layout)
        
        # Voice selection
        voice_layout = QVBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["C3PO", "Darth Vader", "Yoda"])  # Add your available voices
        self.voice_combo.currentTextChanged.connect(self.change_voice)
        voice_layout.addWidget(self.voice_combo)
        selection_layout.addLayout(voice_layout)
        
        settings_layout.addLayout(selection_layout)
        controls_layout.addWidget(settings_group)
        
        # Standard controls hidden in a collapsible section
        self.controls_group = QWidget()
        manual_layout = QVBoxLayout(self.controls_group)
        
        manual_title = QLabel("Manual Controls")
        manual_title.setStyleSheet("font-weight: bold; margin-top: 5px;")
        manual_layout.addWidget(manual_title)
        
        # Input area
        input_layout = QHBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your message here...")
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
        
        input_layout.addLayout(button_layout)
        manual_layout.addLayout(input_layout)
        
        # Hide controls by default
        self.controls_group.setVisible(False)
        controls_layout.addWidget(self.controls_group)
        
        # Utility buttons
        utility_layout = QHBoxLayout()
        
        # Toggle manual controls button
        toggle_controls_button = QPushButton("⚙️ Toggle Manual Controls")
        toggle_controls_button.clicked.connect(self.toggle_manual_controls)
        utility_layout.addWidget(toggle_controls_button)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.clicked.connect(self.clear_conversation)
        utility_layout.addWidget(self.clear_button)
        
        self.stop_button = QPushButton("🔇 Stop Speaking")
        self.stop_button.clicked.connect(self.interrupt_speech)
        self.stop_button.setEnabled(False)  # Disabled until speech is playing
        utility_layout.addWidget(self.stop_button)
        
        controls_layout.addLayout(utility_layout)
        
        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        controls_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Click the green button to start a fluid conversation")
        self.status_label.setStyleSheet("color: #555555;")
        controls_layout.addWidget(self.status_label)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(controls_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 200])
        
        # Add initial welcome message
        self.add_message_bubble(
            "Hello! I'm your OARC AI assistant. Click the green button above to start our conversation. "
            "I'll listen to you and respond automatically with voice.", 
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
        self.store_message("user", message)
        
        # Clear input field
        self.text_input.clear()
        
        # Process in background thread
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        """Process user message and generate AI response using MultiModalPrompting."""
        try:
            self.status_signal.emit("Thinking...")
            self.progress_signal.emit(30, "Processing")
            
            # Get selected model
            model = self.model_combo.currentText()
            
            # Create a minimal agent configuration for MultiModalPrompting
            agent = {
                "agent_core": {
                    "agent_id": "speech_chat",
                    "prompts": {
                        "userInput": message,
                        "llmSystem": "You are a helpful AI assistant in a voice conversation. Keep responses concise and natural."
                    },
                    "models": {
                        "largeLanguageModel": {
                            "names": [model]
                        }
                    },
                    "conversation": {
                        "load_name": "speech_chat_conversation"
                    },
                    "modalityFlags": {
                        "EMBEDDING_FLAG": False,
                        "MEMORY_CLEAR_FLAG": False,
                        "LLAVA_FLAG": False,
                        "LLM_BOOSTER_PROMPT_FLAG": False,
                        "TTS_FLAG": True
                    }
                }
            }
            
            # Create a simple handler object to store messages
            class SimpleHandler:
                async def store_message(self, message):
                    pass
            
            # Get conversation history in the format expected by MultiModalPrompting
            history = self.get_conversation_history(limit=10)
            
            # Use the powerful send_prompt method from MultiModalPrompting
            self.progress_signal.emit(50, "Getting response")
            
            # Run the async send_prompt method in the event loop
            ai_message = asyncio.run(self.prompt_manager.send_prompt(
                agent=agent,
                handler=SimpleHandler(),
                history=history
            ))
            
            if not ai_message or ai_message.startswith("Error:"):
                self.status_signal.emit(f"Error: {ai_message}")
                self.progress_signal.emit(0, "Error")
                return
            
            # Add AI message to UI
            self.message_signal.emit(ai_message, False)
            
            # Store in database
            self.store_message("assistant", ai_message)
            
            # Convert to speech
            self.status_signal.emit("Generating speech...")
            self.progress_signal.emit(75, "Generating speech")
            
            # Add to speech queue
            self.speech_queue.put(ai_message)
            
        except Exception as e:
            log.error(f"Error processing message: {str(e)}", exc_info=True)
            self.status_signal.emit(f"Error: {str(e)}")
            self.progress_signal.emit(0, "Error")
    
    # Modified speech_generation_thread to use TextToSpeech
    def speech_generation_thread(self):
        """Background thread for generating speech audio."""
        while True:
            try:
                # Get next text from queue
                text = self.speech_queue.get()
                self.is_speaking = True
                
                # Update UI state
                self.status_signal.emit("Generating speech...")
                self.progress_signal.emit(75, "Processing text")
                
                # Split text into sentences for better chunking
                sentences = self.tts.split_into_sentences(text)
                
                if len(sentences) > 1:
                    # Use the advanced chunking and parallel processing for multiple sentences
                    self.status_signal.emit(f"Processing {len(sentences)} sentences...")
                    
                    # Generate and play in streaming mode
                    # This automatically handles chunking and parallel generation/playback
                    self.tts.generate_play_audio_loop(sentences)
                else:
                    # For single sentence, we can use simpler approach
                    self.status_signal.emit("Generating speech...")
                    audio_data = self.tts.process_tts_responses(text, self.voice_name)
                    
                    # Save to temporary file and play
                    temp_dir = os.path.join(Paths().get_output_dir(), "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file = os.path.join(temp_dir, f"speech_{int(time.time())}.wav")
                    
                    # Save audio data to file
                    sf.write(temp_file, audio_data, self.tts.sample_rate)
                    
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
    
    # Update UI state in process_queues method
    def process_queues(self):
        """Process any pending queue messages and update UI state."""
        # Enable/disable stop button based on speaking state
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(self.is_speaking)
        
        # Check if fluid conversation should be reset
        if hasattr(self, 'fluid_button') and self.fluid_button.isChecked() and not self.is_listening:
            self.fluid_button.setChecked(False)
            self.fluid_button.setText("🎙️ START FLUID CONVERSATION")
    
    # Modified change_voice method to use TextToSpeech
    def change_voice(self, voice_name):
        """Change the TTS voice."""
        try:
            self.voice_name = voice_name
            # Reinitialize TextToSpeech with new voice
            self.tts = TextToSpeech(voice_name=voice_name, voice_type="xtts_v2")
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
    
    # Modified closeEvent to clean up TTS resources
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop any active processes
        self.is_listening = False
        
        # Clean up resources
        try:
            if hasattr(self, 'stt'):
                self.stt.cleanup()
            
            # Use TextToSpeech cleanup instead of SpeechManager
            if hasattr(self, 'tts'):
                self.tts.cleanup()
                
            # Save conversation history
            if hasattr(self, 'db'):
                self.db.export_conversation("json")
                
        except Exception as e:
            log.error(f"Error during cleanup: {str(e)}", exc_info=True)
        
        # Accept the close event
        event.accept()
    
    # Add interrupt_speech method
    def interrupt_speech(self):
        """Interrupt ongoing speech generation."""
        if self.is_speaking and hasattr(self, 'tts'):
            log.info("Interrupting speech generation")
            self.tts.interrupt_generation()
            self.status_signal.emit("Speech interrupted")
            self.progress_signal.emit(0, "Ready")
    
    def store_message(self, role, content):
        """Store a message in the conversation history."""
        try:
            # Add message to the dataframe
            timestamp = datetime.now().isoformat()
            new_row = {
                'timestamp': timestamp,
                'role': role, 
                'content': content,
                'metadata': {}
            }
            
            # Use pandas loc to add row
            if hasattr(self.db, 'df'):
                new_index = len(self.db.df)
                self.db.df.loc[new_index] = new_row
                log.info(f"Message stored: {role} - {content[:50]}...")
        except Exception as e:
            log.error(f"Error storing message: {str(e)}", exc_info=True)
    
    def get_conversation_history(self, limit=10):
        """Get the conversation history from the database."""
        try:
            if hasattr(self.db, 'df') and self.db.df is not None and len(self.db.df) > 0:
                # Get the most recent 'limit' messages
                recent_messages = self.db.df.tail(limit).copy()
                
                # Format them as a list of dictionaries
                history = []
                for _, row in recent_messages.iterrows():
                    history.append({
                        'role': row['role'],
                        'content': row['content']
                    })
                
                return history
            return []
        except Exception as e:
            log.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
            return []
    
    def toggle_manual_controls(self):
        """Toggle visibility of manual controls."""
        self.controls_group.setVisible(not self.controls_group.isVisible())

    def toggle_fluid_conversation(self, checked):
        """Toggle fluid speech-to-speech conversation mode."""
        if checked:
            # First verify that we have a model available
            if not self.available_models:
                QMessageBox.warning(self, "No Models Available", 
                                    "No Ollama models are available. Please install at least one model.")
                self.fluid_button.setChecked(False)
                return
                
            # Start fluid conversation
            self.running = True  # Set running flag to true
            self.fluid_button.setText("🛑 STOP CONVERSATION")
            self.status_signal.emit("I'm listening... Go ahead and speak.")
            self.progress_signal.emit(0, "Listening")
            
            # Start the fluid conversation thread
            threading.Thread(target=self.fluid_conversation_loop, daemon=True).start()
        else:
            # Stop fluid conversation
            self.running = False  # Set running flag to false
            self.fluid_button.setText("🎙️ START FLUID CONVERSATION")
            self.status_signal.emit("Conversation stopped")
            self.progress_signal.emit(0, "Ready")
            
            # Stop any active processing
            self.is_listening = False

    def fluid_conversation_loop(self):
        """Background thread that runs the continuous conversation flow."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            log.info("Starting fluid conversation loop")
            self.status_signal.emit("Listening...")
            
            while self.running:
                try:
                    # Instead of waiting for wake word, just listen for voice
                    self.status_signal.emit("Listening for speech...")
                    self.progress_signal.emit(10, "Listening")
                    
                    # Use the regular listen method instead of listen_for_wake_word
                    temp_file = self.stt.listen(threshold=600, silence_duration=0.5)
                    
                    if not self.running:
                        break  # Exit if we've been told to stop
                    
                    if not temp_file:
                        continue  # No speech detected, continue listening
                    
                    # Process the recorded speech
                    self.status_signal.emit("Processing speech...")
                    self.progress_signal.emit(30, "Processing")
                    
                    # Get the transcription
                    transcribed_text = self.stt.recognize_speech(temp_file)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        log.warning(f"Failed to remove temporary file: {e}")
                    
                    if not transcribed_text or len(transcribed_text.strip()) == 0:
                        self.status_signal.emit("Couldn't hear anything clearly. Please try again.")
                        self.progress_signal.emit(0, "Idle")
                        continue
                    
                    # Add user message to UI
                    self.message_signal.emit(transcribed_text, True)
                    
                    # Store in database
                    self.store_message("user", transcribed_text)
                    
                    # Get selected model
                    model = self.model_combo.currentText()
                    
                    # Get response from Ollama using the powerful MultiModalPrompting
                    try:
                        self.status_signal.emit(f"Getting response from {model}...")
                        
                        # Create a minimal agent configuration
                        agent = {
                            "agent_core": {
                                "agent_id": "speech_chat",
                                "prompts": {
                                    "userInput": transcribed_text,
                                    "llmSystem": "You are a helpful AI assistant in a voice conversation. Keep responses concise and natural."
                                },
                                "models": {
                                    "largeLanguageModel": {
                                        "names": [model]
                                    }
                                },
                                "conversation": {
                                    "load_name": "speech_chat_conversation"
                                },
                                "modalityFlags": {
                                    "EMBEDDING_FLAG": False,
                                    "MEMORY_CLEAR_FLAG": False,
                                    "LLAVA_FLAG": False,
                                    "LLM_BOOSTER_PROMPT_FLAG": False,
                                    "TTS_FLAG": True
                                }
                            }
                        }
                        
                        # Create a simple handler object
                        class SimpleHandler:
                            async def store_message(self, message):
                                pass
                        
                        # Get conversation history
                        history = self.get_conversation_history(limit=10)
                        
                        # Use the event loop we created for this thread
                        ai_message = loop.run_until_complete(self.prompt_manager.send_prompt(
                            agent=agent,
                            handler=SimpleHandler(),
                            history=history
                        ))
                        
                        if not ai_message or (isinstance(ai_message, str) and ai_message.startswith("Error:")):
                            self.status_signal.emit(f"Error: {ai_message}")
                            self.progress_signal.emit(0, "Error")
                            continue
                    
                        # Add AI message to UI
                        self.message_signal.emit(ai_message, False)
                        
                        # Store in database
                        self.store_message("assistant", ai_message)
                        
                        # Convert to speech
                        self.status_signal.emit("Generating speech...")
                        self.progress_signal.emit(75, "Generating speech")
                        
                        # Add to speech queue
                        self.speech_queue.put(ai_message)
                        
                    except Exception as e:
                        log.error(f"Error getting response: {str(e)}", exc_info=True)
                        self.status_signal.emit(f"Error: {str(e)}")
                        self.progress_signal.emit(0, "Error")
                    
                    self.status_signal.emit("Listening...")
                    self.progress_signal.emit(0, "Idle")
                    
                except Exception as e:
                    if self.running:  # Only log if we're still supposed to be running
                        log.error(f"Error in fluid conversation: {str(e)}", exc_info=True)
                        self.status_signal.emit(f"Error: {str(e)}")
                        self.progress_signal.emit(0, "Error")
                        time.sleep(2)  # Pause briefly before continuing
                
        except Exception as e:
            log.error(f"Fatal error in fluid conversation thread: {str(e)}", exc_info=True)
        
        finally:
            # Clean up the event loop when the thread exits
            try:
                loop.close()
            except:
                pass


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