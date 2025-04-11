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

Author: @Borch & @P3nGu1nZz
Date: 4/8/2025
"""

import os
import sys
import time
import threading
import queue
from datetime import datetime
import asyncio
import pandas as pd
import numpy as np
import sounddevice as sd

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox, QProgressBar,
    QMessageBox, QSplitter, QFrame, QScrollArea, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot

# OARC imports
from oarc.speech import SpeechToText, TextToSpeech
from oarc.ollama.utils.ollama_commands import OllamaCommands
from oarc.database.pandas_db import PandasDB
from oarc.prompt import MultiModalPrompting
from oarc.utils.log import log

# External library
import ollama

class MessageBubble(QFrame):
    """Widget to display a single message in chat-like bubble style with neon highlights."""
    
    def __init__(self, message, is_user=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Set styles based on message sender
        if is_user:
            self.setStyleSheet("""
                background-color: #2a2a2a;
                border-left: 4px solid #00e5ff;
                border-radius: 5px;
                padding: 5px;
                margin: 5px 20px 5px 50px;
            """)
            alignment = Qt.AlignmentFlag.AlignRight
            icon = "🧠"  # User icon
        else:
            self.setStyleSheet("""
                background-color: #252525;
                border-left: 4px solid #ff1744;
                border-radius: 5px;
                padding: 5px;
                margin: 5px 50px 5px 20px;
            """)
            alignment = Qt.AlignmentFlag.AlignLeft
            icon = "🤖"  # AI icon
        
        # Layout and label for message content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Header with icon and timestamp
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        # Icon label
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("background-color: transparent; font-size: 14px;")
        header_layout.addWidget(icon_label)
        
        # Add entity label
        entity_label = QLabel("YOU" if is_user else "OARC AI")
        entity_label.setStyleSheet(f"""
            background-color: transparent;
            color: {'#00e5ff' if is_user else '#ff1744'};
            font-weight: bold;
            font-size: 10px;
            letter-spacing: 1px;
        """)
        header_layout.addWidget(entity_label)
        
        header_layout.addStretch()
        
        # Add timestamp 
        timestamp = QLabel(datetime.now().strftime("%H:%M:%S"))
        timestamp.setStyleSheet("color: #666666; background-color: transparent; font-size: 9px;")
        timestamp.setAlignment(alignment)
        header_layout.addWidget(timestamp)
        
        layout.addLayout(header_layout)
        
        # Message text with improved styling
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"""
            background-color: transparent;
            color: #e0e0e0;
            font-size: 13px;
            letter-spacing: 0.2px;
            line-height: 1.4;
        """)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        layout.addWidget(message_label)


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
        self.running = False  # Initialize running attribute for fluid conversation
        self.ollama_model = "llama3"  # Default model only used if setup fails
        self.voice_name = "C3PO"  # Default voice
        self.language = "en"  # Default language
        self.available_models = []  # Initialize empty, will be filled by setup_components
        
        # Animation variables
        self.pulse_opacity = 1.0
        self.pulse_direction = -0.1
        self.indicator_active = False
        
        # Setup the components first to get available models
        self.setup_components()
        # Then setup UI which needs the model list
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
            
            log.info(f"Voice reference path: {self.tts.voice_name_reference_speech_path}")
            if not os.path.exists(self.tts.voice_name_reference_speech_path):
                log.error(f"Voice reference file not found: {self.tts.voice_name_reference_speech_path}")
                log.error("This will prevent TTS from working properly")
            
            # Initialize Ollama interface
            log.info("Initializing Ollama Commands...")
            self.ollama = OllamaCommands()
            
            # Get available models - use the class method instead of direct API call
            try:
                log.info("Fetching available Ollama models...")
                
                # Create a new event loop for async operation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Use the ollama_list method from the OllamaCommands class
                models_list = loop.run_until_complete(self.ollama.ollama_list())
                loop.close()
                
                if models_list and len(models_list) > 0:
                    self.available_models = models_list
                    # Set the first model as default if available
                    self.ollama_model = models_list[0]
                    log.info(f"Found {len(self.available_models)} Ollama models: {self.available_models}")
                else:
                    log.warning("No Ollama models found, using defaults")
                    self.available_models = ["llama3", "llama2", "mistral"]  # Fallback defaults
            except Exception as e:
                log.warning(f"Error getting Ollama models: {str(e)}")
                # Use fallback defaults if model fetching fails
                self.available_models = ["llama3", "llama2", "mistral"]
                
            # Initialize MultiModalPrompting
            log.info("Initializing MultiModalPrompting...")
            self.prompt_manager = MultiModalPrompting()
            
            # Set necessary attributes for proper functioning
            self.prompt_manager.AGENT_FLAG = True
            self.prompt_manager.SYSTEM_SELECT_FLAG = False
            self.prompt_manager.TTS_FLAG = True  # Enable TTS integration
            self.prompt_manager.LLAVA_FLAG = False  # Ensure LLAVA flag is set
            self.prompt_manager.EMBEDDING_FLAG = False
            self.prompt_manager.MEMORY_CLEAR_FLAG = False
            self.prompt_manager.LLM_BOOSTER_PROMPT_FLAG = False

            # Initialize TTS processor for MultiModalPrompting if needed
            if not hasattr(self.prompt_manager, 'tts_processor_instance'):
                self.prompt_manager.tts_processor_instance = self.tts
            
            log.info("All components initialized successfully")
            
            # Add below your existing component initialization
            # Test sound playback to ensure audio device is working
            log.info("Testing sound output...")
            
            # Generate a short beep sound
            sample_rate = 44100
            duration = 0.5  # seconds
            frequency = 440  # Hz (A4 note)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = 0.2 * np.sin(2 * np.pi * frequency * t)  # Sine wave
            
            # Play the test sound
            sd.play(beep, sample_rate)
            sd.wait()
            
            log.info("Sound test complete")
        except Exception as e:
            log.error(f"Sound test failed: {str(e)}. Please check your audio configuration.")
            
        except Exception as e:
            log.error(f"Error initializing components: {str(e)}", exc_info=True)
            # Ensure we have at least some default models if they weren't set due to error
            if not hasattr(self, 'available_models') or not self.available_models:
                self.available_models = ["llama3", "llama2", "mistral"]
            QMessageBox.critical(self, "Initialization Error", 
                                f"Failed to initialize OARC components: {str(e)}")
    
        # Initialize TTS processor
        log.info("Initializing TTS processor...")
        self.tts = TextToSpeech(voice_name=self.voice_name, voice_type="xtts_v2")
        
        # Initialize SpeechManager singleton
        log.info("Initializing SpeechManager...")
        from oarc.speech.speech_manager import SpeechManager
        self.speech_manager = SpeechManager(voice_name=self.voice_name)
    
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
        """Set up the modernized dark mode neon UI."""
        self.setWindowTitle("OARC Speech-to-Speech Chat")
        self.setMinimumSize(900, 700)
        
        # Set global stylesheet for the entire application
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #e0e0e0;
            }
            QScrollArea {
                background-color: #1a1a1a;
                border: none;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #1a1a1a;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #00b8d4;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QComboBox {
                background-color: #2a2a2a;
                border: 1px solid #00b8d4;
                border-radius: 5px;
                padding: 5px;
                color: #e0e0e0;
                selection-background-color: #00b8d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                selection-background-color: #00b8d4;
                selection-color: #121212;
                border: 1px solid #00b8d4;
            }
            QLabel {
                color: #e0e0e0;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #00b8d4;
                border-radius: 5px;
                padding: 8px;
                color: #e0e0e0;
                selection-background-color: #00b8d4;
                selection-color: #121212;
            }
            QPushButton {
                background-color: #2a2a2a;
                color: #00b8d4;
                border: 1px solid #00b8d4;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #00e5ff;
                color: #00e5ff;
            }
            QPushButton:pressed {
                background-color: #00b8d4;
                color: #121212;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #5a5a5a;
                border-color: #5a5a5a;
            }
            QProgressBar {
                border: 1px solid #00b8d4;
                border-radius: 5px;
                background-color: #1a1a1a;
                text-align: center;
                color: #e0e0e0;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b8d4, stop:1 #00e5ff);
                border-radius: 4px;
            }
            QSplitter::handle {
                background-color: #00b8d4;
                height: 2px;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Animated title section with neon effect
        title_widget = QWidget()
        title_widget.setFixedHeight(80)
        title_widget.setStyleSheet("""
            background-color: #1a1a1a;
            border-radius: 10px;
            border: 1px solid #00b8d4;
        """)
        title_layout = QVBoxLayout(title_widget)
        
        title_label = QLabel("OARC NEURAL SPEECH INTERFACE")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #00e5ff;
            text-shadow: 0 0 10px #00b8d4, 0 0 20px #00b8d4, 0 0 30px #00b8d4;
            font-family: 'Segoe UI', Arial, sans-serif;
            letter-spacing: 2px;
        """)
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("AI-POWERED VOICE COMMUNICATION SYSTEM")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("""
            font-size: 12px;
            color: #b0b0b0;
            letter-spacing: 1px;
        """)
        title_layout.addWidget(subtitle_label)
        
        main_layout.addWidget(title_widget)
        
        # Create a pulsing animation for the title
        self.pulse_effect = QTimer(self)
        self.pulse_effect.timeout.connect(self.pulse_title)
        self.pulse_effect.start(2000)  # Pulse every 2 seconds
        self.pulse_opacity = 1.0
        self.pulse_direction = -0.1
        
        # Create splitter for chat and controls
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("QSplitter::handle { height: 2px; }")
        main_layout.addWidget(splitter, 1)
        
        # Chat area with scroll and neon border
        chat_widget = QWidget()
        chat_widget.setStyleSheet("""
            background-color: #1a1a1a;
            border-radius: 10px;
            border: 1px solid #00b8d4;
        """)
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add chat header
        chat_header = QWidget()
        chat_header.setFixedHeight(40)
        chat_header.setStyleSheet("background-color: transparent;")
        header_layout = QHBoxLayout(chat_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        chat_icon = QLabel("💬")
        chat_icon.setStyleSheet("font-size: 20px; color: #00e5ff;")
        header_layout.addWidget(chat_icon)
        
        chat_title = QLabel("CONVERSATION")
        chat_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #00e5ff;
            letter-spacing: 1px;
        """)
        header_layout.addWidget(chat_title)
        
        header_layout.addStretch()
        
        # Add animated indicator for active listening
        self.listening_indicator = QLabel("●")
        self.listening_indicator.setStyleSheet("""
            font-size: 16px;
            color: #333333;
        """)
        header_layout.addWidget(self.listening_indicator)
        
        # Create listening indicator animation
        self.indicator_timer = QTimer(self)
        self.indicator_timer.timeout.connect(self.animate_indicator)
        self.indicator_timer.start(500)  # Toggle every 500ms
        self.indicator_active = False
        
        chat_layout.addWidget(chat_header)
        
        # Line separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setStyleSheet("background-color: #00b8d4; margin: 0px 0px 10px 0px;")
        separator.setFixedHeight(1)
        chat_layout.addWidget(separator)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: transparent;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_layout.setSpacing(10)
        
        # Add spacer at the bottom to push content up
        self.scroll_layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        
        self.scroll_area.setWidget(self.scroll_content)
        chat_layout.addWidget(self.scroll_area)
        
        # Controls panel with neon glow
        controls_widget = QWidget()
        controls_widget.setStyleSheet("""
            background-color: #1a1a1a;
            border-radius: 10px;
            border: 1px solid #00b8d4;
        """)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(15, 15, 15, 15)
        controls_layout.setSpacing(15)
        
        # FLUID SPEECH-TO-SPEECH BUTTON with neon glow effect
        self.fluid_button = QPushButton("🎙️ START NEURAL VOICE INTERFACE")
        self.fluid_button.setCheckable(True)
        self.fluid_button.setStyleSheet("""
            QPushButton {
                font-size: 18pt;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #0d333d;
                color: #00e5ff;
                border: 2px solid #00b8d4;
                border-radius: 10px;
                font-weight: bold;
                text-shadow: 0 0 10px #00b8d4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #164450;
                border-color: #00e5ff;
                text-shadow: 0 0 15px #00e5ff;
            }
            QPushButton:checked {
                background-color: #701c23;
                border-color: #ff1744;
                color: #ff1744;
                text-shadow: 0 0 10px #ff1744;
            }
        """)
        self.fluid_button.clicked.connect(self.toggle_fluid_conversation)
        controls_layout.addWidget(self.fluid_button)
        
        # Settings section with dark theme
        settings_group = QWidget()
        settings_group.setStyleSheet("""
            background-color: #252525;
            border-radius: 8px;
        """)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(10)
        
        # Add a section title
        settings_title = QLabel("CONFIGURATION")
        settings_title.setStyleSheet("""
            font-weight: bold;
            color: #00e5ff;
            font-size: 12px;
            letter-spacing: 1px;
            padding-bottom: 5px;
            border-bottom: 1px solid #00b8d4;
        """)
        settings_layout.addWidget(settings_title)
        
        # Model and voice selection
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(15)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_layout.setSpacing(5)
        model_label = QLabel("AI MODEL:")
        model_label.setStyleSheet("color: #b0b0b0; font-size: 10px; letter-spacing: 1px;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(35)
        if self.available_models and len(self.available_models) > 0:
            self.model_combo.addItems(self.available_models)
            # Select the default model if it's in the available models
            default_index = self.model_combo.findText(self.ollama_model)
            if default_index >= 0:
                self.model_combo.setCurrentIndex(default_index)
        else:
            self.model_combo.addItem("No models available")
            self.model_combo.setEnabled(False)
        model_layout.addWidget(self.model_combo)
        selection_layout.addLayout(model_layout)
        
        # Voice selection
        voice_layout = QVBoxLayout()
        voice_layout.setSpacing(5)
        voice_label = QLabel("VOICE SYNTHESIS:")
        voice_label.setStyleSheet("color: #b0b0b0; font-size: 10px; letter-spacing: 1px;")
        voice_layout.addWidget(voice_label)
        
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumHeight(35)
        self.voice_combo.addItems(["C3PO", "Darth Vader", "Yoda"])  # Add your available voices
        self.voice_combo.currentTextChanged.connect(self.change_voice)
        voice_layout.addWidget(self.voice_combo)
        selection_layout.addLayout(voice_layout)
        
        # Add language selection
        language_layout = QVBoxLayout()
        language_layout.setSpacing(5)
        language_label = QLabel("LANGUAGE:")
        language_label.setStyleSheet("color: #b0b0b0; font-size: 10px; letter-spacing: 1px;")
        language_layout.addWidget(language_label)
        
        self.language_combo = QComboBox()
        self.language_combo.setMinimumHeight(35)
        # Add language names for better UX
        language_options = [
            ("en", "English"),
            ("fr", "French"),
            ("de", "German"),
            ("es", "Spanish"),
            ("it", "Italian"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("pt", "Portuguese"),
            ("ru", "Russian"),
            ("zh", "Chinese"),
            ("nl", "Dutch"),
            ("tr", "Turkish"),
            ("pl", "Polish"),
            ("ar", "Arabic")
        ]
        for code, name in language_options:
            self.language_combo.addItem(f"{name} ({code})", code)
        self.language_combo.currentIndexChanged.connect(self.change_language_from_combo)
        language_layout.addWidget(self.language_combo)
        selection_layout.addLayout(language_layout)
        
        settings_layout.addLayout(selection_layout)
        
        controls_layout.addWidget(settings_group)
        
        # Standard controls hidden in a collapsible section
        self.controls_group = QWidget()
        self.controls_group.setStyleSheet("""
            background-color: #252525;
            border-radius: 8px;
        """)
        manual_layout = QVBoxLayout(self.controls_group)
        manual_layout.setContentsMargins(10, 10, 10, 10)
        manual_layout.setSpacing(10)
        
        manual_title = QLabel("MANUAL CONTROL SYSTEM")
        manual_title.setStyleSheet("""
            font-weight: bold;
            color: #00e5ff;
            font-size: 12px;
            letter-spacing: 1px;
            padding-bottom: 5px;
            border-bottom: 1px solid #00b8d4;
        """)
        manual_layout.addWidget(manual_title)
        
        # Input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your message here...")
        self.text_input.setMaximumHeight(80)
        input_layout.addWidget(self.text_input, 1)
        
        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)
        
        self.listen_button = QPushButton("🎤 LISTEN")
        self.listen_button.setCheckable(True)
        self.listen_button.clicked.connect(self.toggle_listening)
        button_layout.addWidget(self.listen_button)
        
        self.send_button = QPushButton("📤 TRANSMIT")
        self.send_button.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_button)
        
        input_layout.addLayout(button_layout)
        manual_layout.addLayout(input_layout)
        
        # Hide controls by default
        self.controls_group.setVisible(False)
        controls_layout.addWidget(self.controls_group)
        
        # Utility buttons with neon style
        utility_layout = QHBoxLayout()
        utility_layout.setSpacing(10)
        
        # Toggle manual controls button
        toggle_controls_button = QPushButton("⚙️ MANUAL CONTROLS")
        toggle_controls_button.clicked.connect(self.toggle_manual_controls)
        utility_layout.addWidget(toggle_controls_button)
        
        self.clear_button = QPushButton("🗑️ RESET CONVERSATION")
        self.clear_button.clicked.connect(self.clear_conversation)
        utility_layout.addWidget(self.clear_button)
        
        self.stop_button = QPushButton("🔇 STOP AUDIO")
        self.stop_button.clicked.connect(self.interrupt_speech)
        self.stop_button.setEnabled(False)  # Disabled until speech is playing
        utility_layout.addWidget(self.stop_button)
        
        controls_layout.addLayout(utility_layout)
        
        # Progress and status with neon accent
        status_container = QWidget()
        status_container.setStyleSheet("""
            background-color: #252525;
            border-radius: 8px;
        """)
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(10)
        
        status_header = QLabel("SYSTEM STATUS")
        status_header.setStyleSheet("""
            font-weight: bold;
            color: #00e5ff;
            font-size: 10px;
            letter-spacing: 1px;
        """)
        status_layout.addWidget(status_header)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("IDLE")
        self.progress_bar.setFixedHeight(15)
        status_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("SYSTEM READY - CLICK THE NEURAL VOICE INTERFACE TO BEGIN")
        self.status_label.setStyleSheet("""
            font-size: 11px;
            padding: 5px;
            background-color: #1a1a1a;
            border-radius: 4px;
        """)
        status_layout.addWidget(self.status_label)
        
        controls_layout.addWidget(status_container)
        
        # Add widgets to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(controls_widget)
        
        # Set splitter proportions
        splitter.setSizes([500, 300])
        
        # Add initial welcome message with neon style
        self.add_message_bubble(
            "OARC Neural Interface initialized. Click the green button above to activate voice communication. "
            "System is ready for bilateral voice interaction.", 
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
            audio_file = self.stt.listen(threshold=100, silence_duration=0.5)
            if not self.is_listening or not audio_file:
                return  # User stopped listening or no audio detected
            
            self.status_signal.emit("Processing speech...")
            self.progress_signal.emit(50, "Transcribing")
            
            # Use the async recognizer method with our event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            transcribed_text = loop.run_until_complete(self.stt.recognize_speech(audio_file))
            loop.close()
            
            if transcribed_text and len(transcribed_text.strip()) > 0:
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
                        "TTS_FLAG": True  # Enable TTS integration
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
                queue_item = self.speech_queue.get()
                
                # Handle both string and tuple formats
                if isinstance(queue_item, tuple):
                    text, language = queue_item
                else:
                    text = queue_item
                    language = "en"  # Default language
                    
                self.is_speaking = True
                
                # Update UI state
                self.status_signal.emit("Generating speech...")
                self.progress_signal.emit(75, "Processing text")
                
                try:
                    # Split text into sentences
                    sentences = self.tts.split_into_sentences(text)
                    log.info(f"Text split into {len(sentences)} sentences")
                    
                    # Modify to handle sentence batching for more natural pauses and flow
                    sentence_batches = []
                    current_batch = ""
                    
                    # Group sentences into logical batches (around 100-150 chars)
                    for sentence in sentences:
                        if len(current_batch) + len(sentence) < 150:
                            current_batch += " " + sentence if current_batch else sentence
                        else:
                            if current_batch:
                                sentence_batches.append(current_batch)
                            current_batch = sentence
                    
                    # Add final batch if it exists
                    if current_batch:
                        sentence_batches.append(current_batch)
                    
                    log.info(f"Created {len(sentence_batches)} sentence batches for audio generation")
                    
                    # Check that speech_manager exists before using it
                    if not hasattr(self, 'speech_manager'):
                        log.error("Speech manager not initialized! Initializing now...")
                        from oarc.speech.speech_manager import SpeechManager
                        self.speech_manager = SpeechManager(voice_name=self.voice_name)
                    
                    # Process each sentence batch
                    for i, batch in enumerate(sentence_batches):
                        if not self.is_speaking:  # Check if interrupted
                            break
                            
                        self.status_signal.emit(f"Playing speech part {i+1}/{len(sentence_batches)}...")
                        self.progress_signal.emit(80 + (i * 20 // len(sentence_batches)), f"Playing part {i+1}")
                        
                        # Generate audio for this batch
                        audio_data = self.speech_manager.generate_speech(batch, language=language)
                        
                        if audio_data is not None and len(audio_data) > 0:
                            # Play the audio
                            sd.play(audio_data, self.speech_manager.sample_rate)
                            sd.wait()  # Wait for audio to finish
                        else:
                            log.warning(f"Failed to generate audio for batch {i+1}")
                    
                    # Update UI when done
                    self.progress_signal.emit(100, "Speech complete")
                    self.status_signal.emit("Ready to listen")
                    
                except Exception as e:
                    log.error(f"Error processing speech: {str(e)}", exc_info=True)
                    self.status_signal.emit(f"Speech error: {str(e)}")
                
            except Exception as e:
                log.error(f"Error in speech generation: {str(e)}", exc_info=True)
                
            finally:
                self.speech_queue.task_done()
                self.is_speaking = False
                self.progress_signal.emit(0, "Ready")
    
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
    
    def change_language(self, language):
        """Change the language for TTS and STT."""
        log.info(f"Changing language to: {language}")
        self.language = language
        self.status_signal.emit(f"Language changed to {language}")
    
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
        # If we're already running and trying to start again, prevent it
        if self.running and checked:
            return
            
        if checked:
            # First verify that we have a model available
            if not self.available_models:
                QMessageBox.warning(self, "No Models Available", 
                                    "No Ollama models are available. Please install at least one model.")
                self.fluid_button.setChecked(False)
                return
                    
            # Stop any existing process first
            self.running = False
            # Wait a moment for any existing threads to recognize they should stop
            time.sleep(0.5)
                
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
            self.fluid_button.setText("🎙️ START NEURAL VOICE INTERFACE")
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
                    # Use the regular listen method with MORE SENSITIVE threshold
                    self.status_signal.emit("Listening for speech...")
                    self.progress_signal.emit(10, "Listening")
                    
                    # Lower threshold to make it more sensitive
                    temp_file = self.stt.listen(threshold=350, silence_duration=1.0)
                    
                    if not self.running:
                        break  # Exit if we've been told to stop
                    
                    if not temp_file:
                        log.info("No speech detected above threshold, continuing to listen...")
                        self.status_signal.emit("No speech detected. Listening again...")
                        continue  # No speech detected, continue listening
                    
                    # Process the recorded speech
                    self.status_signal.emit("Processing speech...")
                    self.progress_signal.emit(30, "Processing")
                    
                    # Use Google STT directly with the selected language
                    try:
                        import speech_recognition as sr
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(temp_file) as source:
                            audio_data = recognizer.record(source)
                        # Use the language from the UI dropdown
                        transcribed_text = recognizer.recognize_google(audio_data, language=self.language)
                        log.info(f"Google STT result: '{transcribed_text}'")
                    except Exception as e:
                        log.error(f"Google STT failed: {e}")
                        # Fallback to async recognizer if Google fails
                        transcribed_text = loop.run_until_complete(self.stt.recognize_speech(temp_file))
                    
                    # Clean up temp file with retries
                    cleanup_success = False
                    for retry in range(3):
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                cleanup_success = True
                                break
                        except Exception as e:
                            log.warning(f"Failed to remove temp file (attempt {retry+1}/3): {e}")
                            time.sleep(0.5)  # Wait a bit before retrying
                    
                    if not cleanup_success:
                        log.warning(f"Could not remove temp file: {temp_file}")
                    
                    # Skip "Google Speech Recognition could not understand audio" messages
                    if transcribed_text and "could not understand audio" in transcribed_text.lower():
                        self.status_signal.emit("Couldn't hear clearly. Please try again.")
                        self.progress_signal.emit(0, "Idle")
                        continue
                    
                    if not transcribed_text or len(transcribed_text.strip()) == 0:
                        self.status_signal.emit("Couldn't hear anything clearly. Please try again.")
                        self.progress_signal.emit(0, "Idle")
                        continue
                    
                    # Log what was heard for debugging
                    log.info(f"Heard: '{transcribed_text}'")
                    
                    # Add user message to UI
                    self.message_signal.emit(transcribed_text, True)
                    
                    # Store in database
                    self.store_message("user", transcribed_text)
                    
                    # Get selected model
                    model = self.model_combo.currentText()
                    
                    # Get selected language from UI if available
                    language = getattr(self, 'language', 'en')  # Default to 'en' if not set

                    # Update agent with language parameter
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
                            "language": language,  # Add language to the agent config
                            "modalityFlags": {
                                "EMBEDDING_FLAG": False,
                                "MEMORY_CLEAR_FLAG": False,
                                "LLAVA_FLAG": False,
                                "LLM_BOOSTER_PROMPT_FLAG": False,
                                "TTS_FLAG": True,
                                "STREAMING_FLAG": True
                            }
                        }
                    }
                    
                    # Get conversation history
                    history = self.get_conversation_history(limit=10)
                    
                    # Create a streaming handler class
                    class StreamingHandler:
                        def __init__(self, parent):
                            self.parent = parent
                            self.full_response = ""
                        
                        async def store_message(self, message):
                            # This method is expected by MultiModalPrompting but we don't need it
                            pass
                        
                        async def handle_stream(self, chunk):
                            # This is the method that will be called with streaming chunks
                            # We'll update the UI with each chunk
                            if chunk:
                                self.full_response += chunk
                                self.parent.message_signal.emit(self.full_response, False)
                    
                    # Try to use MultiModalPrompting with an event handler
                    try:
                        self.status_signal.emit(f"Getting response from {model}...")
                        
                        # Create the handler instance
                        handler = StreamingHandler(self)
                        
                        # Use the send_prompt method (this call is simplified, not using 'stream' param)
                        ai_message = loop.run_until_complete(self.prompt_manager.send_prompt(
                            agent=agent,
                            handler=handler,
                            history=history
                        ))
                        
                        # If we have a handler with accumulated response, use that
                        if hasattr(handler, 'full_response') and handler.full_response:
                            ai_message = handler.full_response
                        
                        # Store in database
                        self.store_message("assistant", ai_message)
                        
                        # Convert to speech
                        self.status_signal.emit("Generating speech...")
                        self.progress_signal.emit(75, "Generating speech")
                        
                        # Add to speech queue
                        self.speech_queue.put((ai_message, self.language))
                        
                    except Exception as e:
                        log.error(f"Streaming with MultiModalPrompting failed: {e}")
                        
                        # Direct Ollama API fallback
                        try:
                            self.status_signal.emit(f"Fallback: Direct API call to {model}...")
                            
                            # Set up messages for direct Ollama API call
                            messages = [{"role": "system", "content": "You are a helpful AI assistant in a voice conversation. Keep responses concise and natural."}]
                            for msg in history:
                                messages.append({"role": msg["role"], "content": msg["content"]})
                            messages.append({"role": "user", "content": transcribed_text})
                            
                            # Make the direct API call
                            response = loop.run_until_complete(ollama.chat(
                                model=model,
                                messages=messages
                            ))
                            
                            # Extract content from response
                            ai_message = response["message"]["content"]
                            
                            # Add AI message to UI
                            self.message_signal.emit(ai_message, False)
                            
                            # Store in database
                            self.store_message("assistant", ai_message)
                            
                            # Add to speech queue
                            self.speech_queue.put((ai_message, self.language))
                            
                        except Exception as e2:
                            log.error(f"Fallback to direct API also failed: {e2}")
                            self.status_signal.emit(f"Error: {str(e2)}")
                            self.progress_signal.emit(0, "Error")
                    
                    self.status_signal.emit("Listening...")
                    self.progress_signal.emit(0, "Idle")
                    time.sleep(0.5)  # Brief pause before listening again
                    
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

    # Add these new animation methods to your class
    def pulse_title(self):
        """Create a pulsing effect for the title."""
        title_widget = self.findChild(QWidget, "title_widget")
        if title_widget:
            title_label = title_widget.findChildren(QLabel)[0]  # The title label
            
            # Update opacity based on direction
            self.pulse_opacity += self.pulse_direction
            
            # Change direction when reaching boundaries
            if self.pulse_opacity <= 0.7 or self.pulse_opacity >= 1.0:
                self.pulse_direction *= -1
            
            # Instead of using text-shadow, use color and font options
            title_label.setStyleSheet(f"""
                font-size: 28px;
                font-weight: bold;
                color: #00e5ff;
                font-family: 'Segoe UI', Arial, sans-serif;
                letter-spacing: 2px;
            """)

    def animate_indicator(self):
        """Animate the listening indicator."""
        self.indicator_active = not self.indicator_active
        if self.running:  # Only show active animation when in conversation mode
            if self.indicator_active:
                self.listening_indicator.setStyleSheet("font-size: 16px; color: #ff1744;")
            else:
                self.listening_indicator.setStyleSheet("font-size: 16px; color: #701c23;")
        else:
            if self.indicator_active:
                self.listening_indicator.setStyleSheet("font-size: 16px; color: #00e5ff;")
            else:
                self.listening_indicator.setStyleSheet("font-size: 16px; color: #0d333d;")

    # Add method to extract language code from combo box
    def change_language_from_combo(self, index):
        # Get the actual language code from the item data
        language_code = self.language_combo.itemData(index)
        self.change_language(language_code)


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