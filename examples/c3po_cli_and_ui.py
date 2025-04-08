#!/usr/bin/env python3
"""
OARC C3PO Text-to-Speech Demo

This script demonstrates how to use the OARC package to generate speech 
using the C3PO voice from Star Wars. The application provides both:
- A command-line interface for quick speech generation
- A graphical user interface for interactive use

Usage:
    # CLI mode:
    python c3po_tts_demo.py text "Hello, I am C-3PO" --output hello.wav
    python c3po_tts_demo.py file script.txt --output speech.wav
    
    # GUI mode:
    python c3po_tts_demo.py ui

Requirements:
    - OARC package installed (pip install oarc)
    - PyQt6 library for GUI mode
    - Internet connection (for first-time model download)
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import logging

# PyQt6 for GUI
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QSlider, QComboBox,
    QProgressBar, QMessageBox, QLineEdit, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot

# Import OARC components
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.speech.speech_manager import SpeechManager
from oarc.speech.speech_utils import SpeechUtils


class C3POGenerator:
    """Core C3PO voice generation functionality."""
    
    def __init__(self):
        self.speech_manager = None
        self.voice_name = "C3PO"
        self.voice_type = "xtts_v2"
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the speech manager with C3PO voice."""
        try:
            log.info(f"Ensuring {self.voice_name} voice reference exists...")
            SpeechUtils.ensure_voice_reference_exists(self.voice_name)
            
            log.info("Initializing SpeechManager with C3PO voice...")
            self.speech_manager = SpeechManager(voice_name=self.voice_name, voice_type=self.voice_type)
            self.is_initialized = True
            return True
        except Exception as e:
            log.error(f"Error initializing C3PO generator: {str(e)}", exc_info=True)
            return False
    
    def generate_speech(self, text: str, output_file: str, speed: float = 1.0, language: str = "en") -> bool:
        """Generate speech with C3PO voice.
        
        Args:
            text: The text to convert to speech
            output_file: Path to save the output WAV file
            speed: Speech speed multiplier (0.5-2.0)
            language: Language code (default: "en")
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(os.path.abspath(output_file))
            os.makedirs(output_dir, exist_ok=True)
            
            log.info(f"Generating speech for: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Generate speech to file
            self.speech_manager.generate_speech_to_file(
                text=text,
                output_file=output_file,
                speed=speed,
                language=language,
                overwrite=True
            )
            
            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                log.info(f"Speech generated successfully! File saved to: {output_file} ({file_size/1024:.1f} KB)")
                return True
            else:
                log.error(f"Speech generation appeared to succeed but file {output_file} does not exist")
                return False
        except Exception as e:
            log.error(f"Error generating speech: {str(e)}", exc_info=True)
            return False
    
    def process_text_file(self, file_path: str, output_file: str, max_chunk_length: int = 1000, 
                           speed: float = 1.0, language: str = "en") -> bool:
        """Process a text file and convert it to speech.
        
        Args:
            file_path: Path to the text file to process
            output_file: Path to save the output WAV file
            max_chunk_length: Maximum length of each text chunk for processing
            speed: Speech speed multiplier
            language: Language code
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the text file
            log.info(f"Reading text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                log.error(f"Text file is empty: {file_path}")
                return False
            
            # Get file size for logging
            file_size = os.path.getsize(file_path)
            log.info(f"Text file size: {file_size/1024:.1f} KB, {len(content)} characters")
            
            # Process the file (for now we're just handling it as one chunk)
            # In a more advanced implementation, we could split large texts
            return self.generate_speech(content, output_file, speed, language)
            
        except Exception as e:
            log.error(f"Error processing text file: {str(e)}", exc_info=True)
            return False


class C3POTTSUI(QMainWindow):
    """PyQt6-based GUI for C3PO TTS Demo."""
    
    def __init__(self):
        super().__init__()
        self.generator = C3POGenerator()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("C3PO Text-to-Speech Demo")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Title and description
        title_label = QLabel("C3PO Voice Synthesizer")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 10px;")
        description = (
            "Enter text for C3PO to speak, adjust speed settings, and generate speech. "
            "You can also load text from a file or save your generated audio."
        )
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(desc_label)
        
        # Create a splitter for the main content
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter, 1)
        
        # Top section - Input controls
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # Text input area
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text for C3PO to speak...")
        self.text_input.setMinimumHeight(100)
        input_layout.addWidget(QLabel("Input Text:"))
        input_layout.addWidget(self.text_input)
        
        # Settings panel
        settings_widget = QWidget()
        settings_layout = QHBoxLayout(settings_widget)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speech Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(25)
        self.speed_value = QLabel("1.0x")
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value)
        settings_layout.addLayout(speed_layout)
        
        # Language selection
        language_layout = QVBoxLayout()
        language_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English (en)", "Spanish (es)", "French (fr)", "German (de)"])
        language_layout.addWidget(self.language_combo)
        settings_layout.addLayout(language_layout)
        
        # Output file selection
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Output File:"))
        output_file_layout = QHBoxLayout()
        self.output_path = QLineEdit("c3po_speech.wav")
        output_file_layout.addWidget(self.output_path)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_file)
        output_file_layout.addWidget(browse_button)
        output_layout.addLayout(output_file_layout)
        settings_layout.addLayout(output_layout)
        
        input_layout.addWidget(settings_widget)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Load text file button
        self.load_button = QPushButton("Load Text File")
        self.load_button.clicked.connect(self.load_text_file)
        button_layout.addWidget(self.load_button)
        
        # Generate button
        self.generate_button = QPushButton("Generate Speech")
        self.generate_button.clicked.connect(self.generate_speech)
        self.generate_button.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.generate_button)
        
        # Play button (disabled until speech is generated)
        self.play_button = QPushButton("Play Generated Audio")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.play_button)
        
        input_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        input_layout.addWidget(self.progress_bar)
        
        # Bottom section - Log output
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(QLabel("Log Output:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: monospace;")
        log_layout.addWidget(self.log_output)
        
        # Add the sections to the splitter
        splitter.addWidget(input_widget)
        splitter.addWidget(log_widget)
        splitter.setSizes([400, 200])
        
        # Set up the log handler to show logs in the UI
        self.setup_log_handler()
        
        # Initialize speech manager in a separate thread
        threading.Thread(target=self.initialize_speech_manager, daemon=True).start()
        
        # Set up a timer to update UI state
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui_state)
        self.timer.start(200)
        
        # Initialize state
        self.last_generated_file = None
        self.is_generating = False
    
    def update_speed_label(self, value):
        """Update the speed label when the slider is moved."""
        speed = value / 100.0
        self.speed_value.setText(f"{speed:.1f}x")
    
    def browse_output_file(self):
        """Open a file dialog to select output file location."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Speech File", "", "WAV Files (*.wav);;All Files (*)"
        )
        if filename:
            self.output_path.setText(filename)
    
    def load_text_file(self):
        """Load text from a file into the text input area."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Text File", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.text_input.setText(f.read())
                self.log_output.append(f"Loaded text from {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load text file: {str(e)}")
    
    def generate_speech(self):
        """Generate speech from the current text and settings."""
        if self.is_generating:
            QMessageBox.information(self, "In Progress", "Speech generation is already in progress.")
            return
        
        # Get inputs
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing Input", "Please enter text to generate speech.")
            return
        
        output_file = self.output_path.text()
        if not output_file:
            QMessageBox.warning(self, "Missing Output", "Please specify an output file.")
            return
        
        # Extract settings
        speed = self.speed_slider.value() / 100.0
        language = self.language_combo.currentText().split("(")[1].strip(")")
        
        # Update UI
        self.progress_bar.setValue(10)
        self.progress_bar.setFormat("Initializing...")
        self.is_generating = True
        self.update_ui_state()
        
        # Run in thread to keep UI responsive
        threading.Thread(
            target=self._generate_speech_thread,
            args=(text, output_file, speed, language),
            daemon=True
        ).start()
    
    def _generate_speech_thread(self, text, output_file, speed, language):
        """Thread function for speech generation."""
        try:
            self.progress_bar.setValue(20)
            self.progress_bar.setFormat("Generating speech...")
            
            # Generate speech
            result = self.generator.generate_speech(text, output_file, speed, language)
            
            # Update UI based on result
            if result:
                self.last_generated_file = output_file
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat("Speech generation complete!")
                QApplication.instance().beep()  # Audio notification
            else:
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("Generation failed")
                
        except Exception as e:
            self.log_output.append(f"Error: {str(e)}")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Error")
        finally:
            self.is_generating = False
            self.update_ui_state()
    
    def play_audio(self):
        """Play the generated audio file."""
        if not self.last_generated_file or not os.path.exists(self.last_generated_file):
            QMessageBox.warning(self, "File Not Found", "The audio file does not exist.")
            return
        
        # On Windows
        if sys.platform == "win32":
            os.startfile(self.last_generated_file)
        # On macOS
        elif sys.platform == "darwin":
            from subprocess import call
            call(["open", self.last_generated_file])
        # On Linux
        else:
            from subprocess import call
            call(["xdg-open", self.last_generated_file])
    
    def initialize_speech_manager(self):
        """Initialize the speech manager in a separate thread."""
        try:
            self.log_output.append("Initializing C3PO voice, please wait...")
            result = self.generator.initialize()
            if result:
                self.log_output.append("C3PO voice initialized successfully!")
            else:
                self.log_output.append("Failed to initialize C3PO voice.")
        except Exception as e:
            self.log_output.append(f"Initialization error: {str(e)}")
    
    def update_ui_state(self):
        """Update UI button states based on current state."""
        self.generate_button.setEnabled(not self.is_generating and self.generator.is_initialized)
        self.play_button.setEnabled(self.last_generated_file is not None and os.path.exists(self.last_generated_file))
        self.load_button.setEnabled(not self.is_generating)
    
    def setup_log_handler(self):
        """Set up a custom handler to show logs in the UI."""
        class QTextEditLogHandler(logging.Handler):
            def __init__(self, text_edit):
                super().__init__()
                self.text_edit = text_edit
                self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            
            def emit(self, record):
                msg = self.format(record)
                self.text_edit.append(msg)
        
        # Get the actual logger object from OARC's logging system
        from oarc.utils.log import get_logger
        logger = get_logger('oarc')
        
        # Add our custom handler to the logger
        handler = QTextEditLogHandler(self.log_output)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Add a log message to show it's working
        self.log_output.append("Log handler connected to OARC logging system")


def cli_mode(args):
    """Run in command-line mode."""
    generator = C3POGenerator()
    
    # Default output file if not specified
    if args.output is None:
        args.output = "c3po_speech.wav"
    
    # Process based on command
    if args.command == "text":
        if not args.text:
            log.error("No text provided for 'text' command")
            return 1
        
        log.info(f"Generating speech for text: '{args.text}'")
        result = generator.generate_speech(
            args.text, 
            args.output,
            speed=args.speed,
            language=args.language
        )
        
    elif args.command == "file":
        if not args.file or not os.path.exists(args.file):
            log.error(f"Text file not found: {args.file}")
            return 1
            
        log.info(f"Generating speech from file: {args.file}")
        result = generator.process_text_file(
            args.file, 
            args.output,
            speed=args.speed, 
            language=args.language
        )
    else:
        log.error(f"Unknown command: {args.command}")
        return 1
    
    if result:
        log.info(f"Speech generated successfully! File saved to: {args.output}")
        return 0
    else:
        log.error("Failed to generate speech.")
        return 1


def ui_mode():
    """Run in graphical user interface mode."""
    app = QApplication(sys.argv)
    window = C3POTTSUI()
    window.show()
    return app.exec()


def main():
    """Command-line interface for the C3PO TTS demo."""
    parser = argparse.ArgumentParser(description="OARC C3PO Text-to-Speech Demo")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Text command
    text_parser = subparsers.add_parser('text', help='Convert text to speech')
    text_parser.add_argument("text", help="Text to convert to speech")
    text_parser.add_argument("--output", help="Output WAV file path")
    text_parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    text_parser.add_argument("--language", default="en", help="Language code (e.g., en, es, fr)")
    
    # File command
    file_parser = subparsers.add_parser('file', help='Convert text file to speech')
    file_parser.add_argument("file", help="Text file to convert to speech")
    file_parser.add_argument("--output", help="Output WAV file path")
    file_parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    file_parser.add_argument("--language", default="en", help="Language code (e.g., en, es, fr)")
    
    # UI command
    subparsers.add_parser('ui', help='Launch graphical user interface')
    
    args = parser.parse_args()
    
    # Default to UI mode if no command specified
    if not args.command:
        return ui_mode()
    
    # Run in UI mode if requested
    if args.command == 'ui':
        return ui_mode()
    
    # Otherwise run in CLI mode
    return cli_mode(args)


if __name__ == "__main__":
    sys.exit(main())