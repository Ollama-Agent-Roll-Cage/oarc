import os
import sys
import random
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextBrowser, 
                            QLineEdit, QPushButton, QVBoxLayout, QWidget,
                            QHBoxLayout, QLabel, QComboBox, QSplitter, 
                            QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QImage

import ollama

# Default save location for game states
DEFAULT_SAVE_DIR = os.path.join(str(Path.home()), '.adventure_game')
SAVE_DIR = os.getenv('ADVENTURE_GAME_SAVE_DIR', DEFAULT_SAVE_DIR)
IMAGE_DIR = os.path.join(SAVE_DIR, 'images')

# Create necessary directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Thread for handling Ollama API calls
class OllamaThread(QThread):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    complete_signal = pyqtSignal()
    image_prompt_signal = pyqtSignal(str)
    
    def __init__(self, prompt, model, character_info, game_history, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.model = model
        self.character_info = character_info
        self.game_history = game_history
        
    def run(self):
        try:
            # Enhanced system prompt for better AI narration and image generation
            system_prompt = f"""You are the narrator of an immersive, epic fantasy text adventure game.

PLAYER CHARACTER: {self.character_info}

YOUR ROLE AS NARRATOR:
1. Create an epic, immersive fantasy experience with rich, vivid descriptions and world-building
2. Present meaningful choices, challenges, and consequences that respect player agency
3. Use descriptive, engaging language to make environments, characters and magic come alive
4. Balance action, discovery, dialogue, and character development
5. Track the player's inventory, abilities, and key plot developments

RESPONSE FORMAT:
- Provide detailed narrative responses (1-3 paragraphs)
- Use atmospheric descriptions that engage all senses
- For magical elements, describe their visual effects, sounds, and magical essence
- End ALL your responses with an image prompt: [IMAGE PROMPT: detailed, specific visual description]

IMAGE PROMPT GUIDELINES:
- Your [IMAGE PROMPT: ] MUST focus on the CURRENT scene's most visually interesting element
- Include SPECIFIC details like lighting, atmosphere, character positioning, environment details
- Describe magical effects, spellcasting visuals, and elemental manifestations in detail
- Format as [IMAGE PROMPT: detailed, specific visual description]
- Ensure the image prompt would make a compelling visual scene

Remember: You're creating an epic magical adventure. Focus on creating wonder, danger, and discovery in a rich fantasy world.
"""
            # Format the full conversation history to maintain context
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add the conversation history
            for entry in self.game_history:
                messages.append(entry)
                
            # Add the current user prompt
            messages.append({"role": "user", "content": self.prompt})
            
            # Get the response from Ollama
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )
            
            full_response = ""
            for chunk in stream:
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                elif isinstance(chunk, dict):
                    content = chunk.get('message', {}).get('content', '')
                
                if content:
                    full_response += content
                    self.response_signal.emit(content)
            
            # Extract image prompt if present
            image_prompt = None
            if "[IMAGE PROMPT:" in full_response:
                start_idx = full_response.find("[IMAGE PROMPT:")
                end_idx = full_response.find("]", start_idx)
                if end_idx > start_idx:
                    image_prompt = full_response[start_idx + 14:end_idx].strip()
                    self.image_prompt_signal.emit(image_prompt)
            
            self.complete_signal.emit()
                
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
            self.complete_signal.emit()

# Thread for generating images using FastFlux
class ImageGenerationThread(QThread):
    image_ready_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, prompt, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        
    def run(self):
        try:
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = random.randint(1000, 9999)
            filename = f"scene_{timestamp}_{random_suffix}"
            
            # Call the FastFlux script to generate an image
            script_path = os.path.abspath("fastFlux.py")
            
            # Modify the command according to your FastFlux setup
            # This is a simplified example - you'll need to adapt this to work with your actual FastFlux configuration
            modified_script = self.create_temp_script(self.prompt, filename)
            
            # Run the modified script
            result = subprocess.run(["python", modified_script], 
                                   capture_output=True, text=True)
            
            if result.returncode != 0:
                self.error_signal.emit(f"Image generation failed: {result.stderr}")
                return
                
            # Look for the generated image file
            image_path = self.find_generated_image(filename)
            if image_path:
                self.image_ready_signal.emit(image_path)
            else:
                self.error_signal.emit("Could not find the generated image.")
            
        except Exception as e:
            self.error_signal.emit(f"Error generating image: {str(e)}")
    
    def create_temp_script(self, prompt, filename):
        """Create a temporary modified version of fastFlux.py with our custom prompt"""
        
        # Create a temp file with modifications to the fastFlux.py script
        script_path = os.path.abspath("fastFlux.py")
        temp_script_path = os.path.join(SAVE_DIR, "temp_flux_script.py")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Enhanced prompt formatting for better image generation
        enhanced_prompt = f"{prompt}, fantasy artwork, 4k, highly detailed, epic lighting, cinematic, professional quality, vibrant colors"
        
        # Replace the specific prompt in the cliptextencode_6 section - this matches the exact text in your script
        search_text = 'text="neon, horrific armored ogre boss knight holding battle axe, ready to fight, guarding a cobblestone dungeon, video game engine, fpv, pov"'
        replacement_text = f'text="{enhanced_prompt}"'
        
        # Handle the case where the search text doesn't match exactly
        if search_text not in content:
            # Try to find the cliptextencode section more generically
            cliptextencode_pattern = r'cliptextencode_\d+\s*=\s*cliptextencode\.encode\(\s*text="[^"]*"'
            match = re.search(cliptextencode_pattern, content)
            if match:
                original_text = match.group(0)
                # Extract just the encode call part
                encode_part = original_text.split('text=')[0]
                # Create the new text with our prompt
                new_text = f'{encode_part}text="{enhanced_prompt}"'
                modified_content = content.replace(original_text, new_text)
            else:
                # Fallback modification if pattern not found
                self.status_label.setText("Warning: Custom pattern used for image generation")
                modified_content = content.replace('def main():', f'''def main():
    # Custom prompt: {enhanced_prompt}
    ''')
        else:
            modified_content = content.replace(search_text, replacement_text)
        
        # Also replace the filename_prefix in saveimage_9
        saveimage_pattern = r'filename_prefix\s*=\s*"[^"]*"'
        replacement = f'filename_prefix="{filename}"'
        
        modified_content = re.sub(saveimage_pattern, replacement, modified_content)
        
        # Save the modified script
        with open(temp_script_path, 'w') as f:
            f.write(modified_content)
            
        return temp_script_path
            
    def find_generated_image(self, filename_prefix):
        """Find the image file generated by FastFlux"""
        # Enhanced image finding with more detailed logging and directory scanning
        self.status_label.setText("Searching for generated image...")
        
        # Expanded list of possible output directories
        possible_directories = [
            os.getcwd(),  # Current directory
            os.path.join(os.getcwd(), "output"),  # Common output dir
            os.path.join(os.getcwd(), "ComfyUI", "output"),  # ComfyUI output
            os.path.expanduser("~/ComfyUI/output"),  # Home ComfyUI output
            os.path.join(os.getcwd(), "output", "samples"),  # Nested output
            os.path.join(os.getcwd(), "outputs"),  # Alternative name
            os.path.join(os.getcwd(), "generated"),  # Alternative name
            # Add ComfyUI specific paths - including output and its default date-based subdirectories
            os.path.join(os.getcwd(), "ComfyUI", "output", datetime.now().strftime("%Y-%m-%d")),
        ]
        
        # For debugging - print all directories being searched
        print(f"Searching for image with prefix: {filename_prefix}")
        print(f"Checking directories: {possible_directories}")
        
        for directory in possible_directories:
            if os.path.exists(directory):
                print(f"Scanning directory: {directory}")
                files = os.listdir(directory)
                print(f"Found {len(files)} files in directory")
                
                # Look for exact filename match first
                for file in files:
                    if file.startswith(filename_prefix) and file.endswith((".png", ".jpg", ".jpeg")):
                        print(f"Found matching file: {file}")
                        source_path = os.path.join(directory, file)
                        dest_path = os.path.join(IMAGE_DIR, file)
                        
                        # Simple file copy
                        with open(source_path, 'rb') as src:
                            with open(dest_path, 'wb') as dst:
                                dst.write(src.read())
                                
                        print(f"Image copied to: {dest_path}")
                        self.status_label.setText("Image found and loaded!")
                        return dest_path
                
                # If no exact match, look for the most recent image file as fallback
                image_files = [f for f in files if f.endswith((".png", ".jpg", ".jpeg"))]
                if image_files:
                    # Sort by creation time, most recent first
                    image_files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True)
                    newest_file = image_files[0]
                    print(f"No exact match found, using most recent image: {newest_file}")
                    
                    source_path = os.path.join(directory, newest_file)
                    dest_path = os.path.join(IMAGE_DIR, f"{filename_prefix}_{newest_file}")
                    
                    # Copy the file
                    with open(source_path, 'rb') as src:
                        with open(dest_path, 'wb') as dst:
                            dst.write(src.read())
                    
                    print(f"Most recent image copied to: {dest_path}")
                    self.status_label.setText("Using most recent generated image")
                    return dest_path
        
        print("No image found in any directory")
        self.status_label.setText("Image not found - using placeholder")
        
        # If no image found, create a placeholder image
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a placeholder image with text
        img = Image.new('RGB', (800, 450), color=(0, 20, 40))
        d = ImageDraw.Draw(img)
        
        # Add text explaining the issue
        text = "Image Generation Pending..."
        text_color = (0, 200, 255)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
            
        # Calculate text position for centering
        text_width, text_height = d.textbbox((0, 0), text, font=font)[2:4]
        position = ((800 - text_width) // 2, (450 - text_height) // 2)
        
        # Draw the text
        d.text(position, text, fill=text_color, font=font)
        
        # Draw a futuristic border
        d.rectangle([(20, 20), (780, 430)], outline=(0, 150, 200), width=2)
        
        # Save the placeholder image
        placeholder_path = os.path.join(IMAGE_DIR, f"{filename_prefix}_placeholder.png")
        img.save(placeholder_path)
        
        return placeholder_path

class FantasyAdventureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_threads = []
        self.character_info = ""
        self.game_history = []
        self.setup_ui()
        self.fetch_models()
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        
        # Header with title - Futuristic design
        header_layout = QHBoxLayout()
        
        # Animated logo (simulated with multiple symbols)
        self.logo_index = 0
        self.logo_symbols = ["⚡", "🔥", "💧", "✨"]
        self.logo_label = QLabel(self.logo_symbols[0])
        self.logo_label.setFont(QFont("Arial", 28))
        self.logo_label.setStyleSheet("color: #00FFFF;")
        
        # Animation timer for logo
        self.logo_timer = QTimer(self)
        self.logo_timer.timeout.connect(self.animate_logo)
        self.logo_timer.start(500)  # Update every 500ms
        
        header = QLabel("ARCANE NEXUS")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #0CCCFF; letter-spacing: 3px; text-shadow: 0 0 10px #00AAFF;")
        
        header_layout.addWidget(self.logo_label)
        header_layout.addWidget(header)
        header_layout.addStretch()
        
        # Model selection with futuristic styling
        model_layout = QHBoxLayout()
        model_label = QLabel("◢ NEURAL CORE ◣")
        model_label.setStyleSheet("color: #00FFAA; font-weight: bold;")
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(40)
        self.model_combo.addItem("⌛ Initializing neural cores...")
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(0, 40, 60, 0.8);
                color: #00FFFF;
                border: 1px solid #00AAAA;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: 0px;
                background: rgba(0, 80, 120, 0.6);
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(0, 20, 40, 0.95);
                color: #00FFFF;
                selection-background-color: rgba(0, 120, 180, 0.8);
                selection-color: #FFFFFF;
                border: 1px solid #00AAAA;
            }
        """)
        
        refresh_button = QPushButton("⟳")
        refresh_button.setFixedWidth(50)
        refresh_button.setToolTip("Rescan neural cores")
        refresh_button.clicked.connect(self.fetch_models)
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 80, 120, 0.8);
                color: #00FFFF;
                border: 1px solid #00AAAA;
                border-radius: 4px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 100, 140, 0.8);
                border: 1px solid #00FFFF;
            }
            QPushButton:pressed {
                background-color: rgba(0, 60, 100, 0.8);
            }
        """)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 3)
        model_layout.addWidget(refresh_button)
        
        # Image display area with holographic styling
        self.image_label = QLabel("◢◤ REALITY PROJECTION INITIALIZING ◥◣")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(320)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 20, 40, 0.7);
                color: #00AACC;
                border: 2px solid #00AACC;
                border-radius: 8px;
                padding: 4px;
                background-image: radial-gradient(circle, rgba(0, 40, 80, 0.6), rgba(0, 10, 30, 0.9));
            }
        """)
        
        # Add a frame around the image for more futuristic look
        image_frame = QWidget()
        image_frame_layout = QVBoxLayout(image_frame)
        image_frame_layout.setContentsMargins(0, 0, 0, 0)
        image_frame_layout.addWidget(self.image_label)
        
        image_frame.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: 1px solid #00FFFF;
                border-radius: 10px;
                padding: 2px;
            }
        """)
        
        # Game text display with terminal-like styling
        self.game_display = QTextBrowser()
        self.game_display.setOpenExternalLinks(False)
        self.game_display.setReadOnly(True)
        self.game_display.setMinimumHeight(300)
        self.game_display.setStyleSheet("""
            QTextBrowser {
                background-color: rgba(0, 15, 30, 0.9);
                color: #00FFDD;
                border: 1px solid #00AAAA;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                selection-background-color: rgba(0, 200, 255, 0.5);
                selection-color: #FFFFFF;
            }
            QScrollBar:vertical {
                background-color: rgba(0, 20, 40, 0.6);
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(0, 180, 220, 0.7);
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Character creation button with futuristic styling
        self.character_button = QPushButton("◢ INITIALIZE CHARACTER MATRIX ◣")
        self.character_button.setMinimumHeight(50)
        self.character_button.clicked.connect(self.show_character_creation)
        self.character_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #004455, stop:1 #006699);
                color: #00FFFF;
                border: 2px solid #00AACC;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #005566, stop:1 #0088CC);
                border: 2px solid #00FFFF;
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #003344, stop:1 #005577);
            }
        """)
        
        # Input area with futuristic styling
        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("▶ ENTER COMMAND SEQUENCE...")
        self.prompt_input.setMinimumHeight(50)
        self.prompt_input.returnPressed.connect(self.send_action)
        self.prompt_input.setEnabled(False)  # Disabled until character is created
        self.prompt_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 30, 50, 0.8);
                color: #00FFDD;
                border: 2px solid #00AACC;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: bold;
                font-family: 'Consolas', 'Courier New', monospace;
                selection-background-color: rgba(0, 180, 255, 0.5);
            }
            QLineEdit:focus {
                border: 2px solid #00FFFF;
                background-color: rgba(0, 40, 70, 0.9);
            }
            QLineEdit:disabled {
                background-color: rgba(0, 20, 40, 0.6);
                color: #007799;
                border: 2px solid #005577;
            }
        """)
        
        self.send_button = QPushButton("◢ TRANSMIT ◣")
        self.send_button.setMinimumHeight(50)
        self.send_button.setMinimumWidth(120)
        self.send_button.clicked.connect(self.send_action)
        self.send_button.setEnabled(False)  # Disabled until character is created
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 100, 160, 0.8);
                color: #00FFFF;
                border: 2px solid #00AACC;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 130, 200, 0.8);
                border: 2px solid #00FFFF;
                color: white;
            }
            QPushButton:pressed {
                background-color: rgba(0, 80, 130, 0.8);
            }
            QPushButton:disabled {
                background-color: rgba(0, 60, 100, 0.4);
                color: #007788;
                border: 2px solid #005566;
            }
        """)
        
        # Status indicator with animated styling
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00AACC;
                font-weight: bold;
                background-color: transparent;
                padding: 5px;
            }
        """)
        
        input_layout.addWidget(self.prompt_input, 4)
        input_layout.addWidget(self.send_button, 1)
        
        # Add all elements to main layout
        main_layout.addLayout(header_layout)
        main_layout.addLayout(model_layout)
        main_layout.addWidget(image_frame, 3)
        main_layout.addWidget(self.game_display, 4)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.character_button)
        main_layout.addLayout(input_layout)
        
        # Set main widget
        self.setCentralWidget(main_widget)
        
        # Window settings - fixed reasonable size to avoid geometry errors
        self.setWindowTitle("Arcane Nexus: Interactive Realm")
        self.setGeometry(300, 200, 1000, 800)
        self.setMinimumSize(900, 700)  # Set minimum size to avoid geometry issues
        
        # Apply futuristic theme to the entire window
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #081520;
                color: #00CCDD;
                font-family: 'Arial', sans-serif;
            }
            QToolTip {
                background-color: rgba(0, 20, 40, 0.9);
                color: #00FFFF;
                border: 1px solid #00AAAA;
                padding: 5px;
            }
        """)
        self.show()
    
    def animate_logo(self):
        """Animate the logo by cycling through symbols"""
        self.logo_index = (self.logo_index + 1) % len(self.logo_symbols)
        self.logo_label.setText(self.logo_symbols[self.logo_index])
    
    def fetch_models(self):
        try:
            # Get the list of models from ollama
            result = ollama.list()
            
            models = []
            if isinstance(result, dict) and 'models' in result:
                models = [model.get('name', '') for model in result['models']]
            elif hasattr(result, 'models'):
                models = [model.model for model in result.models if hasattr(model, 'model')]
            
            self.model_combo.clear()
            if models:
                self.model_combo.addItems(models)
                self.model_combo.setCurrentIndex(0)
                self.add_game_message(f"Ready to begin your adventure! Click 'Create Character' to start.")
            else:
                self.model_combo.addItem("No models found")
                self.add_game_message("No AI models found. Please pull a model using 'ollama pull <model>' command first.")
        
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItem("Error loading models")
            self.add_game_message(f"Error loading AI models: {str(e)}\nMake sure Ollama is installed and running.")
    
    def add_game_message(self, content):
        """Add a message to the game display"""
        # Properly format the content with paragraph tags if needed
        if not content.startswith("<"):
            formatted_content = f"<p>{content}</p>"
        else:
            formatted_content = content
            
        self.game_display.append(formatted_content)
        
        # Scroll to bottom
        scrollbar = self.game_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def show_character_creation(self):
        """Show the character creation dialog"""
        self.game_display.clear()
        self.add_game_message("<h2>Character Creation</h2>")
        self.add_game_message("Enter your character details and the setting for your adventure.")
        
        # Change button to submission
        self.character_button.setText("Submit Character & Begin Adventure")
        self.character_button.clicked.disconnect()
        self.character_button.clicked.connect(self.create_character)
        
        # Enable input for character creation
        self.prompt_input.setEnabled(True)
        self.prompt_input.setPlaceholderText("Enter character name, traits, and world setting...")
        self.send_button.setEnabled(True)
        
        # Connect send button to character creation
        self.send_button.clicked.disconnect()
        self.send_button.clicked.connect(self.create_character)
    
    def create_character(self):
        """Process character creation and start the game"""
        character_info = self.prompt_input.text().strip()
        if not character_info:
            self.add_game_message("<span style='color: #ff5555;'>Please enter information about your character and setting.</span>")
            return
        
        # Store character info
        self.character_info = character_info
        
        # Clear previous game
        self.game_history = []
        
        # Switch UI to game mode
        self.character_button.hide()
        self.prompt_input.clear()
        self.prompt_input.setPlaceholderText("What do you do?")
        
        # Reconnect send button to regular action
        self.send_button.clicked.disconnect()
        self.send_button.clicked.connect(self.send_action)
        
        # Start the game
        self.add_game_message("<h2>Adventure Begins</h2>")
        self.add_game_message(f"<b>Character/Setting:</b> {self.character_info}")
        
        # Send initial prompt to get the game started
        initial_prompt = f"I'm playing as: {self.character_info}. Start the adventure by describing the opening scene."
        self.send_to_ai(initial_prompt, is_start=True)
    
    def send_action(self):
        """Send player action to the AI"""
        action = self.prompt_input.text().strip()
        if not action:
            return
        
        # Add the action to the display
        self.add_game_message(f"<p><b>You:</b> {action}</p>")
        
        # Clear input
        self.prompt_input.clear()
        
        # Send to AI
        self.send_to_ai(action)
    
    def send_to_ai(self, prompt, is_start=False):
        """Send prompt to Ollama and process response"""
        selected_model = self.model_combo.currentText()
        if selected_model in ["Loading models...", "No models found", "Error loading models"]:
            self.add_game_message("<span style='color: #ff5555;'>Please select a valid AI model first</span>")
            return
        
        # Disable input during processing
        self.prompt_input.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Add user message to history if not the start prompt
        if not is_start:
            self.game_history.append({"role": "user", "content": prompt})
        
        # Show status
        self.status_label.setText("The narrator is thinking...")
        
        # Create and start Ollama thread
        self.ollama_thread = OllamaThread(prompt, selected_model, self.character_info, self.game_history)
        
        # Connect signals
        self.ollama_thread.response_signal.connect(self.handle_ai_response)
        self.ollama_thread.error_signal.connect(self.handle_ai_error)
        self.ollama_thread.complete_signal.connect(self.on_ai_complete)
        self.ollama_thread.image_prompt_signal.connect(self.generate_scene_image)
        
        # Track thread
        self.active_threads.append(self.ollama_thread)
        self.ollama_thread.start()
    
    def handle_ai_response(self, content):
        """Handle streaming response from AI"""
        # Only display content without the image prompt tags
        displayed_content = content
        if "[IMAGE PROMPT:" in displayed_content:
            displayed_content = displayed_content.split("[IMAGE PROMPT:")[0]
        
        # Add to game display
        if displayed_content:
            # Instead of adding a new message, update the current response
            # Check if we already have a narrator message
            if not hasattr(self, 'current_response'):
                self.current_response = ""
                self.game_display.append("<span style='color: #aaffaa;'></span>")
            
            # Update the current response
            self.current_response += displayed_content
            
            # Replace the last paragraph with the updated text
            cursor = self.game_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.movePosition(cursor.MoveOperation.PreviousBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(f"<span style='color: #aaffaa;'>{self.current_response}</span>")
    
    def handle_ai_error(self, error):
        """Handle AI error"""
        self.add_game_message(f"<span style='color: #ff5555;'>{error}</span>")
        self.status_label.setText("")
        
        # Re-enable input
        self.prompt_input.setEnabled(True)
        self.send_button.setEnabled(True)
    
    def on_ai_complete(self):
        """Clean up when AI response is complete"""
        # Re-enable input
        self.prompt_input.setEnabled(True)
        self.send_button.setEnabled(True)
        
        # Clear status
        self.status_label.setText("")
        
        # Add the full response to game history
        if hasattr(self, 'current_response'):
            # Strip out image prompt tags for the history
            clean_response = self.current_response
            if "[IMAGE PROMPT:" in clean_response:
                clean_response = clean_response.split("[IMAGE PROMPT:")[0]
            
            self.game_history.append({"role": "assistant", "content": clean_response})
            
            # Reset current response
            delattr(self, 'current_response')
        
        # Clean up thread
        if self.ollama_thread in self.active_threads:
            self.active_threads.remove(self.ollama_thread)
    
    def generate_scene_image(self, image_prompt):
        """Generate an image for the current scene"""
        # Update status
        self.status_label.setText("Generating scene image...")
        
        # Create and start image generation thread
        self.image_thread = ImageGenerationThread(image_prompt)
        
        # Connect signals
        self.image_thread.image_ready_signal.connect(self.display_image)
        self.image_thread.error_signal.connect(self.handle_image_error)
        
        # Track thread
        self.active_threads.append(self.image_thread)
        self.image_thread.start()
    
    def display_image(self, image_path):
        """Display the generated image"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
            self.status_label.setText("")
        else:
            self.handle_image_error(f"Image file not found: {image_path}")
    
    def handle_image_error(self, error):
        """Handle image generation error"""
        self.status_label.setText(f"Image error: {error}")
        
        # Set a placeholder or default image
        self.image_label.setText("Failed to generate scene image")
    
    def closeEvent(self, event):
        # Clean up threads when closing
        for thread in self.active_threads:
            if thread.isRunning():
                thread.terminate()
                thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FantasyAdventureApp()
    sys.exit(app.exec())