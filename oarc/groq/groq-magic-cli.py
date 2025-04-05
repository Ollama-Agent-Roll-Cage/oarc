#!/usr/bin/env python3
"""
✨ GroqMagic CLI
A command-line interface for interacting with Groq AI models.

Usage:
  groq-magic chat --model llama-3-8b --prompt "Tell me a joke"
  groq-magic vision --image photo.jpg --prompt "What's in this image?"
  groq-magic transcribe --file audio.mp3
  groq-magic moderate --content "Check if this content is appropriate"
  groq-magic code --prompt "Create a Python function to sort a list"
  groq-magic list
"""

import sys
import os

# Add the parent directory to sys.path to import GroqMagic module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from groq_magic import GroqMagic

if __name__ == "__main__":
    # Call the CLI function from the module
    from groq_magic import cli
    cli()
