"""
Speech utilities for OARC.
This module provides utility functions for working with speech files and audio processing.
"""

import os
import re
from oarc.utils.log import log
from oarc.utils.paths import Paths

class SpeechUtils:
    """Utility functions for speech processing."""

    @staticmethod
    def ensure_voice_reference_file(voice_name):
        """
        Ensure the voice reference file exists for TTS.
        Raises an error if the voice reference file doesn't exist.

        Args:
            voice_name (str): Name of the voice reference to ensure
            
        Returns:
            bool: True if the voice reference file exists
            
        Raises:
            FileNotFoundError: If the voice reference file doesn't exist
        """
        # Properly get the Paths instance
        paths = Paths()  # The singleton decorator will return the instance
        voice_ref_dir = os.path.join(paths.get_voice_reference_dir(), voice_name)
        voice_ref_file = os.path.join(voice_ref_dir, "clone_speech.wav")
        
        if not os.path.exists(voice_ref_file):
            models_dir = paths.get_model_dir()
            error_message = (
                f"Voice reference file not found: {voice_ref_file}\n"
                f"Please ensure you have created the voice reference directory at: {voice_ref_dir}\n"
                f"And added a valid 'clone_speech.wav' file in that directory.\n"
                f"For more information, see the documentation on setting up voice references."
            )
            log.error(error_message)
            raise FileNotFoundError(error_message)
        
        log.info(f"Voice reference file found: {voice_ref_file}")
        return True
    
    @staticmethod
    def extract_repo_id_from_url(url):
        """
        Extract the repository ID from a HuggingFace repository URL.
        
        This function handles various URL formats from HuggingFace:
        - https://huggingface.co/username/repo-name/tree/main
        - https://huggingface.co/username/repo-name
        - https://huggingface.co/username/repo-name/
        - https://huggingface.co/username/repo-name/tree/branch
        
        Args:
            url (str): The HuggingFace repository URL
            
        Returns:
            str: The extracted repository ID in the format "username/repo-name"
            
        Raises:
            ValueError: If the URL does not appear to be a valid HuggingFace repository URL
        """
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        
        # Check if it's a HuggingFace URL
        if "huggingface.co" not in url:
            raise ValueError("URL does not appear to be a HuggingFace repository URL")
        
        # Method 1: Simple string replacement for standard format
        repo_id = url.replace("https://huggingface.co/", "")
        
        # Remove any trailing paths like /tree/main, /tree/branch, etc.
        pattern = r'(/tree/[^/]+/?.*$)|(/blob/.*$)|(/resolve/.*$)'
        repo_id = re.sub(pattern, '', repo_id)
        
        # Remove any trailing slashes
        repo_id = repo_id.rstrip('/')
        
        # Validate the result has the expected format (username/repo-name)
        if not re.match(r'^[^/]+/[^/]+$', repo_id):
            log.warning(f"Extracted repo ID '{repo_id}' may not be in the expected format")
        
        log.debug(f"Extracted repo ID '{repo_id}' from URL '{url}'")
        return repo_id
    
    @staticmethod
    def download_voice_pack(url, voice_name=None):
        """
        Download a voice pack from a HuggingFace URL.
        
        This is a convenience method that extracts the repo ID from the URL
        and delegates to SpeechManager for the actual download.
        
        Args:
            url (str): The HuggingFace repository URL
            voice_name (str, optional): Name to save the voice pack as
            
        Returns:
            tuple: (path to voice pack, success status)
            
        Raises:
            ValueError: If the URL is invalid
            ImportError: If SpeechManager cannot be imported
        """
        try:
            # Extract the repo ID from the URL
            repo_id = SpeechUtils.extract_repo_id_from_url(url)
            log.info(f"Downloading voice pack from repo_id: {repo_id}")
            
            # Import SpeechManager here to avoid circular imports
            from oarc.speech import SpeechManager
            
            # Get SpeechManager instance and download the pack - use the singleton instance directly
            manager = SpeechManager()  # The singleton decorator will return the instance
            return manager.download_voice_pack(repo_id, voice_name)
            
        except ImportError as e:
            log.error(f"Failed to import SpeechManager: {e}")
            raise
        except Exception as e:
            log.error(f"Failed to download voice pack: {e}")
            return None, False

    @staticmethod
    def find_voice_reference_file(voice_name, paths=None):
        """
        Find voice reference file for a specific voice.
        Will check both the 'reference.wav' and 'clone_speech.wav' files.
        
        Args:
            voice_name (str): Name of the voice to find
            paths (Paths, optional): Paths instance to use
            
        Returns:
            str: Path to voice reference file or None if not found
        """
        if paths is None:
            paths = Paths()  # Get singleton instance
            
        voice_ref_dir = os.path.join(paths.get_voice_reference_dir(), voice_name)
        
        if not os.path.exists(voice_ref_dir):
            log.warning(f"Voice reference directory not found: {voice_ref_dir}")
            return None
            
        # Check for clone_speech.wav first (preferred)
        clone_speech_path = os.path.join(voice_ref_dir, "clone_speech.wav")
        if os.path.exists(clone_speech_path):
            log.info(f"Found clone_speech.wav for voice {voice_name}")
            return clone_speech_path
            
        # Then check for reference.wav
        reference_path = os.path.join(voice_ref_dir, "reference.wav")
        if os.path.exists(reference_path):
            log.info(f"Found reference.wav for voice {voice_name}")
            return reference_path
            
        log.warning(f"No reference files found for voice {voice_name}")
        return None
        
    @staticmethod
    def list_available_voices(paths=None):
        """
        List all available voices in the voice reference directory
        
        Args:
            paths (Paths, optional): Paths instance to use
            
        Returns:
            list: List of available voice names
        """
        if paths is None:
            paths = Paths()  # Get singleton instance
            
        voice_ref_dir = paths.get_voice_reference_dir()
        
        if not os.path.exists(voice_ref_dir):
            log.warning(f"Voice reference directory not found: {voice_ref_dir}")
            return []
            
        try:
            # Get all subdirectories in the voice reference directory
            voices = [d for d in os.listdir(voice_ref_dir) 
                     if os.path.isdir(os.path.join(voice_ref_dir, d))]
                     
            # Filter to only include directories that have a reference file
            valid_voices = []
            for voice in voices:
                if SpeechUtils.find_voice_reference_file(voice, paths):
                    valid_voices.append(voice)
                    
            log.info(f"Found {len(valid_voices)} valid voices: {', '.join(valid_voices)}")
            return valid_voices
            
        except Exception as e:
            log.error(f"Error listing available voices: {e}")
            return []
