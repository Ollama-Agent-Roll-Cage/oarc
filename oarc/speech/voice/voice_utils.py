"""
Voice Utilities Module for OARC.

This module provides utility functions for working with voice references
and voice packs. It includes functions for finding, listing, and 
normalizing voice files and names.
"""

import os
import shutil

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.speech.voice.voice_type import VoiceType
from oarc.speech.voice.voice_ref_pack import VoiceRefPackType

class VoiceUtils:
    """
    A collection of utility functions for working with voice references.
    """
    
    @staticmethod
    def find_voice_ref_file(voice_name, paths=None):
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
            paths = Paths()
            
        voice_ref_dir = os.path.join(paths.get_voice_ref_path(), voice_name)
        
        if not os.path.exists(voice_ref_dir):
            log.info(f"Voice reference directory not found: {voice_ref_dir}")
            return None
            
        # Check for clone_speech.wav first (preferred)
        clone_speech_path = os.path.join(voice_ref_dir, "clone_speech.wav")
        if os.path.exists(clone_speech_path):
            log.info(f"Found clone_speech.wav for voice {voice_name}")
            return clone_speech_path
            
        # If not found, check for reference.wav
        reference_path = os.path.join(voice_ref_dir, "reference.wav")
        if os.path.exists(reference_path):
            log.info(f"Found reference.wav for voice {voice_name}")
            return reference_path
            
        log.info(f"No reference files found for voice {voice_name}")
        return None
    
    @staticmethod
    def list_available_voices(paths=None):
        """
        List all available voices in the voice reference directory.
        
        This method scans the voice reference directory for subdirectories
        that contain valid voice reference files (e.g., 'clone_speech.wav' or 'reference.wav').
        
        Args:
            paths (Paths, optional): An instance of the Paths class to use. If not provided,
                         the singleton instance of Paths will be used.
            
        Returns:
            list: A list of available voice names that have valid reference files.
        """
        if paths is None:
            paths = Paths()
            
        voice_ref_dir = paths.get_voice_ref_path()
        
        if not os.path.exists(voice_ref_dir):
            log.warning(f"Voice reference directory not found: {voice_ref_dir}")
            return []
            
        try:
            # Get all subdirectories in the voice reference directory
            voices = [d for d in os.listdir(voice_ref_dir) 
                     if os.path.isdir(os.path.join(voice_ref_dir, d))]
                     
            # Filter directories with valid voice reference files
            valid_voices = []
            for voice in voices:
                if VoiceUtils.find_voice_ref_file(voice, paths):
                    valid_voices.append(voice)
                    
            log.info(f"Found {len(valid_voices)} valid voices: {', '.join(valid_voices)}")
            return valid_voices
            
        except Exception as e:
            log.error(f"Error listing available voices: {e}")
            return []
    
    @staticmethod
    def normalize_voice_name(repo_id, voice_name=None):
        """
        Process and normalize a voice name from a repository ID.
        
        If voice_name is not provided, it will be derived from the 
        last part of the repository ID. XTTS-v2_ prefix will be removed.
        
        Args:
            repo_id (str): HuggingFace repository ID (e.g., "username/repo-name")
            voice_name (str, optional): Custom name to use instead. Defaults to None.
            
        Returns:
            str: Processed voice name suitable for voice pack directory
        """
        if voice_name:
            return voice_name
            
        voice_name = repo_id.split('/')[-1].lower()
        # Remove XTTS-v2_ prefix if present
        if voice_name.startswith("xtts-v2_"):
            voice_name = voice_name[8:]
            
        return voice_name
    
    @staticmethod
    def get_voice_ref_path(voice_name, paths=None):
        """
        Get the full path to a voice reference directory.
        
        Args:
            voice_name (str): Name of the voice
            paths (Paths, optional): Paths instance to use
            
        Returns:
            str: Full path to the voice reference directory
        """
        if paths is None:
            paths = Paths()
            
        return os.path.join(paths.get_voice_ref_path(), voice_name)
    
    @staticmethod
    def verify_voice_pack(voice_dir):
        """
        Verify that a voice reference pack directory contains the necessary files.
        
        Args:
            voice_dir (str): Path to the voice pack directory
            
        Returns:
            bool: True if the voice pack contains required files
        """
        if not os.path.isdir(voice_dir):
            log.warning(f"Voice directory not found: {voice_dir}")
            return False
            
        # Check for reference audio file (either reference.wav or clone_speech.wav)
        has_ref = (
            os.path.isfile(os.path.join(voice_dir, "reference.wav")) or
            os.path.isfile(os.path.join(voice_dir, "clone_speech.wav"))
        )
        
        # For fine-tuned models, check for model files
        has_model = os.path.isfile(os.path.join(voice_dir, "model.pth"))
        has_config = os.path.isfile(os.path.join(voice_dir, "config.json"))
        has_vocab = os.path.isfile(os.path.join(voice_dir, "vocab.json"))
        
        # Log the verification results
        if not has_ref:
            log.warning(f"Voice pack missing reference audio file")
        if not has_model:
            log.debug(f"Voice pack missing model.pth file")
        if not has_config:
            log.debug(f"Voice pack missing config.json file")
        if not has_vocab:
            log.debug(f"Voice pack missing vocab.json file")
        
        # A valid voice pack must at least have a reference file
        if has_ref:
            log.info(f"Voice pack {os.path.basename(voice_dir)} has valid reference file")
            return True
            
        return False
        
    @staticmethod
    def verify_model_pack(model_dir):
        """
        Verify that a model directory contains the necessary files.
        
        Args:
            model_dir (str): Path to the model directory
            
        Returns:
            bool: True if the model pack contains required files
        """
        if not os.path.isdir(model_dir):
            log.warning(f"Model directory not found: {model_dir}")
            return False
            
        # Define required files for a valid model
        required_files = ["config.json", "model.pth"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        # Log the verification results
        if missing_files:
            log.error(f"Model pack missing required files: {', '.join(missing_files)}")
            return False
        else:
            log.info(f"Model pack {os.path.basename(model_dir)} successfully verified")
            return True
            
    @staticmethod
    def ensure_clone_speech_exists(voice_dir):
        """
        Ensure that a clone_speech.wav file exists in the voice directory.
        If only reference.wav exists, create a copy named clone_speech.wav.
        
        Args:
            voice_dir (str): Path to the voice directory
            
        Returns:
            str: Path to the clone_speech.wav file or None if neither exists
        """
        ref_file = os.path.join(voice_dir, "reference.wav")
        clone_speech_file = os.path.join(voice_dir, "clone_speech.wav")
        
        if os.path.exists(clone_speech_file):
            return clone_speech_file
        elif os.path.exists(ref_file):
            try:
                shutil.copy2(ref_file, clone_speech_file)
                log.info(f"Copied reference.wav to clone_speech.wav for compatibility")
                return clone_speech_file
            except Exception as e:
                log.error(f"Error creating clone_speech.wav: {e}")
                return ref_file
        else:
            return None

    @staticmethod
    def copy_reference_from_model(model_path, voice_name, paths=None):
        """
        Copy reference audio file from model directory to voice reference directory.
        
        Args:
            model_path (str): Path to the model directory
            voice_name (str): Name of the voice
            paths (Paths, optional): Paths instance to use
            
        Returns:
            bool: True if reference file was copied successfully, False otherwise
        """
        if paths is None:
            paths = Paths()
            
        # Clean voice name (remove XTTS-v2_ prefix if present)
        clean_voice_name = voice_name.replace("XTTS-v2_", "")
            
        # Create voice reference directory if it doesn't exist
        voice_ref_dir = os.path.join(paths.get_voice_ref_path(), clean_voice_name)
        os.makedirs(voice_ref_dir, exist_ok=True)
        
        # Find reference file in model directory
        ref_file = None
        for ref_name in ["reference.wav", "clone_speech.wav"]:
            potential_ref = os.path.join(model_path, ref_name)
            if os.path.exists(potential_ref):
                ref_file = potential_ref
                log.info(f"Found reference file in model directory: {ref_file}")
                break
                
        if ref_file:
            # Copy the reference file to the voice reference directory as reference.wav
            voice_ref_path = os.path.join(voice_ref_dir, "reference.wav")
            shutil.copy2(ref_file, voice_ref_path)
            log.info(f"Copied reference file from {ref_file} to {voice_ref_path}")
            return True
        else:
            log.warning(f"No reference file found in model directory: {model_path}")
            return False