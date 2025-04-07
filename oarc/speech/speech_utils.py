"""
Speech Utilities Module for OARC.

This module provides a collection of utility functions to assist with
speech processing tasks, including managing voice references, downloading
voice packs, and interacting with HuggingFace repositories.

Features:
- Ensure the existence of voice reference files.
- Extract repository IDs from HuggingFace URLs.
- Download voice packs from HuggingFace repositories.
- Locate and validate voice reference files.
- List all available voices with valid reference files.
"""

import os
import shutil

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.speech.voice.voice_type import VoiceType

class SpeechUtils:
    """
    A collection of utility functions for speech processing tasks.
    These functions assist in managing voice references, downloading voice packs,
    and interacting with HuggingFace repositories.
    """

    @staticmethod
    def get_non_conflicting_filename(base_filename):
        """
        Generate a non-conflicting filename by appending an incremental number.
        
        Args:
            base_filename (str): The original filename
            
        Returns:
            str: A filename that doesn't conflict with existing files
        """
        if not os.path.exists(base_filename):
            return base_filename
            
        # Split the filename into name and extension
        file_dir = os.path.dirname(base_filename)
        filename_only = os.path.basename(base_filename)
        name, ext = os.path.splitext(filename_only)
        
        counter = 1
        new_filename = os.path.join(file_dir, f"{name}_{counter}{ext}")
        
        # Keep incrementing the counter until we find a non-existing filename
        while os.path.exists(new_filename):
            counter += 1
            new_filename = os.path.join(file_dir, f"{name}_{counter}{ext}")
            
        log.info(f"Generated non-conflicting filename: {new_filename}")
        return new_filename

    @staticmethod
    def ensure_voice_reference_exists(voice_name):
        """
        Ensure voice reference file exists, download if needed.
        
        Flow:
        1. Check if base Coqui XTTS v2 model exists in models/coqui/xtts, download if needed
        2. Check if voice-specific model exists in models/coqui/custom_xtts_v2, download if needed
        3. Copy reference.wav from custom model to voice_ref_pack directory
        
        Directory Structure:
        - Base XTTS model: models/coqui/xtts (no reference files, just model files)
        - Fine-tuned models: models/coqui/custom_xtts_v2/[voice_name] (includes reference.wav)
        - Voice references: models/coqui/voice_ref_pack/[voice_name]/reference.wav
        
        Args:
            voice_name: Name of the voice to check/download
            
        Returns:
            bool: True if voice reference exists or was successfully downloaded
        """
        # Import here to avoid circular imports
        from oarc.speech.voice.voice_ref_pack import VoiceRefPackType
        from oarc.hf.hf_utils import HfUtils
        from oarc.speech.voice.voice_utils import VoiceUtils
        
        paths = Paths()
        
        # Get the relevant paths
        voice_ref_dir = os.path.join(paths.get_voice_ref_path(), voice_name)
        coqui_dir = paths.get_coqui_path()
        # Base XTTS model goes in models/coqui/xtts directory (this is the standard location)
        base_xtts_dir = os.path.join(coqui_dir, "xtts")  
        # Fine-tuned voice models go in models/coqui/custom_xtts_v2 directory
        custom_model_dir = os.path.join(paths.get_model_dir(), "custom_xtts_v2")
        custom_voice_dir = os.path.join(custom_model_dir, voice_name)
        custom_xtts_voice_dir = os.path.join(custom_model_dir, f"XTTS-v2_{voice_name}")
        
        # Create directories if they don't exist
        os.makedirs(voice_ref_dir, exist_ok=True)
        os.makedirs(base_xtts_dir, exist_ok=True)
        os.makedirs(custom_model_dir, exist_ok=True)
        
        # STEP 1: Check if base Coqui XTTS v2 model exists, download if needed
        # This is the foundation model that goes in models/coqui/xtts
        log.info("Checking for base Coqui XTTS v2 model")
        if not os.path.exists(os.path.join(base_xtts_dir, "model.pth")):
            log.info("Base Coqui XTTS v2 model not found, downloading...")
            # Download the base XTTS v2 model to the base_xtts_dir directory
            model_path, model_success = HfUtils.download_voice_ref_pack(
                "coqui/XTTS-v2",  # Base XTTS v2 repository
                "xtts",
                target_type="base_model"  # Use special type for base model to target correct directory
            )
            
            if not model_success:
                log.error("Failed to download base XTTS v2 model")
                return False
            log.info("Base Coqui XTTS v2 model downloaded successfully")
        else:
            log.info("Base Coqui XTTS v2 model already exists")
            
        # STEP 2: Check if default c3po voice model exists in custom_xtts_v2 directory
        log.info("Checking for C3PO custom voice model")
        custom_model_found = False
        custom_model_dir_to_use = None
        
        for model_dir in [custom_voice_dir, custom_xtts_voice_dir]:
            if os.path.exists(model_dir):
                log.info(f"Found custom voice model directory at {model_dir}")
                custom_model_found = True
                custom_model_dir_to_use = model_dir
                break
        
        # If custom model not found, download it
        if not custom_model_found:
            log.info(f"Custom voice model not found for {voice_name}, downloading...")
            
            # Try to get repository URL from our known voice types
            voice_pack = VoiceRefPackType.get_by_name(voice_name)
            if voice_pack:
                repo_url = voice_pack.value.repo_url
                log.info(f"Found repository URL for {voice_name}: {repo_url}")
                
                # Download to custom_xtts_v2 as a full model
                model_name = f"XTTS-v2_{voice_name}" 
                model_path, model_success = HfUtils.download_voice_ref_pack(
                    repo_url, model_name, target_type="model"
                )
                
                if model_success:
                    log.info(f"Successfully downloaded custom voice model to {model_path}")
                    custom_model_found = True
                    custom_model_dir_to_use = model_path
                else:
                    log.error(f"Failed to download custom voice model from {repo_url}")
                    return False
            else:
                log.error(f"No repository URL found for voice {voice_name}")
                return False
        
        # STEP 3: Check if reference file already exists in voice_ref_pack
        reference_path = VoiceUtils.find_voice_ref_file(voice_name)
        if reference_path:
            log.info(f"Found existing voice reference file: {reference_path}")
            return True
        
        # STEP 4: Copy reference.wav from custom model to voice_ref_pack directory
        if custom_model_found and custom_model_dir_to_use:
            ref_file = None
            for ref_name in ["reference.wav", "clone_speech.wav"]:
                potential_ref = os.path.join(custom_model_dir_to_use, ref_name)
                if os.path.exists(potential_ref):
                    ref_file = potential_ref
                    log.info(f"Found reference file in custom model directory: {ref_file}")
                    break
                    
            if ref_file:
                # Copy the reference file to the voice reference directory as reference.wav
                voice_ref_path = os.path.join(voice_ref_dir, "reference.wav")
                shutil.copy2(ref_file, voice_ref_path)
                log.info(f"Copied reference file from {ref_file} to {voice_ref_path}")
                return True
            else:
                log.warning(f"No reference file found in custom model for {voice_name}")
                return False
        
        log.error(f"Could not find or download voice reference for {voice_name}")
        return False

    @staticmethod
    def find_voice_ref_pack_file(voice_name, paths=None):
        """
        Find a voice reference file for a given voice name.
        
        Args:
            voice_name (str): Name of the voice to find
            paths (Paths, optional): Paths instance to use
            
        Returns:
            str: Path to the voice reference file, or None if not found
        """
        from oarc.speech.voice.voice_utils import VoiceUtils
        
        # Get reference file path
        ref_file = VoiceUtils.find_voice_ref_file(voice_name, paths)
        if ref_file:
            return ref_file
            
        # Try to ensure it exists (will download if needed)
        if SpeechUtils.ensure_voice_reference_exists(voice_name):
            return VoiceUtils.find_voice_ref_file(voice_name, paths)
            
        return None

    @staticmethod
    def diagnose_model_config(model_path):
        """
        Diagnose issues with model config file.
        
        This utility helps troubleshoot why a model isn't being recognized 
        as a valid XTTS model by examining its config file in detail.
        
        Args:
            model_path (str): Path to the model directory
            
        Returns:
            dict: Diagnostic information about the model config
        """
        import json
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            log.error(f"Config file not found at {config_path}")
            return {"error": "Config file not found"}
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
                
            # Try to parse the JSON
            try:
                config = json.loads(config_content)
                
                # Check structure
                diagnostics = {
                    "has_model_key": 'model' in config,
                    "model_keys": list(config.keys()),
                    "model_type_path_exists": False,
                    "model_type_value": None,
                    "actual_structure": str(config)[:500] + "..." if len(str(config)) > 500 else str(config)
                }
                
                # If model key exists, check for model_type
                if 'model' in config and isinstance(config['model'], dict):
                    diagnostics["model_keys_nested"] = list(config['model'].keys())
                    diagnostics["model_type_path_exists"] = 'model_type' in config['model']
                    
                    if 'model_type' in config['model']:
                        diagnostics["model_type_value"] = config['model']['model_type']
                        diagnostics["is_xtts_model"] = 'xtts' in str(config['model']['model_type']).lower()
                        
                return diagnostics
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, check encoding and content
                return {
                    "error": f"JSON parsing error: {str(e)}",
                    "position": e.pos,
                    "line": e.lineno,
                    "column": e.colno,
                    "content_excerpt": config_content[:1000] + "..." if len(config_content) > 1000 else config_content
                }
                
        except Exception as e:
            log.error(f"Error diagnosing model config: {str(e)}", exc_info=True)
            return {"error": f"Unexpected error: {str(e)}"}