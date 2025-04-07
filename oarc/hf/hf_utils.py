"""Utility functions for HuggingFace repository URLs."""

import re
import os
import shutil
from huggingface_hub import snapshot_download

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.const import HF_URL
from oarc.speech.voice.voice_utils import VoiceUtils

class HfUtils:
    """
    A collection of utility functions for interacting with HuggingFace repositories.
    These functions assist in extracting repository IDs from URLs and downloading
    voice packs."""

    @staticmethod
    def extract_repo_id_from_hf_url(url):
        """
        Extract the repository ID from a HuggingFace repository URL or return the repo ID directly.
        
        This function handles various URL formats from HuggingFace:
        - https://huggingface.co/username/repo-name/tree/main
        - https://hf.co/username/repo-name
        - username/repo-name (direct repository ID)
        
        Args:
            url (str): The HuggingFace repository URL or direct repository ID
            
        Returns:
            str: The extracted repository ID in the format "username/repo-name"
        """
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        
        # If the input is already in the format "username/repo-name" without a URL prefix,
        # assume it's a direct repository ID
        if re.match(r'^[^/]+/[^/]+$', url) and '://' not in url:
            log.info(f"Using direct repository ID: {url}")
            return url
            
        # Check if it's a HuggingFace URL (either huggingface.co or hf.co)
        if "huggingface.co" in url:
            # Standard domain
            repo_id = url.replace("https://huggingface.co/", "")
        elif "hf.co" in url:
            # Short domain
            repo_id = url.replace("https://hf.co/", "")
        else:
            # If not recognized as a URL but contains a slash, try to use it as-is
            if "/" in url and not url.startswith("http"):
                log.warning(f"URL format not recognized, trying to use as direct repo ID: {url}")
                return url
            else:
                raise ValueError(f"Unrecognized HuggingFace URL format: {url}")
        
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
    def download_voice_ref_pack(url_or_repo_id, voice_name=None, target_type="reference"):
        """
        Download a voice pack or custom model from HuggingFace.
        
        Args:
            url_or_repo_id (str): The HuggingFace repository URL or direct repository ID
            voice_name (str, optional): Name to save the voice pack as
            target_type (str, optional): Type of download: "reference" for voice reference samples,
                                        "model" for fine-tuned models. Defaults to "reference".
            
        Returns:
            tuple: (path to downloaded content, success status)
        """
        try:
            # Validate the input
            if not url_or_repo_id or not isinstance(url_or_repo_id, str):
                raise ValueError("URL or repository ID must be a non-empty string")
            
            # Extract repo ID or use directly if already in correct format
            try:
                repo_id = HfUtils.extract_repo_id_from_hf_url(url_or_repo_id)
            except ValueError as e:
                # If URL extraction fails but looks like a repo ID, try using it directly
                if "/" in url_or_repo_id and not url_or_repo_id.startswith("http"):
                    log.warning(f"URL extraction failed, trying direct repo ID: {url_or_repo_id}")
                    repo_id = url_or_repo_id
                else:
                    raise e
            
            # Process voice name using VoiceUtils directly
            voice_name = VoiceUtils.normalize_voice_name(repo_id, voice_name)
                
            # Get the paths
            paths = Paths()
            
            # Determine target directory based on target_type
            if target_type.lower() == "reference":
                # For voice reference samples (small WAV files)
                target_dir = paths.get_voice_ref_path()
                log.info(f"Downloading voice reference pack from {repo_id} as {voice_name}")
            elif target_type.lower() == "model":
                # For fine-tuned models (full model files)
                target_dir = os.path.join(paths.get_model_dir(), "custom_xtts_v2")
                log.info(f"Downloading fine-tuned model from {repo_id} as {voice_name}")
            else:
                raise ValueError(f"Unknown target type: {target_type}")
                
            # Create parent directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Create voice/model directory
            voice_path = os.path.join(target_dir, voice_name)
            os.makedirs(voice_path, exist_ok=True)
            log.debug(f"Created directory at: {voice_path}")
            
            # Download the repository files directly using huggingface_hub
            log.info(f"Starting HuggingFace Hub download for {repo_id} to {voice_path}")
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=voice_path,
                local_dir_use_symlinks=False,  # Full download, no symlinks
                repo_type="model"  # Explicitly specify this is a model repo
            )
            
            # List the downloaded files to help with debugging
            files = os.listdir(downloaded_path)
            log.info(f"Downloaded to {downloaded_path} with {len(files)} files: {', '.join(files)}")
            
            # Copy files from downloaded_path to voice_dir if they're different
            if downloaded_path != voice_path:
                for filename in files:
                    source_file = os.path.join(downloaded_path, filename)
                    target_file = os.path.join(voice_path, filename)
                    if os.path.isfile(source_file) and not os.path.exists(target_file):
                        shutil.copy2(source_file, target_file)
                log.info(f"Copied files from {downloaded_path} to {voice_path}")
            
            # Verify the downloaded content using the appropriate method in VoiceUtils
            if target_type.lower() == "reference":
                # Verify voice reference pack
                if VoiceUtils.verify_voice_pack(voice_path):
                    # Ensure clone_speech.wav exists for compatibility
                    VoiceUtils.ensure_clone_speech_exists(voice_path)
                    log.info(f"Voice reference pack {voice_name} successfully downloaded and verified")
                    return voice_path, True
                else:
                    log.error(f"Downloaded voice reference pack {voice_name} failed verification")
                    return voice_path, False
            else:
                # Verify model pack
                if VoiceUtils.verify_model_pack(voice_path):
                    log.info(f"Model {voice_name} successfully downloaded and verified")
                    return voice_path, True
                else:
                    log.error(f"Downloaded model {voice_name} failed verification")
                    return voice_path, False
                
        except Exception as e:
            log.error(f"Error downloading from {url_or_repo_id}: {e}", exc_info=True)
            return None, False