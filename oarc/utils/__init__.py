"""Utility packages for OARC.

This package contains various utility modules and subpackages 
used throughout the OARC project.
"""

from .log import get_logger, log
from .paths import Paths
from oarc.utils import setup
from oarc.utils.const import (
    SUCCESS,
    FAILURE,
    DEFAULT_MODELS_DIR,
    HUGGINGFACE_DIR,
    OLLAMA_MODELS_DIR,
    SPELLS_DIR,
    COQUI_DIR,
    CUSTOM_COQUI_DIR,
    WHISPER_DIR,
    GENERATED_DIR,
    VOICE_REFERENCE_DIR,
    YOLO_DIR,
    OUTPUT_DIR,
    TTS_MODEL_SUBDIR,
    WHISPER_MODEL_SUBDIR,
    LLAVA_MODEL_SUBDIR,
    EMBEDDINGS_SUBDIR,
    HF_URL,
)

__all__ = [
    # Functions
    'get_logger',
    'log',
    'Paths',
    'setup',

    # Constants
    'SUCCESS',
    'FAILURE',
    'DEFAULT_MODELS_DIR',
    'HUGGINGFACE_DIR',
    'OLLAMA_MODELS_DIR',
    'SPELLS_DIR',
    'COQUI_DIR',
    'CUSTOM_COQUI_DIR',
    'WHISPER_DIR',
    'GENERATED_DIR',
    'VOICE_REFERENCE_DIR',
    'YOLO_DIR',
    'OUTPUT_DIR',
    'TTS_MODEL_SUBDIR',
    'WHISPER_MODEL_SUBDIR',
    'LLAVA_MODEL_SUBDIR',
    'EMBEDDINGS_SUBDIR',
    'HF_URL',
]
