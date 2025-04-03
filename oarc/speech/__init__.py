"""
This module initializes the speechToSpeech package by importing and exposing key 
classes related to text-to-speech and speech-to-text functionalities along with 
their associated API interfaces. The module imports classes such as TextToSpeech, 
SpeechToText, and their corresponding API classes, as well as TTSRequest for handling 
text-to-speech requests.
"""

from .speech_manager import SpeechManager
from .text_to_speech import TextToSpeech
from .text_to_speech_api import TextToSpeechAPI
from .speech_to_text import SpeechToText
from .speech_to_text_api import SpeechToTextAPI
from .tts_request import TTSRequest
from .transcription_app import TranscriptionApp
from .speech_errors import TTSInitializationError

__all__ = [
    # Classes
    'SpeechManager',
    'TextToSpeech', 
    'SpeechToText', 
    'TextToSpeechAPI', 
    'SpeechToTextAPI',
    'TTSRequest',
    'TranscriptionApp',

    # Errors
    'TTSInitializationError',
]