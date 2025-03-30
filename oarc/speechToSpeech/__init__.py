#__init__.py
from .textToSpeech import textToSpeech
from .speechToText import speechToText
from .speechToSpeech import SpeechToSpeechAPI

__all__ = ['textToSpeech', 'speechToText', 'SpeechToSpeechAPI']