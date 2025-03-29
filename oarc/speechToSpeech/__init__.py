#__init__.py
from .textToSpeech import textToSpeech
from .speechToText import speechToText
from oarc.speechToSpeech import TextToSpeechAPI
from oarc.speechToSpeech import SpeechToTextAPI

__all__ = ['textToSpeech', 'speechToText', 'TextToSpeechAPI', 'SpeechToTextAPI']