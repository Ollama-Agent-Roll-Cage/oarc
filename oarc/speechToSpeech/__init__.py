#__init__.py
from .textToSpeech import textToSpeech
from .textToSpeech import TextToSpeechAPI
from .speechToText import speechToText
from .speechToText import SpeechToTextAPI
# from .speechToSpeech import SpeechToSpeechAPI

__all__ = ['textToSpeech', 'speechToText', 'TextToSpeechAPI', 'SpeechToTextAPI']