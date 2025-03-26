""" test_tts.py
This script tests the TTS endpoint of the API.
to run this script, use the following command:

# Test TTS endpoint
curl -X POST "http://localhost:2020/tts/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello! I am C-3PO!", "voice_name": "c3po"}'

# List available voices
curl "http://localhost:2020/tts/voices"
"""

# test_components.py
from oarc import oarcAPI
from oarc.speechToSpeech import textToSpeech, speechToText

def test_tts():
    # Initialize TTS
    developer_tools_dict = {
        'current_dir': os.getcwd(),
        'parent_dir': os.path.dirname(os.getcwd()),
        'speech_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'coqui'),
        'recognize_speech_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'whisper'),
        'generate_speech_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'generated'),
        'tts_voice_ref_wav_pack_path_dir': os.path.join(os.getenv('OARC_MODEL_GIT'), 'coqui', 'voice_reference_pack')
    }
    
    tts = textToSpeech(
        developer_tools_dict=developer_tools_dict,
        voice_type="xtts_v2",
        voice_name="c3po"
    )
    
    # Test speech generation
    audio = tts.process_tts_responses("Hello! I am C-3PO, human-cyborg relations!", "c3po")
    print("TTS test complete")

def test_stt():
    stt = speechToText()
    print("Press Ctrl+Shift to start recording...")
    stt.hotkeyRecognitionLoop()

if __name__ == "__main__":
    test_tts()
    test_stt()