"""
OARC Tests Package

This initialization file ensures that the project root directory
is added to the Python path, allowing tests to import OARC modules
without path manipulation in each test file.
"""

# Add project root to Python path
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import test modules
from tests import start_test_agent
from tests import test_agent
from tests import test_yolo
from tests import test_oarc
from tests import test_tts
from tests import test_tts_fast
from tests import test_stt
from tests import test_stt_fast

__all__ = [
    "test_agent",
    "test_tts",
    "start_test_agent",
    "test_yolo",
    "test_stt_fast",
    "test_oarc",
    "test_stt",
    "test_stt_fast",
    "test_tts_fast",
]