"""
OARC Test Runner

This script runs all available OARC test harnesses in sequence.
It uses the AsyncTestHarness framework to execute tests and report results.
"""

import sys
import os

# Add the project root to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from oarc.utils.log import log
from tests.runner import Runner
from tests.async_harness import AsyncTestHarness

# Import test harnesses from their respective directories
from tests.core.oarc_fast_tests import OARCFastTests
from tests.speech.tts_fast_tests import TTSFastTests
from tests.core.oarc_tests import OARCTests

# Remove the previous AllTests class
# Replace it with a new class that inherits from Runner
class AllTests(Runner):
    def __init__(self):
        super().__init__([
            OARCFastTests,
            TTSFastTests,
            OARCTests,
            # Add more test harnesses here as needed
        ])

from oarc.utils.const import SUCCESS, FAILURE

# Keep the main function for potential future use, but don't use it in __main__
async def main():
    # Create an instance of AllTests (which extends Runner)
    runner = AllTests()
    success = await runner.execute()
    return SUCCESS if success else FAILURE

# Use the run method correctly - it expects a class, not a function call
if __name__ == "__main__":
    # Pass the class itself, not the result of main()
    AsyncTestHarness.run(AllTests)
