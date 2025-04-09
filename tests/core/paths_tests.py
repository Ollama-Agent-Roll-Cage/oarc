"""Basic test for path functionality in OARC."""

import os
import sys

# Add the project root to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from oarc.utils.log import log
from oarc.utils.const import SUCCESS, FAILURE
from oarc.utils.paths import Paths

from tests.async_harness import AsyncTestHarness

# Constants
TEST_VOICE_NAME = "C3PO"
TEST_OUTPUT_FILE_NAME = "test_output_C3PO.wav"

class PathsAsyncTests(AsyncTestHarness):
    """Async test implementation for Paths functionality."""
    
    def __init__(self):
        """Initialize the Paths async test harness."""
        super().__init__("Paths")
    
    async def setup(self) -> bool:
        """Set up test environment."""
        try:
            log.info(f"Setting up {self.test_name} test environment")
            # Initialize Paths utility - already done in parent class
            return True
        except Exception as e:
            log.error(f"Error in test setup: {e}", exc_info=True)
            return False
    
    async def test_path_functionality(self) -> bool:
        """Test basic path functionality."""
        try:
            log.info("Testing basic path functionality with C3PO voice")
            
            # Use the test output directory API
            output_dir = self.paths.get_test_output_dir()
            output_file = os.path.join(output_dir, TEST_OUTPUT_FILE_NAME)
            
            # Log all paths for verification
            self.paths.log_paths()
            
            return True
        
        except Exception as e:
            log.error(f"Error in path test: {str(e)}", exc_info=True)
            return False
    
    async def run_tests(self) -> bool:
        """Run the Paths test suite."""
        try:
            # Test path functionality
            self.results["Path Functionality"] = await self.test_path_functionality()
            
            # Return overall success
            return all(self.results.values())
            
        except Exception as e:
            log.error(f"Error running tests: {e}", exc_info=True)
            return False

# Use the async harness runner
if __name__ == "__main__":
    AsyncTestHarness.run(PathsAsyncTests)