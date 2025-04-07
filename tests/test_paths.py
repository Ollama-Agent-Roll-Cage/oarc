"""Basic test for path functionality in OARC."""

import os
import sys

from oarc.utils.log import log
from oarc.utils.const import SUCCESS, FAILURE
from oarc.utils.paths import Paths

TEST_VOICE_NAME = "C3PO"
TEST_OUTPUT_FILE_NAME = "test_output_C3PO.wav"

def main():
    """Test basic path functionality using the custom C3PO voice."""
    log.info("Starting basic path test with C3PO voice")
    
    # Get paths using the OARC utility singleton
    paths = Paths()
    
    # Use the test output directory API
    output_dir = paths.get_test_output_dir()
    output_file = os.path.join(output_dir, TEST_OUTPUT_FILE_NAME)
    
    try:
        # Get paths using the OARC utility singleton
        paths.log_paths()
        return True
    
    except Exception as e:
        log.error(f"Error in path test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    result = main()
    sys.exit(SUCCESS if result else FAILURE)