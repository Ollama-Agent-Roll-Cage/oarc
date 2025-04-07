"""
Basic test for Coqui TTS functionality.
This test uses the TTS API directly with the OARC path management system.
"""

import os
import torch
import sys
import asyncio
import sys
from pathlib import Path

# Now import from oarc
from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.const import (
    SUCCESS, 
    FAILURE
)


from oarc.hf.hf_utils import HfUtils
from oarc.speech.voice.voice_utils import VoiceUtils
from oarc.utils.const import (
    HF_VOICE_REF_PACK_C3PO, 
    SUCCESS,
    FAILURE
)
from oarc.utils.log import log
from oarc.utils.paths import Paths

TEST_VOICE_NAME = "C3PO"
TEST_OUTPUT_FILE_NAME = "test_output_C3PO.wav"

def main():
    """Test basic TTS functionality using the custom C3PO voice."""
    log.info("Starting basic TTS test with C3PO voice")
    
    # Get paths using the OARC utility singleton
    paths = Paths()
    
    # Use the test output directory API
    output_dir = paths.get_test_output_dir()
    output_file = os.path.join(output_dir, TEST_OUTPUT_FILE_NAME)
    
    try:
        # Get paths using the OARC utility singleton
        paths.log_paths()
        
        log.info(f"Speech generated successfully and saved to {output_file}")
        return True
    
    except Exception as e:
        log.error(f"Error in TTS test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    result = main()
    sys.exit(SUCCESS if result else FAILURE)