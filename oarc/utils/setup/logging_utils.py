#!/usr/bin/env python3
"""
Logging utilities for OARC setup process.
"""

import sys
from datetime import datetime
from pathlib import Path

from .setup_utils import PROJECT_ROOT, LOG_DIR

def setup_logging():
    """Set up logging for the build process."""
    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"build_{timestamp}.log"
    
    # Import our logging module
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from oarc.utils.log import Log
        Log.set_default_log_dir(LOG_DIR)
        Log.add_file_handler("build", f"build_{timestamp}.log")
        logger = Log.get_logger("build", with_file=True)
        print(f"Logging to {log_file}")
        return logger
    except ImportError:
        print(f"Warning: Unable to import Log module. Logs will only be shown on console.")
        return None
