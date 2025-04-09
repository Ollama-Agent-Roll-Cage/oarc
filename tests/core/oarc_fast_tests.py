"""
Fast test for OARC basic functionality.
This test verifies the project structure and basic logging.
"""

from typing import Dict
from pathlib import Path
import sys
import os

# Add the project root to the path to make imports work when running directly
# Adjusted to account for being in a subdirectory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from oarc.utils.log import log
from oarc.utils.paths import Paths
# Updated import to reflect the module path from project root
from tests.async_harness import AsyncTestHarness

class OARCFastTests(AsyncTestHarness):
    """Fast test implementation using base harness."""

    def __init__(self):
        """Initialize the fast test harness."""
        super().__init__("OARC Fast")
        
    async def run_tests(self) -> bool:
        """Run the fast test suite."""
        try:
            # Initialize paths singleton and log all paths for debugging
            paths = Paths()  # Get singleton instance
            paths.log_paths()  # This will show the full path configuration
            
            # Use proper path APIs to get and verify essential directories
            path_checks: Dict[str, Path] = {
                "Project Root": Path(paths.get_project_root()),
                "Models": Path(paths.get_model_dir()),
                "Coqui": Path(paths.get_coqui_path()),
                "Custom Coqui": Path(paths.get_custom_coqui_dir()),
                "Voice Reference": Path(paths.get_voice_ref_path()),
                "Test Output": Path(paths.get_test_output_dir()),
                "HF Cache": Path(paths.get_hf_cache_dir()),
                "Ollama Models": Path(paths.get_ollama_models_dir()),
                "Spells": Path(paths.get_spell_path()),
                "Whisper": Path(paths.get_whisper_dir()),
                "Generated": Path(paths.get_generated_dir()),
            }
            
            # Check each path and log status
            failed_paths = []
            for name, path in path_checks.items():
                exists = path.exists()
                status = "exists" if exists else "missing"
                log.info(f"{name}: {path} ({status})")
                
                # Store individual test results
                self.results[f"Check {name}"] = exists
                
                if not exists:
                    failed_paths.append(name)
            
            # Final status
            if failed_paths:
                log.error(f"The following paths are missing: {', '.join(failed_paths)}")
                return False
            
            log.info("Path verification completed successfully")
            return True
            
        except Exception as e:
            log.error(f"Error in path verification: {e}", exc_info=True)
            return False

# Use the async harness runner
if __name__ == "__main__":
    AsyncTestHarness.run(OARCFastTests)
