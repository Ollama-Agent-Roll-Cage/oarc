"""
OARC Async Test Harness Base Class

This module provides a base class for OARC test harnesses, implementing
common test setup, teardown, and utility functions.
"""
import sys
import asyncio
from typing import Dict
from abc import ABC, abstractmethod

from oarc.utils.log import log
from oarc.utils.paths import Paths
from oarc.utils.const import SUCCESS, FAILURE

class AsyncTestHarness(ABC):
    """Base class for OARC test harnesses."""
    
    def __init__(self, test_name: str):
        """Initialize the test harness.
        
        Args:
            test_name: Name of the test for logging
        """
        self.test_name = test_name
        self.paths = Paths()  # Get singleton instance
        self.results: Dict[str, bool] = {}
    
    async def setup(self) -> bool:
        """Set up test environment.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            log.info(f"Setting up {self.test_name} test environment")
            return True
        except Exception as e:
            log.error(f"Error in test setup: {e}", exc_info=True)
            return False
    
    async def teardown(self) -> None:
        """Clean up after tests."""
        log.info(f"Cleaning up {self.test_name} test environment")
    
    @abstractmethod
    async def run_tests(self) -> bool:
        """Run the test suite.
        
        Returns:
            bool: True if all tests passed, False otherwise
        """
        pass
    
    def log_results(self) -> None:
        """Log test results summary."""
        if not self.results:
            log.warning("No test results to report")
            return
            
        log.info("=" * 50)
        log.info(f"{self.test_name} Test Results")
        log.info("=" * 50)
        
        all_passed = True
        for test_name, result in self.results.items():
            status = "PASSED" if result else "FAILED"
            log.info(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        log.info("=" * 50)
        log.info(f"Overall Status: {'PASSED' if all_passed else 'FAILED'}")
        log.info("=" * 50)
    
    async def execute(self) -> bool:
        """Execute the full test suite with setup and teardown.
        
        Returns:
            bool: True if all tests passed, False otherwise
        """
        try:
            if not await self.setup():
                log.error("Test setup failed")
                return False
            
            success = await self.run_tests()
            self.log_results()
            
            await self.teardown()
            return success
            
        except Exception as e:
            log.error(f"Error executing tests: {e}", exc_info=True)
            return False

    @classmethod
    def run(cls, harness_class: type) -> None:
        """Class method to create and run a test harness.
        
        Args:
            harness_class: The test harness class to instantiate and run
        """
        if not issubclass(harness_class, AsyncTestHarness):
            raise TypeError("Harness class must inherit from AsyncTestHarness")
            
        harness = harness_class()
        sys.exit(cls._run_harness(harness))

    @staticmethod
    def _run_harness(harness: "AsyncTestHarness") -> int:
        """Helper method to run a test harness.
        
        Args:
            harness: Test harness instance to run
            
        Returns:
            int: System exit code (SUCCESS/FAILURE)
        """
        result = asyncio.run(harness.execute())
        return SUCCESS if result else FAILURE
