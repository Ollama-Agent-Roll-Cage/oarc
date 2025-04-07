from tests.async_harness import AsyncTestHarness
from oarc.utils.log import log

class Runner(AsyncTestHarness):
    """Generic runner to execute multiple AsyncTestHarness classes."""

    def __init__(self, harness_classes):
        super().__init__("OARC Master Runner")
        self.harness_classes = harness_classes
        # Store references to child harnesses
        self.harnesses = []

    async def run_tests(self) -> bool:
        overall_success = True
        for harness_class in self.harness_classes:
            harness = harness_class()
            self.harnesses.append(harness)
            success = await harness.execute()
            # Store the result but don't log it yet
            self.results[harness.test_name] = success
            if not success:
                overall_success = False
        return overall_success

    def log_results(self) -> None:
        """Override log_results to prevent duplicate logging from child harnesses."""
        # Child harnesses have already logged their individual results
        # We'll only log the final summary
        log.info("=" * 70)
        log.info("FINAL TEST SUMMARY".center(70))
        log.info("=" * 70)
        for test_name, result in self.results.items():
            status = "PASSED" if result else "FAILED"
            log.info(f"{test_name}: {status}")
        log.info("=" * 70)
