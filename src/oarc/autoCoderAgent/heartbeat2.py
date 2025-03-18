import asyncio

class Heartbeat:
    def __init__(self, interval):
        """
        Initialize the Heartbeat with a specified interval.

        Args:
        - interval (int): Time in seconds between heartbeat signals.
        """
        self.interval = interval
        self._task = None

    async def _send_heartbeat(self):
        """
        Private coroutine to send heartbeat signals.
        """
        print("💓 Heartbeat signal sent")

    async def _run_heartbeat(self):
        """
        Private coroutine to continuously send heartbeats.
        """
        while True:
            await self._send_heartbeat()
            await asyncio.sleep(self.interval)

    def start(self):
        """
        Starts the heartbeat timer.
        """
        if not self._task:
            self._task = asyncio.create_task(self._run_heartbeat())
            print("Heartbeat started!")

    def stop(self):
        """
        Stops the heartbeat timer.
        """
        if self._task:
            self._task.cancel()
            self._task = None
            print("Heartbeat stopped!")

async def main():
    hb = Heartbeat(interval=2)  # 2 seconds interval
    hb.start()
    await asyncio.sleep(10)  # Run for 10 seconds
    hb.stop()

# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())