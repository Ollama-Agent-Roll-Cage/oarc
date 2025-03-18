import asyncio

async def heartbeat(interval: int):
    """
    Asynchronous heartbeat method.

    Args:
    - interval (int): Time in seconds between heartbeat signals.

    Sends a heartbeat signal at the specified interval.
    """
    while True:
        print("💓 Heartbeat signal sent")
        await asyncio.sleep(interval)

async def main():
    # Start the heartbeat coroutine with a 2-second interval
    await asyncio.gather(heartbeat(2))

# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())