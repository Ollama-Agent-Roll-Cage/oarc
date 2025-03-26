import asyncio
from TestAgent import TestAgent

async def main():
    # Initialize and run test agent
    agent = TestAgent()
    
    # Start Gradio interface
    agent.launch_gradio()
    
    # Keep the script running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())