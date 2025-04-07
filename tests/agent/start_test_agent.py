"""
This script initializes a TestAgent, launches its Gradio interface, and maintains an active event loop for continuous interaction. The infinite loop is designed to keep the application running until a KeyboardInterrupt is received, at which point the script performs a graceful shutdown. This setup is primarily used for testing and development purposes.
"""

import asyncio

from tests.agent.test_agent import TestAgent

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