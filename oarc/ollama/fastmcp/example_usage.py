"""
Example usage of FastMCPWrapper
"""
import asyncio
from FastMCPWrapper import FastMCPWrapper

# Create a FastMCPWrapper instance
mcp_wrapper = FastMCPWrapper("MyAwesomeApp", dependencies=["pandas", "numpy"])

# Add a tool using the decorator syntax
@mcp_wrapper.add_tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a tool using the method syntax
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

mcp_wrapper.add_tool(multiply)

# Add a resource
@mcp_wrapper.add_resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Add a prompt
@mcp_wrapper.add_prompt
def code_review(code_snippet: str) -> str:
    """Generate a standard code review request."""
    return f"Please review the following code snippet for potential bugs and style issues:\n```python\n{code_snippet}\n```"

# Example of using the client to call tools
async def main():
    # Call a tool
    result = await mcp_wrapper.call_tool("add", {"a": 5, "b": 3})
    print(f"5 + 3 = {result.content[0].text}")
    
    # Read a resource
    greeting = await mcp_wrapper.read_resource("greeting://world")
    print(f"Greeting: {greeting[0].content}")
    
    # Get a prompt
    review_request = await mcp_wrapper.get_prompt("code_review", {"code_snippet": "print('Hello, world!')"})
    print(f"Review request: {review_request}")

if __name__ == "__main__":
    # Option 1: Run the server
    # mcp_wrapper.run()
    
    # Option 2: Install the server for Claude Desktop
    # mcp_wrapper.install(name="MyAwesomeApp")
    
    # Option 3: Use the client to interact with the server
    asyncio.run(main())
