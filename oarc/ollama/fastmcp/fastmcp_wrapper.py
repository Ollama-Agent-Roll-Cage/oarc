"""
FastMCPWrapper - A simple wrapper for FastMCP
This class provides an easy way to create and use FastMCP servers and clients
anywhere in your application.
"""
import asyncio
from typing import Callable, Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

# Import FastMCP
try:
    from fastmcp import FastMCP, Client
except ImportError:
    raise ImportError("FastMCP is not installed. Please install it with 'uv pip install fastmcp'")

class FastMCPWrapper:
    """
    A wrapper class for FastMCP to easily create and use MCP servers and clients.
    """

    def __init__(self, name: str = "MCPWrapper", dependencies: List[str] = None):
        """
        Initialize the FastMCPWrapper with a name and optional dependencies.
        
        Args:
            name: Name of the MCP server
            dependencies: List of dependencies needed when deployed via `fastmcp install`
        """
        self.name = name
        self.dependencies = dependencies or []
        self.mcp = FastMCP(name, dependencies=self.dependencies)
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        
    def add_tool(self, func: Callable = None, **kwargs) -> Callable:
        """
        Add a function as a tool to the MCP server.
        This can be used as a decorator or a method.
        
        Args:
            func: The function to add as a tool
            **kwargs: Additional keyword arguments to pass to mcp.tool()
            
        Returns:
            The decorated function
        """
        if func is None:
            return lambda f: self.add_tool(f, **kwargs)
        
        decorated = self.mcp.tool(**kwargs)(func)
        self.tools[func.__name__] = decorated
        return decorated
    
    def add_resource(self, uri: str, func: Callable = None, **kwargs) -> Callable:
        """
        Add a function as a resource to the MCP server.
        This can be used as a decorator or a method.
        
        Args:
            uri: The URI for the resource
            func: The function to add as a resource
            **kwargs: Additional keyword arguments to pass to mcp.resource()
            
        Returns:
            The decorated function
        """
        if func is None:
            return lambda f: self.add_resource(uri, f, **kwargs)
        
        decorated = self.mcp.resource(uri, **kwargs)(func)
        self.resources[uri] = decorated
        return decorated
    
    def add_prompt(self, func: Callable = None, **kwargs) -> Callable:
        """
        Add a function as a prompt to the MCP server.
        This can be used as a decorator or a method.
        
        Args:
            func: The function to add as a prompt
            **kwargs: Additional keyword arguments to pass to mcp.prompt()
            
        Returns:
            The decorated function
        """
        if func is None:
            return lambda f: self.add_prompt(f, **kwargs)
        
        decorated = self.mcp.prompt(**kwargs)(func)
        self.prompts[func.__name__] = decorated
        return decorated
    
    def run(self, transport: str = None, **kwargs):
        """
        Run the MCP server.
        
        Args:
            transport: The transport method to use (e.g., 'sse', 'ws')
            **kwargs: Additional keyword arguments to pass to mcp.run()
        """
        return self.mcp.run(transport=transport, **kwargs)
    
    def install(self, script_path: str = None, name: str = None, with_deps: List[str] = None):
        """
        Install the MCP server for use with Claude Desktop.
        This is a helper method that executes the CLI command.
        
        Args:
            script_path: The path to the script file
            name: Custom name for the server in Claude
            with_deps: Additional dependencies to install
        """
        import subprocess
        import sys
        import tempfile
        
        if script_path is None:
            # Create a temporary script file
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                script_path = temp.name
                with open(script_path, 'w') as f:
                    f.write(f"""
from fastmcp import FastMCP

mcp = FastMCP("{self.name or 'MCPWrapper'}")

# Add tools, resources, and prompts from the wrapper
{self._generate_tool_code()}
{self._generate_resource_code()}
{self._generate_prompt_code()}

if __name__ == "__main__":
    mcp.run()
""")
        
        cmd = ["fastmcp", "install", script_path]
        
        if name:
            cmd.extend(["--name", name])
            
        if with_deps:
            for dep in with_deps:
                cmd.extend(["--with", dep])
                
        # Add dependencies from init
        for dep in self.dependencies:
            cmd.extend(["--with", dep])
            
        subprocess.run(cmd, check=True)
        
    def _generate_tool_code(self) -> str:
        """Generate code for tools to be included in the temporary script."""
        tool_code = []
        for name, func in self.tools.items():
            # Extract the original function definition
            import inspect
            source = inspect.getsource(func.__wrapped__)
            # Replace the first line with a decorated version
            lines = source.split('\n')
            first_line = lines[0]
            indentation = len(first_line) - len(first_line.lstrip())
            decorator = ' ' * indentation + '@mcp.tool()'
            lines.insert(0, decorator)
            tool_code.append('\n'.join(lines))
        return '\n\n'.join(tool_code)

    def _generate_resource_code(self) -> str:
        """Generate code for resources to be included in the temporary script."""
        resource_code = []
        for uri, func in self.resources.items():
            import inspect
            source = inspect.getsource(func.__wrapped__)
            lines = source.split('\n')
            first_line = lines[0]
            indentation = len(first_line) - len(first_line.lstrip())
            decorator = f'{" " * indentation}@mcp.resource("{uri}")'
            lines.insert(0, decorator)
            resource_code.append('\n'.join(lines))
        return '\n\n'.join(resource_code)
        
    def _generate_prompt_code(self) -> str:
        """Generate code for prompts to be included in the temporary script."""
        prompt_code = []
        for name, func in self.prompts.items():
            import inspect
            source = inspect.getsource(func.__wrapped__)
            lines = source.split('\n')
            first_line = lines[0]
            indentation = len(first_line) - len(first_line.lstrip())
            decorator = ' ' * indentation + '@mcp.prompt()'
            lines.insert(0, decorator)
            prompt_code.append('\n'.join(lines))
        return '\n\n'.join(prompt_code)
        
    @asynccontextmanager
    async def client_session(self, transport=None, sampling_handler=None, **kwargs):
        """
        Create a client session to interact with the MCP server.
        
        Args:
            transport: The transport method to use
            sampling_handler: Handler for LLM sampling requests
            **kwargs: Additional keyword arguments to pass to Client constructor
            
        Yields:
            An MCP Client instance
        """
        async with Client(self.mcp, transport=transport, sampling_handler=sampling_handler, **kwargs) as client:
            yield client
            
    async def call_tool(self, tool_name: str, params: Dict[str, Any], transport=None, **kwargs):
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
            transport: The transport method to use
            **kwargs: Additional keyword arguments to pass to Client constructor
            
        Returns:
            The result of the tool call
        """
        async with self.client_session(transport=transport, **kwargs) as client:
            return await client.call_tool(tool_name, params)
            
    async def read_resource(self, uri: str, transport=None, **kwargs):
        """
        Read a resource from the MCP server.
        
        Args:
            uri: URI of the resource to read
            transport: The transport method to use
            **kwargs: Additional keyword arguments to pass to Client constructor
            
        Returns:
            The content of the resource
        """
        async with self.client_session(transport=transport, **kwargs) as client:
            return await client.read_resource(uri)
            
    async def get_prompt(self, prompt_name: str, params: Dict[str, Any] = None, transport=None, **kwargs):
        """
        Get a prompt from the MCP server.
        
        Args:
            prompt_name: Name of the prompt to get
            params: Parameters to pass to the prompt
            transport: The transport method to use
            **kwargs: Additional keyword arguments to pass to Client constructor
            
        Returns:
            The content of the prompt
        """
        async with self.client_session(transport=transport, **kwargs) as client:
            return await client.get_prompt(prompt_name, params or {})
            
    def mount(self, prefix: str, other_wrapper):
        """
        Mount another FastMCPWrapper onto this one with a prefix.
        
        Args:
            prefix: The prefix to use for the mounted wrapper
            other_wrapper: Another FastMCPWrapper instance to mount
        """
        self.mcp.mount(prefix, other_wrapper.mcp)
        return self
