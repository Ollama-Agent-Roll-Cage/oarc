"""
OARC Gradio Server Module

This package provides Gradio-based server implementations for the OARC project,
enabling easy creation of web interfaces for AI models and components.

Key components:
- GradioServer: Concrete implementation of the Server abstract base class using Gradio
- GradioServerAPI: API interface for Gradio servers
"""

from oarc.server.gradio.gradio_server import GradioServer
from oarc.server.gradio.gradio_server_api import GradioServerAPI

__all__ = [
    'GradioServer',
    'GradioServerAPI',
]
