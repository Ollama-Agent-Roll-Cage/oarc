"""
This module serves as a central hub for various API components, including:

- `LLMPromptAPI`: Handles operations related to language model prompts.
- `AgentAPI`: Manages agent-related functionalities.
- `API`: Provides core API functionalities.
- `SpellLoader`: Facilitates loading of spell configurations.
- `AgentAccess`: Manages access control for agents.
- `ModelRequest`: Handles model request operations.

The `__all__` list defines the public interface, exposing only the specified components for import. 

TODOs:
1. Develop APIs for multimodal tools, including speech processing, vision tools, and Ollama integration.
2. Implement an API for loading wizard agent configurations from stored JSON templates.
3. Enhance the multimodal pip installation to support seamless integration of tools like text-to-speech and speech-to-text.
4. Ensure the design prioritizes maintainability and gradual development, emphasizing quality and long-term stability.

This module is a work in progress, aiming to provide a robust and extensible framework for agent and multimodal tool integration.
"""

from .llm_prompt_api import LLMPromptAPI
from .agent_api import AgentAPI
from .api import API
from .spell_loader import SpellLoader
from .agent_access import AgentAccess
from .model_request import ModelRequest

__all__ = [
    'LLMPromptAPI', 
    'AgentAPI', 
    'API', 
    'SpellLoader', 
    'AgentAccess', 
    'ModelRequest'
]

#TODO CREATE API FOR SMOL AGENTS SPEECH, OLLAMA, AND VISION TOOLS
#TODO CREATE API FOR  WIZARD AGENT CONFIG LOADING FOR STORED AGENT CORE JSON TEMPLATE CONFIGS
#TODO BUILD  pip install such that from .speechtoSpeech import textToSpeech, or speechtoText etc.
#TODO so essentially all of the tools in the multimodal pip install  package can be written into scripts, 
# or you can access the entire  api for loading agent configs, HANDLE WITH GRACE, BUILD WITH CARE, TAKE IT SLOW THIS IS A MARATHON NOT A SPRINT.
