"""
Prompt template utilities for OARC.

This module provides simple template functionality for formatting prompts
with variable substitution.
"""

from typing import Any
from oarc.decorators.log import log


@log()
class PromptTemplate:
    """A simple template class for formatting prompts with variable substitution.
    
    This class provides basic functionality similar to f-strings but with
    dynamic variable injection from keyword arguments.
    """
    
    def __init__(self, template_str: str):
        """Initialize the prompt template with a template string.
        
        Args:
            template_str: String template with placeholders in {variable_name} format
        """
        self.template_str = template_str
        log.info("Prompt template initialized")
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with the provided keyword arguments.
        
        Args:
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            str: The formatted string with variables replaced
        """
        try:
            # Use the built-in string format method to replace variables
            formatted = self.template_str.format(**kwargs)
            return formatted
        except KeyError as e:
            log.error(f"Missing required template variable: {e}")
            return self.template_str
        except Exception as e:
            log.error(f"Error formatting template: {e}")
            return self.template_str
            
    def __str__(self) -> str:
        """Get a string representation of the template."""
        return self.template_str
    
    def __repr__(self) -> str:
        """Get a developer representation of the template."""
        return f"PromptTemplate({repr(self.template_str)})"
    
    def __call__(self, **kwargs: Any) -> str:
        """Allow calling the template directly to format it.
        
        Args:
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            str: The formatted string with variables replaced
        """
        return self.format(**kwargs)
