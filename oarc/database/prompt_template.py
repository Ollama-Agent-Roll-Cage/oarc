"""
Prompt template utilities for OARC.

This module provides simple template functionality for formatting prompts
with variable substitution.
"""

import logging
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class PromptTemplate:
    """A lightweight template class for dynamically formatting prompt strings.

    This class simplifies the process of inserting variable data into prompts by utilizing
    Python's string formatting mechanism. It is similar in spirit to f-strings, allowing for
    clear and concise variable substitution while providing custom error handling and logging.
    """
    

    def __init__(self, template_str: str):
        """Initialize the PromptTemplate instance with a template string.

        This constructor stores a template string that includes placeholders in
        the {variable_name} format, enabling dynamic substitution of variables
        when formatting the prompt.

        Args:
            template_str (str): The template string to be used for prompt formatting.
        """
        self.template_str = template_str
        log.info("Prompt template initialized")
    

    def format(self, **kwargs: Any) -> str:
        """
        Format the stored template string by substituting the provided keyword arguments.

        This method uses Python's built-in string formatting to replace placeholders in the 
        template with corresponding values. If a required variable is missing, the method
        logs the error and returns the original template string.

        Args:
            **kwargs: Keyword arguments mapping template placeholders to their substitution values.

        Returns:
            str: The resulting string after successful formatting, or the original template string 
             if an error occurs.
        """
        try:
            formatted = self.template_str.format(**kwargs)
            return formatted
        except KeyError as e:
            log.error(f"Missing required template variable: {e}")
            return self.template_str
        except Exception as e:
            log.error(f"Error formatting template: {e}")
            return self.template_str


    def __str__(self) -> str:
        """
        Return a detailed string representation of the template instance.

        This method retrieves and returns the stored template string,
        providing a human-readable representation of the instance.

        Returns:
            str: The underlying template string.
        """
        """Get a string representation of the template."""
        return self.template_str
    

    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the PromptTemplate instance.

        This method outputs the instance in a format that includes the template string,
        making it easier to debug or log the internal state of the template.
        """
        """Get a developer representation of the template."""
        return f"PromptTemplate({repr(self.template_str)})"
    

    def __call__(self, **kwargs: Any) -> str:
        """
        Allow invoking the instance directly to format the template.

        This enables calling the PromptTemplate object as if it were a function, passing in
        keyword arguments for variable substitution. The stored template string will then have
        its placeholders replaced with the provided values.

        Args:
            **kwargs: Key-value pairs used to replace placeholders in the template.
            
        Returns:
            str: The formatted string with all provided variables substituted.
        """
        return self.format(**kwargs)
