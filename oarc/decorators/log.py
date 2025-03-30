"""Log decorator module.

This module provides decorators for logging function calls and returns.
"""

import functools
import inspect
import logging
import sys
from typing import Any, Callable, Optional, TypeVar, Type

T = TypeVar("T")

# Configure basic logging format
LF = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class LogDecorator:
    """Class for providing log decorators."""

    _level = logging.INFO
    _instance = None

    @staticmethod
    def _get_name(caller_depth=2) -> str:
        """Get the calling module and class name for logging context."""
        frame = inspect.stack()[caller_depth]
        module = inspect.getmodule(frame[0])
        module_name = module.__name__ if module else "oarc"
        class_name = frame[3]
        return f"{module_name}.{class_name}"

    @staticmethod
    def _setup() -> None:
        """Initialize logging system with basic configuration."""
        logging.basicConfig(level=LogDecorator._level,
                            format=LF,
                            datefmt="%Y.%m.%d:%H:%M:%S",
                            handlers=[logging.StreamHandler(sys.stdout)])
        LogDecorator._instance = logging.getLogger("oarc")
        LogDecorator._instance.setLevel(LogDecorator._level)

    @staticmethod
    def get() -> logging.Logger:
        """Get or create the singleton logger instance."""
        if LogDecorator._instance is None:
            LogDecorator._setup()
        return LogDecorator._instance

    @staticmethod
    def get_context_logger(context_name: str) -> logging.Logger:
        """Get a logger with specific context name."""
        # Ensure base logger is initialized
        if LogDecorator._instance is None:
            LogDecorator._setup()
        
        # Create a logger with the provided context
        logger = logging.getLogger(context_name)
        logger.setLevel(LogDecorator._level)
        return logger

    @staticmethod
    def set_level(level: int) -> None:
        """Set the global logging level."""
        LogDecorator._level = level
        if LogDecorator._instance:
            LogDecorator._instance.setLevel(level)


# Create a module-level logger that files can import directly
module_logger = LogDecorator.get_context_logger("oarc")


def log(level: Optional[int] = None):
    """Decorator to provide a logger instance to a function."""
    def decorate_function(func: Callable[..., T]) -> Callable[..., T]:
        # Get a context-specific logger for this function
        func_name = f"{func.__module__}.{func.__name__}"
        logger = LogDecorator.get_context_logger(func_name)
        
        # Set logging level if specified
        if level is not None:
            logger.setLevel(level)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Make logger available in global namespace if possible
            if hasattr(func, '__globals__'):
                # Inject the logger
                func.__globals__['log'] = logger
            
            # Call the function with logger available
            return func(*args, **kwargs)
                
        return wrapper
    
    # For class decoration, we'll simply return the class unchanged
    # and advise users to use the module logger instead
    def decorator(obj):
        if isinstance(obj, type):  # It's a class
            # Just add a class attribute that points to a logger
            obj._logger = LogDecorator.get_context_logger(f"{obj.__module__}.{obj.__name__}")
            return obj
        else:  # It's a function
            return decorate_function(obj)
        
    return decorator
