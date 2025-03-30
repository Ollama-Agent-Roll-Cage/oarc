"""Log decorator module.

This module provides decorators for logging function calls and returns.
"""

import functools
import inspect
import logging
import sys
from typing import Any, Callable, Optional, TypeVar, Union, Type

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
        """Get a logger with specific context name.
        
        Args:
            context_name (str): The context name for the logger
            
        Returns:
            logging.Logger: A logger instance with the specified context
        """
        # Ensure base logger is initialized
        if LogDecorator._instance is None:
            LogDecorator._setup()
        
        # Create a logger with the provided context
        logger = logging.getLogger(context_name)
        logger.setLevel(LogDecorator._level)
        return logger

    @staticmethod
    def set_level(level: int) -> None:
        """Set the global logging level.

        Args:
            level (int): Logging level to set
        """
        LogDecorator._level = level
        if LogDecorator._instance:
            LogDecorator._instance.setLevel(level)


def log(level: Optional[int] = None):
    """Decorator to provide a logger instance to a function or class.
    
    This decorator can be applied to:
    1. Functions/methods - injects a logger instance named 'log' into the function's scope
    2. Classes - applies the decorator to all methods in the class automatically
    
    Args:
        level (Optional[int]): Logging level to use for this logger
    
    Returns:
        Callable: The decorator function
    """
    def is_class_decorator(obj):
        """Determine if the decorated object is a class."""
        return isinstance(obj, type)
    
    def decorate_function(func: Callable[..., T]) -> Callable[..., T]:
        """Apply the log decorator to a function or method."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a context-specific logger for this function
            func_name = f"{func.__module__}.{func.__name__}"
            logger = LogDecorator.get_context_logger(func_name)
            
            # Set logging level if specified
            if level is not None:
                logger.setLevel(level)
            
            # Make logger available as 'log' in the function scope
            func.__globals__['log'] = logger
            
            try:
                # Call the function (now with 'log' available in its scope)
                result = func(*args, **kwargs)
                return result
            finally:
                pass
                
        return wrapper
    
    def decorate_class(cls: Type) -> Type:
        """Apply the log decorator to all methods of a class."""
        # Create a class-specific logger
        class_name = f"{cls.__module__}.{cls.__name__}"
        class_logger = LogDecorator.get_context_logger(class_name)
        
        # Set logging level if specified
        if level is not None:
            class_logger.setLevel(level)
        
        # Store the logger in the class for access by all methods
        setattr(cls, 'log', class_logger)
        
        # Process all methods in the class
        for attr_name, attr_value in cls.__dict__.items():
            # Skip special methods, non-callable attributes, and already decorated methods
            if (attr_name.startswith('__') or 
                not callable(attr_value) or
                hasattr(attr_value, '_is_log_decorated')):
                continue
                
            # Create method-specific context
            method_name = f"{class_name}.{attr_name}"
            method_logger = LogDecorator.get_context_logger(method_name)
            
            if level is not None:
                method_logger.setLevel(level)
                
            # Create a wrapped method that has access to the logger
            @functools.wraps(attr_value)
            def wrapped_method(self, *args, **kwargs):
                # Store the original globals
                original_globals = attr_value.__globals__.copy()
                
                # Inject the logger
                attr_value.__globals__['log'] = method_logger
                
                try:
                    # Call the original method
                    return attr_value(self, *args, **kwargs)
                finally:
                    pass
                    
            # Mark as decorated to prevent double decoration
            wrapped_method._is_log_decorated = True
            
            # Replace the original method with the wrapped one
            setattr(cls, attr_name, wrapped_method)
            
        return cls
    
    def decorator(obj):
        """Apply appropriate decoration based on object type."""
        if is_class_decorator(obj):
            return decorate_class(obj)
        else:
            return decorate_function(obj)
        
    return decorator
