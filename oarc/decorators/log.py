"""Log decorator module.

This module provides decorators for logging function calls and returns.
"""

import functools
import inspect
import logging
import sys
import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

# Configure basic logging format
LF = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class LogDecorator:
    """Class for providing log decorators."""

    _level = logging.INFO
    _instance = None

    @staticmethod
    def _get_name() -> str:
        """Get the calling module and class name for logging context."""
        frame = inspect.stack()[2]
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
    def set_level(level: int) -> None:
        """Set the global logging level.

        Args:
            level (int): Logging level to set
        """
        LogDecorator._level = level
        if LogDecorator._instance:
            LogDecorator._instance.setLevel(level)


def log(level: Optional[int] = None, log_args: bool = True, log_return: bool = True):
    """Decorator to log function calls with arguments and return values.
    
    Args:
        level (Optional[int]): Logging level to use
        log_args (bool): Whether to log function arguments
        log_return (bool): Whether to log function return values
    
    Returns:
        Callable: The decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = LogDecorator.get()
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Use specified level or default
            log_level = level if level is not None else LogDecorator._level
            
            # Log function entry with arguments if enabled
            if log_args:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.log(log_level, f"Calling {func_name}({signature})")
            else:
                logger.log(log_level, f"Calling {func_name}")
            
            # Measure execution time
            start_time = time.time()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                duration = time.time() - start_time
                
                # Log function return if enabled
                if log_return:
                    logger.log(log_level, 
                              f"{func_name} returned {result!r} (took {duration:.4f}s)")
                else:
                    logger.log(log_level, 
                              f"{func_name} completed in {duration:.4f}s")
                
                return result
            except Exception as e:
                # Log exceptions
                logger.exception(f"{func_name} raised {e.__class__.__name__}: {e}")
                raise
                
        return wrapper
    return decorator
