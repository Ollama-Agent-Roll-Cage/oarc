"""
Singleton decorator for OARC.

This module provides a decorator that implements the singleton pattern,
ensuring only one instance of a class exists regardless of how many times
it is instantiated. It supports initialization arguments and provides 
logging of instance creation and reuse.
"""

import functools
import inspect
from typing import Type, TypeVar, Dict, Any, Callable

from oarc.utils.log import log

T = TypeVar('T')

# Dictionary to store singleton instances by class
_instances: Dict[Type, Any] = {}

def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator that implements the singleton pattern.
    
    This decorator ensures that only one instance of a class is created,
    regardless of how many times the constructor is called. If an instance
    already exists, it is returned instead of creating a new one.
    
    The decorator logs instance creation and reuse, and provides warnings
    when the instance is requested with different parameters than those
    used to create it.
    
    Args:
        cls: The class to make into a singleton
        
    Returns:
        The decorated class with singleton behavior
    """
    original_init = cls.__init__
    original_new = cls.__new__
    
    try:
        # Get the parameter names from the original __init__ method
        init_signature = inspect.signature(original_init)
        param_names = [p for p in init_signature.parameters if p != 'self']
    except (ValueError, TypeError) as e:
        # Handle case where signature cannot be inspected
        log.warning(f"Could not inspect signature for {cls.__name__}: {e}")
        param_names = []
    
    @functools.wraps(original_new)
    def __new__(cls, *args, **kwargs):
        if cls not in _instances:
            instance = original_new(cls)
            _instances[cls] = instance
            return instance
        return _instances[cls]
    
    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        # Check if this instance has already been initialized
        if not hasattr(self, '_initialized'):
            log.info(f"Creating new {cls.__name__} instance")
            original_init(self, *args, **kwargs)
            
            # Store initialization parameters for comparison
            self._init_args = args
            self._init_kwargs = kwargs
            self._initialized = True
        else:
            # Check if initialization parameters differ and log a warning if they do
            if args != self._init_args or kwargs != self._init_kwargs:
                # Create a more informative message about the parameter differences
                new_params = {}
                old_params = {}
                
                # Combine positional and keyword arguments
                for i, param_name in enumerate(param_names):
                    if i < len(args):
                        new_params[param_name] = args[i]
                    if i < len(self._init_args):
                        old_params[param_name] = self._init_args[i]
                
                # Add keyword arguments
                new_params.update(kwargs)
                old_params.update(self._init_kwargs)
                
                # Find differences
                diff_params = []
                for key in set(new_params.keys()) | set(old_params.keys()):
                    if key in new_params and key in old_params and new_params[key] != old_params[key]:
                        diff_params.append(f"{key}={new_params[key]} (was {old_params[key]})")
                    elif key in new_params and key not in old_params:
                        diff_params.append(f"{key}={new_params[key]} (was not set)")
                    elif key not in new_params and key in old_params:
                        diff_params.append(f"{key} not set (was {old_params[key]})")
                
                if diff_params:
                    diff_str = ", ".join(diff_params)
                    log.warning(f"Requested {cls.__name__} instance with different parameters: {diff_str}. "
                               f"Using existing instance with original parameters.")
            else:
                log.debug(f"Using existing {cls.__name__} instance")
    
    # Replace the class methods
    cls.__new__ = __new__
    cls.__init__ = __init__
    
    # Add a custom reset method for testing and cleanup
    def _reset_singleton(cls):
        """Remove the singleton instance, allowing a new one to be created"""
        if cls in _instances:
            del _instances[cls]
            log.info(f"Singleton instance of {cls.__name__} has been reset")
    
    cls._reset_singleton = classmethod(_reset_singleton)
    
    return cls
