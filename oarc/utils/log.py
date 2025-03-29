"""Logging Utility Module.

This module provides a unified logging class for the project, managing log
levels, formatting, and specialized logging configurations.

Functions:
    None

Classes:
    Log: Provides unified logging functionality across the project.
"""

import functools
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

LF = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
T = TypeVar("T")


class Log:
    """Unified logging class for the project.

    This class:
    1. Manages a singleton logging instance for consistent logging.
    2. Provides methods for different log levels (debug, info, warn, error).
    3. Handles logging configuration and level management.
    4. Offers specialized logging setup for external libraries.

    Methods:
        Core Setup:
            _get_name(): Gets the calling module and class name.
            _setup(): Initializes the logging system.
            get(): Returns the logger instance.
            get_handler(): Returns stdout stream handler.

        Level Management:
            set_level(level): Sets the global logging level.
            level(level): Decorator for temporary level changes.

        Logging Methods:
            debug(message): Logs debug messages.
            info(message): Logs informational messages.
            warn(message): Logs warning messages.
            error(message, exc_info): Logs error messages with optional trace.

        Configuration:
            configure_sqlalchemy_logging(): Configures SQLAlchemy logging
                levels.
    """

    _instance = None
    _level = logging.INFO
    _file_handlers: Dict[str, logging.FileHandler] = {}
    _default_log_dir = None

    # Logging levels
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    # Core setup methods
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
        logging.basicConfig(level=Log._level,
                            format=LF,
                            datefmt="%Y.%m.%d:%H:%M:%S",
                            handlers=[logging.StreamHandler(sys.stdout)])
        Log._instance = logging.getLogger("oarc")
        Log._instance.setLevel(Log._level)

    @staticmethod
    def get() -> logging.Logger:
        """Get or create the singleton logger instance."""
        if Log._instance is None:
            Log._setup()
        return Log._instance

    @staticmethod
    def get_handler() -> logging.StreamHandler:
        """Get a stdout stream handler for logging configuration."""
        return logging.StreamHandler(sys.stdout)

    @staticmethod
    def set_default_log_dir(log_dir: Union[str, Path]) -> None:
        """Set the default directory for log files.
        
        Args:
            log_dir (Union[str, Path]): Directory to store log files
        """
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        
        log_dir.mkdir(exist_ok=True, parents=True)
        Log._default_log_dir = log_dir

    @staticmethod
    def add_file_handler(logger_name: str = "oarc", 
                         filename: Optional[str] = None,
                         log_dir: Optional[Union[str, Path]] = None) -> logging.FileHandler:
        """Add a file handler to the specified logger.
        
        Args:
            logger_name (str): Name of the logger to add the handler to
            filename (Optional[str]): Name of the log file
            log_dir (Optional[Union[str, Path]]): Directory to store log file, 
                                            uses default_log_dir if None
                
        Returns:
            logging.FileHandler: The created file handler
        """
        # Use default log directory if none specified
        if log_dir is None:
            if Log._default_log_dir is None:
                # Default to project_root/logs if no default set
                project_root = Path(__file__).resolve().parents[2]  # Up three levels from utils/log.py
                log_dir = project_root / "logs"
            else:
                log_dir = Log._default_log_dir
        
        # Create log directory if it doesn't exist
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default filename if not provided
        if filename is None:
            filename = f"{logger_name}.log"
            
        # Create full path
        log_path = log_dir / filename
        
        # Check if handler already exists for this path
        handler_key = f"{logger_name}:{str(log_path)}"
        if handler_key in Log._file_handlers:
            return Log._file_handlers[handler_key]
        
        # Create and configure file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(LF))
        file_handler.setLevel(Log._level)
        
        # Add handler to the specified logger
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
        
        # Store handler for future reference
        Log._file_handlers[handler_key] = file_handler
        
        return file_handler

    @staticmethod
    def get_logger(name: str, with_file: bool = False, 
                  filename: Optional[str] = None,
                  log_dir: Optional[Union[str, Path]] = None) -> logging.Logger:
        """Get a named logger with optional file handler.
        
        Args:
            name (str): Name of the logger
            with_file (bool): Whether to add a file handler
            filename (Optional[str]): Name of the log file (defaults to name.log)
            log_dir (Optional[Union[str, Path]]): Directory for log file
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(Log._level)
        
        if with_file:
            if not filename:
                filename = f"{name}.log"
            Log.add_file_handler(name, filename, log_dir)
            
        return logger

    # Level management
    @staticmethod
    def set_level(level: int) -> None:
        """Set the global logging level for all loggers.

        Args:
            level (int): Logging level to set
        """
        Log._level = level
        if Log._instance:
            Log._instance.setLevel(Log._level)
            logger = logging.getLogger("oarc")
            logger.setLevel(Log._level)
            
        # Update file handler levels
        for handler in Log._file_handlers.values():
            handler.setLevel(level)

    @staticmethod
    def level(level: int) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to temporarily change log level during method execution.

        Args:
            level (int): Logging level to set temporarily

        Returns:
            Callable: Decorator function
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                prev_level = Log._level
                Log.set_level(level)
                try:
                    return func(*args, **kwargs)
                finally:
                    Log.set_level(prev_level)

            return wrapper

        return decorator

    # Logging methods
    @staticmethod
    def debug(message: str) -> None:
        """Log a debug level message with calling context.

        Args:
            message (str): Message to log
        """
        if Log._level <= logging.DEBUG:
            logger = Log.get()
            logger.name = Log._get_name()
            logger.debug(message)

    @staticmethod
    def info(message: str) -> None:
        """Log an info level message with calling context.

        Args:
            message (str): Message to log
        """
        if Log._level <= logging.INFO:
            logger = Log.get()
            logger.name = Log._get_name()
            logger.info(message)

    @staticmethod
    def warn(message: str) -> None:
        """Log a warning level message with calling context.

        Args:
            message (str): Message to log
        """
        if Log._level <= logging.WARNING:
            logger = Log.get()
            logger.name = Log._get_name()
            logger.warning(message)

    @staticmethod
    def error(message: str, exc_info: Optional[bool] = None) -> None:
        """Log an error level message with call context and optional trace.

        Args:
            message (str): Message to log
            exc_info (Optional[bool]): Include exception info if True
        """
        if Log._level <= logging.ERROR:
            logger = Log.get()
            logger.name = Log._get_name()
            logger.error(message, exc_info=exc_info)

    # Configuration methods
    @staticmethod
    def configure_sqlalchemy_logging() -> None:
        """Configure SQLAlchemy logging levels to suppress verbose output."""
        logging_modules = [
            "sqlalchemy.engine.Engine.database", "sqlalchemy.pool",
            "sqlalchemy.dialects"
        ]
        for module in logging_modules:
            logging.getLogger(module).setLevel(logging.CRITICAL)
        Log.info("SQLAlchemy logging configured successfully.")