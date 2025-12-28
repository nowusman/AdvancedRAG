"""
Logging configuration for AdvancedRAG application.

This module provides centralized logging configuration with:
- Structured logging format with timestamps and module info
- File and console handlers
- Log rotation to prevent large files
- Different log levels for development/production

Usage:
    from logging_config import setup_logging
    
    logger = setup_logging(__name__)
    logger.info("Application started")
    logger.error("An error occurred", exc_info=True)
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from constants import (
    LOG_FILE,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_FORMAT,
    LOG_DATE_FORMAT
)


def setup_logging(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        name: Name of the logger (typically __name__ from the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses environment variable LOG_LEVEL or defaults to INFO
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        logging.Logger: Configured logger instance
    
    Examples:
        >>> logger = setup_logging(__name__)
        >>> logger.info("Processing file", extra={"filename": "test.pdf"})
        
        >>> # Debug mode
        >>> logger = setup_logging(__name__, level="DEBUG")
        >>> logger.debug("Detailed debug information")
        
        >>> # Console only (useful for testing)
        >>> logger = setup_logging(__name__, log_to_file=False)
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Determine log level
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    log_level = getattr(logging, level, logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    # Add file handler with rotation
    if log_to_file:
        # Ensure log directory exists
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default configuration.
    
    Args:
        name: Name of the logger (typically __name__ from the calling module)
    
    Returns:
        logging.Logger: Logger instance
    
    Examples:
        >>> from logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return setup_logging(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding contextual information to log records.
    
    This is useful for adding request IDs, user IDs, or other contextual
    information to all log messages within a specific context.
    
    Examples:
        >>> logger = setup_logging(__name__)
        >>> adapter = LoggerAdapter(logger, {'request_id': '12345'})
        >>> adapter.info("Processing file")  # Will include request_id in logs
    """
    
    def process(self, msg: str, kwargs: dict) -> tuple:
        """Add extra context to log messages."""
        # Merge extra context
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger instance to use
    
    Examples:
        >>> logger = setup_logging(__name__)
        >>> 
        >>> @log_function_call(logger)
        >>> def process_file(filename: str) -> dict:
        >>>     return {"status": "success"}
        >>>
        >>> result = process_file("test.pdf")
        # Logs: Called process_file with args=('test.pdf',) kwargs={}
        # Logs: process_file returned {'status': 'success'}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Called {func.__name__} with args={args} kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"{func.__name__} returned {result}"
                )
                return result
            except Exception as e:
                logger.error(
                    f"{func.__name__} raised {type(e).__name__}: {e}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


def log_exceptions(logger: logging.Logger):
    """
    Decorator to log exceptions raised by a function.
    
    Args:
        logger: Logger instance to use
    
    Examples:
        >>> logger = setup_logging(__name__)
        >>> 
        >>> @log_exceptions(logger)
        >>> def risky_operation():
        >>>     raise ValueError("Something went wrong")
        >>>
        >>> risky_operation()  # Exception is logged before being raised
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    }
                )
                raise
        return wrapper
    return decorator


# Configure root logger
def configure_root_logger(level: str = 'WARNING') -> None:
    """
    Configure the root logger to suppress noisy third-party library logs.
    
    Args:
        level: Log level for root logger (default: WARNING)
    
    Examples:
        >>> configure_root_logger('WARNING')
        # Now only warnings and errors from third-party libraries will be shown
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    # Suppress specific noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('llama_index').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


# Initialize root logger configuration
configure_root_logger()

