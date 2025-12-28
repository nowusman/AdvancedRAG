"""
Error handling utilities for AdvancedRAG application.

This module provides:
- Custom exception classes for different error types
- Retry decorators with exponential backoff
- Graceful degradation strategies
- Detailed error context and logging

Usage:
    from error_handling import retry_on_failure, StorageError
    
    @retry_on_failure(max_retries=3)
    def upload_to_database(data):
        # Operation that might fail transiently
        pass
"""

import time
import functools
import logging
from typing import Callable, Optional, Type, Tuple, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from constants import (
    MAX_RETRIES,
    RETRY_MIN_WAIT,
    RETRY_MAX_WAIT,
    RETRY_BACKOFF_FACTOR
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exception Classes
# ============================================================================

class AdvancedRAGError(Exception):
    """Base exception class for all AdvancedRAG errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize the exception with message and optional details.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class StorageError(AdvancedRAGError):
    """
    Raised when storage operations (MongoDB, Weaviate) fail.
    
    Examples:
        >>> raise StorageError(
        ...     "Failed to connect to MongoDB",
        ...     details={"uri": "mongodb://localhost", "timeout": 30}
        ... )
    """
    pass


class ModelLoadError(AdvancedRAGError):
    """
    Raised when ML models fail to load.
    
    Examples:
        >>> raise ModelLoadError(
        ...     "Failed to load embedding model",
        ...     details={"model_path": "/models/embeddings", "error": "File not found"}
        ... )
    """
    pass


class ProcessingError(AdvancedRAGError):
    """
    Raised when file processing operations fail.
    
    Examples:
        >>> raise ProcessingError(
        ...     "Failed to extract text from PDF",
        ...     details={"file": "document.pdf", "page": 5}
        ... )
    """
    pass


class RetrievalError(AdvancedRAGError):
    """
    Raised when retrieval operations fail.
    
    Examples:
        >>> raise RetrievalError(
        ...     "No documents found matching query",
        ...     details={"query": "machine learning", "collection": "docs"}
        ... )
    """
    pass


class ValidationError(AdvancedRAGError):
    """
    Raised when input validation fails.
    
    Examples:
        >>> raise ValidationError(
        ...     "Invalid file type",
        ...     details={"file": "image.xyz", "allowed": [".pdf", ".txt"]}
        ... )
    """
    pass


class ConfigurationError(AdvancedRAGError):
    """
    Raised when configuration is invalid or missing.
    
    Examples:
        >>> raise ConfigurationError(
        ...     "Missing required environment variable",
        ...     details={"variable": "MONGO_URI", "file": ".env"}
        ... )
    """
    pass


# ============================================================================
# Retry Decorators
# ============================================================================

def retry_on_failure(
    max_retries: int = MAX_RETRIES,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    min_wait: int = RETRY_MIN_WAIT,
    max_wait: int = RETRY_MAX_WAIT,
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to retry on
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        logger_instance: Logger to use (defaults to module logger)
    
    Returns:
        Decorated function with retry logic
    
    Examples:
        >>> @retry_on_failure(max_retries=3, exceptions=(ConnectionError,))
        >>> def connect_to_database(uri: str):
        >>>     # Code that might fail with ConnectionError
        >>>     pass
        
        >>> @retry_on_failure(max_retries=5, min_wait=2, max_wait=30)
        >>> def upload_file(file_path: str):
        >>>     # Code that might fail transiently
        >>>     pass
    """
    log = logger_instance or logger
    
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(log, logging.WARNING),
            after=after_log(log, logging.DEBUG)
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                log.error(
                    f"Function {func.__name__} failed: {str(e)}",
                    extra={'args': args, 'kwargs': kwargs}
                )
                raise
        
        return wrapper
    
    return decorator


def retry_with_fallback(
    fallback_func: Callable,
    max_retries: int = MAX_RETRIES,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator to retry a function and use fallback on final failure.
    
    Args:
        fallback_func: Function to call if all retries fail
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to retry on
    
    Returns:
        Decorated function with retry and fallback logic
    
    Examples:
        >>> def fallback_handler(*args, **kwargs):
        >>>     return {"status": "failed", "message": "Using fallback"}
        
        >>> @retry_with_fallback(fallback_handler, max_retries=3)
        >>> def risky_operation(data: dict):
        >>>     # Code that might fail
        >>>     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    wait_time = min(
                        RETRY_MIN_WAIT * (RETRY_BACKOFF_FACTOR ** attempt),
                        RETRY_MAX_WAIT
                    )
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            
            # All retries failed, use fallback
            logger.error(
                f"All {max_retries} attempts failed for {func.__name__}. "
                f"Using fallback. Last error: {last_exception}"
            )
            return fallback_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# Context Managers
# ============================================================================

class ErrorContext:
    """
    Context manager for adding error context to exceptions.
    
    Examples:
        >>> with ErrorContext("processing file", file="document.pdf"):
        >>>     process_document("document.pdf")
        # If an error occurs, it will include the context
    """
    
    def __init__(self, operation: str, **context):
        """
        Initialize error context.
        
        Args:
            operation: Description of the operation
            **context: Additional context key-value pairs
        """
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context and enhance exception if one occurred.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if exc_val is not None:
            # Enhance exception with context
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            enhanced_message = (
                f"Error during {self.operation}: {str(exc_val)}"
                f"{f' ({context_str})' if context_str else ''}"
            )
            
            logger.error(
                enhanced_message,
                exc_info=(exc_type, exc_val, exc_tb),
                extra=self.context
            )
            
            # Don't suppress the exception
            return False


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_not_none(value: Any, name: str) -> None:
    """
    Validate that a value is not None.
    
    Args:
        value: Value to validate
        name: Name of the value (for error message)
    
    Raises:
        ValidationError: If value is None
    
    Examples:
        >>> validate_not_none(some_value, "configuration")
        >>> validate_not_none(None, "config")  # Raises ValidationError
    """
    if value is None:
        raise ValidationError(
            f"{name} cannot be None",
            details={"parameter": name}
        )


def validate_file_exists(file_path: str) -> None:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file
    
    Raises:
        ValidationError: If file doesn't exist
    
    Examples:
        >>> validate_file_exists("/path/to/file.pdf")
        >>> validate_file_exists("/nonexistent.pdf")  # Raises ValidationError
    """
    import os
    if not os.path.exists(file_path):
        raise ValidationError(
            f"File not found: {file_path}",
            details={"file_path": file_path}
        )


def validate_file_extension(
    file_path: str,
    allowed_extensions: set
) -> None:
    """
    Validate that a file has an allowed extension.
    
    Args:
        file_path: Path to the file
        allowed_extensions: Set of allowed extensions (e.g., {'.pdf', '.txt'})
    
    Raises:
        ValidationError: If file extension is not allowed
    
    Examples:
        >>> validate_file_extension("doc.pdf", {'.pdf', '.txt'})
        >>> validate_file_extension("image.xyz", {'.pdf'})  # Raises ValidationError
    """
    import os
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext not in allowed_extensions:
        raise ValidationError(
            f"File extension {ext} not allowed",
            details={
                "file_path": file_path,
                "extension": ext,
                "allowed_extensions": list(allowed_extensions)
            }
        )


def validate_config_var(var_name: str, value: Optional[str] = None) -> str:
    """
    Validate that a configuration variable is set.
    
    Args:
        var_name: Name of the environment variable
        value: Optional value (if None, reads from environment)
    
    Returns:
        str: The configuration value
    
    Raises:
        ConfigurationError: If variable is not set or empty
    
    Examples:
        >>> uri = validate_config_var("MONGO_URI")
        >>> api_key = validate_config_var("API_KEY", value=os.getenv("API_KEY"))
    """
    import os
    
    config_value = value or os.getenv(var_name)
    
    if not config_value:
        raise ConfigurationError(
            f"Configuration variable '{var_name}' is not set or empty",
            details={
                "variable": var_name,
                "hint": "Check your .env file or environment variables"
            }
        )
    
    return config_value


# ============================================================================
# Graceful Degradation
# ============================================================================

def safe_execute(
    func: Callable,
    default: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    Execute a function and return default value if it fails.
    
    Args:
        func: Function to execute
        default: Default value to return on failure
        log_error: Whether to log errors
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Result of func or default value on failure
    
    Examples:
        >>> result = safe_execute(risky_function, default=[], arg1="value")
        >>> # If risky_function fails, returns []
    """
    try:
        return func(**kwargs)
    except Exception as e:
        if log_error:
            logger.error(
                f"safe_execute failed for {func.__name__}: {e}",
                exc_info=True
            )
        return default

