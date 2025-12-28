"""
Unit tests for constants module.
"""

import pytest
from constants import *


@pytest.mark.unit
class TestConstants:
    """Test that constants are properly defined."""
    
    def test_chunk_sizes_are_positive(self):
        """Test that all chunk sizes are positive integers."""
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_CHUNK_OVERLAP >= 0
        assert IMAGE_CHUNK_SIZE > 0
        
        for content_type, size in ADAPTIVE_CHUNK_SIZES.items():
            assert size > 0, f"Chunk size for {content_type} must be positive"
    
    def test_retrieval_parameters(self):
        """Test retrieval parameters are valid."""
        assert DEFAULT_SIMILARITY_TOP_K > 0
        assert BM25_TOP_K > 0
        assert RETRIEVAL_WINDOW_SIZE > 0
        assert 0 < SIMILARITY_DELTA_PERCENT < 1
    
    def test_file_extensions(self):
        """Test file extension sets are properly defined."""
        assert len(SUPPORTED_TEXT_EXTENSIONS) > 0
        assert len(SUPPORTED_DOCLING_EXTENSIONS) > 0
        assert len(SUPPORTED_IMAGE_EXTENSIONS) > 0
        
        # Check all extensions start with dot
        for ext in SUPPORTED_TEXT_EXTENSIONS:
            assert ext.startswith('.')
    
    def test_storage_types(self):
        """Test storage type constants."""
        assert STORAGE_TYPE_MONGODB == "mongodb"
        assert STORAGE_TYPE_WEAVIATE == "weaviate"
    
    def test_retry_configuration(self):
        """Test retry configuration values."""
        assert MAX_RETRIES > 0
        assert RETRY_MIN_WAIT > 0
        assert RETRY_MAX_WAIT > RETRY_MIN_WAIT
        assert RETRY_BACKOFF_FACTOR > 1
    
    def test_timeout_settings(self):
        """Test timeout settings are reasonable."""
        assert DATABASE_CONNECTION_TIMEOUT > 0
        assert MODEL_LOADING_TIMEOUT > 0
        assert FILE_PROCESSING_TIMEOUT > 0
    
    def test_ui_messages_not_empty(self):
        """Test UI messages are defined."""
        assert len(MSG_DATABASE_EMPTY) > 0
        assert len(MSG_NO_RETRIEVER) > 0
        assert len(MSG_UPLOAD_SUCCESS) > 0
    
    def test_performance_settings(self):
        """Test performance optimization settings."""
        assert MAX_WORKER_THREADS > 0
        assert MAX_WORKER_PROCESSES > 0
        assert isinstance(USE_GENERATORS, bool)
        assert isinstance(ENABLE_COMPRESSION, bool)

