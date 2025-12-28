"""
Pytest configuration and fixtures for AdvancedRAG tests.
"""

import pytest
import os
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture
def mock_mongo_uri():
    """Provide a mock MongoDB URI for testing."""
    return "mongodb://localhost:27017/test_db"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    env_vars = {
        'MONGO_URI': 'mongodb://localhost:27017/test',
        'DB_NAME': 'test_db',
        'COLLECTION_NAME': 'test_collection',
        'PERSIST_DIR': './test_persist',
        'INDEX_INFO_PATH': './test_index_info.json',
        'IMAGE_FOLDER': './test_images',
        'OPENAI_API_KEY': 'test_api_key',
        'EMBED_MODEL_PATH': 'test/model/path',
        'RETRIEVER_MODEL': 'gpt-4o-mini',
        'LOG_LEVEL': 'DEBUG'
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def mock_embed_model():
    """Provide a mock embedding model."""
    mock_model = Mock()
    mock_model.get_text_embedding.return_value = [0.1] * 768
    return mock_model


@pytest.fixture
def mock_llm():
    """Provide a mock LLM."""
    mock = Mock()
    mock.complete.return_value = Mock(text="Test response")
    return mock


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            'file_name': 'test1.txt',
            'content': 'This is test document 1',
            'metadata': {'type': 'text'}
        },
        {
            'file_name': 'test2.pdf',
            'content': 'This is test document 2',
            'metadata': {'type': 'pdf'}
        }
    ]


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    return str(test_dir)


@pytest.fixture
def sample_index_info():
    """Provide sample index information."""
    return {
        'test_collection_vector_index': 'test_vector_id_123',
        'test_collection_multimodal_index': 'test_multimodal_id_456'
    }


@pytest.fixture(autouse=True)
def reset_model_manager():
    """Reset ModelManager singleton between tests."""
    from model_manager import ModelManager
    ModelManager.reset()
    yield
    ModelManager.reset()


@pytest.fixture
def mock_storage_context():
    """Provide a mock StorageContext."""
    mock_context = Mock()
    mock_context.docstore = Mock()
    mock_context.index_store = Mock()
    mock_context.vector_store = Mock()
    mock_context.persist = Mock()
    return mock_context


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test"
    )

