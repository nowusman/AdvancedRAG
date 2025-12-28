"""
Storage layer abstraction for AdvancedRAG.

This package provides a unified interface for different storage backends
(MongoDB, Weaviate) with common operations for document management and retrieval.

Usage:
    from storage import StorageFactory, StorageType
    
    # Create MongoDB storage
    storage = StorageFactory.create(
        StorageType.MONGODB,
        uri="mongodb://localhost",
        db_name="mydb"
    )
    
    # Upload files
    storage.upload_files(["file1.pdf", "file2.txt"])
    
    # Build retriever
    retriever = storage.build_retriever()
"""

from storage.storage_interface import StorageInterface, StorageType
from storage.storage_factory import StorageFactory

__all__ = [
    'StorageInterface',
    'StorageType',
    'StorageFactory'
]

