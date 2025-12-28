"""
Storage interface abstraction for AdvancedRAG.

Defines the abstract base class that all storage implementations must follow.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from llama_index.core.schema import BaseNode


class StorageType(str, Enum):
    """Enum for supported storage backend types."""
    MONGODB = "mongodb"
    WEAVIATE = "weaviate"


class StorageInterface(ABC):
    """
    Abstract base class for storage backends.
    
    All storage implementations (MongoDB, Weaviate) must implement this interface
    to ensure consistent behavior across different backends.
    
    This enables:
    - Easy swapping between storage backends
    - Consistent API for all storage operations
    - Simplified testing with mock implementations
    - Future extensibility for new storage backends
    """
    
    @abstractmethod
    def __init__(self, **config):
        """
        Initialize storage backend with configuration.
        
        Args:
            **config: Backend-specific configuration parameters
        """
        pass
    
    @abstractmethod
    def upload_files(
        self,
        file_paths: List[str],
        collection_name: Optional[str] = None
    ) -> str:
        """
        Upload files to the storage backend.
        
        Args:
            file_paths: List of file paths to upload
            collection_name: Optional collection/namespace name
        
        Returns:
            str: Success message or status
        
        Raises:
            StorageError: If upload fails
        """
        pass
    
    @abstractmethod
    def delete_files(
        self,
        file_names: List[str],
        collection_name: Optional[str] = None
    ) -> str:
        """
        Delete specific files from the storage backend.
        
        Args:
            file_names: List of file names to delete
            collection_name: Optional collection/namespace name
        
        Returns:
            str: Success message or status
        
        Raises:
            StorageError: If deletion fails
        """
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> str:
        """
        Delete an entire collection/namespace.
        
        Args:
            collection_name: Name of collection to delete
        
        Returns:
            str: Success message or status
        
        Raises:
            StorageError: If deletion fails
        """
        pass
    
    @abstractmethod
    def get_documents(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get list of documents in storage.
        
        Args:
            collection_name: Optional specific collection name
        
        Returns:
            Dict mapping collection names to lists of file names
        
        Raises:
            StorageError: If retrieval fails
        """
        pass
    
    @abstractmethod
    def build_retriever(
        self,
        collection_name: Optional[str] = None,
        file_filter: Optional[List[str]] = None,
        **retriever_config
    ) -> Any:
        """
        Build a retriever for querying documents.
        
        Args:
            collection_name: Optional collection to build retriever for
            file_filter: Optional list of files to filter by
            **retriever_config: Additional retriever configuration
        
        Returns:
            Retriever instance
        
        Raises:
            StorageError: If retriever creation fails
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        **query_params
    ) -> List[BaseNode]:
        """
        Query documents and return relevant nodes.
        
        Args:
            query_text: Query string
            collection_name: Optional collection to query
            top_k: Number of results to return
            **query_params: Additional query parameters
        
        Returns:
            List of retrieved nodes
        
        Raises:
            RetrievalError: If query fails
        """
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """
        Check if storage backend is healthy and accessible.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def get_statistics(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Args:
            collection_name: Optional specific collection
        
        Returns:
            Dict with statistics (num_documents, num_collections, size, etc.)
        """
        pass

