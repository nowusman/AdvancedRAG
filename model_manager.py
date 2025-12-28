"""
Model Manager for lazy loading and singleton pattern management of ML models.

This module provides a centralized way to manage ML models, ensuring they are:
1. Loaded only when first accessed (lazy loading)
2. Instantiated only once (singleton pattern)
3. Properly configured from environment variables
4. Gracefully handle loading failures

This significantly improves application startup time and resource usage.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms.openai import OpenAI

from constants import (
    OPENAI_MODEL_GPT4O_MINI,
    OPENAI_MAX_NEW_TOKENS,
    DEFAULT_RETRIEVER_MODEL,
    MODEL_LOADING_TIMEOUT
)

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ModelLoadError(Exception):
    """Raised when a model fails to load."""
    pass


class ModelManager:
    """
    Singleton manager for ML models with lazy loading.
    
    This class ensures models are:
    - Loaded only when first accessed (lazy loading)
    - Instantiated only once (singleton pattern)
    - Properly error handled with fallback strategies
    
    Attributes:
        _instance: Singleton instance of ModelManager
        _embed_model: HuggingFace embedding model instance
        _openai_mm_llm: OpenAI multimodal LLM instance
        _llm: OpenAI LLM instance for retrieval
    
    Examples:
        >>> manager = ModelManager.get_instance()
        >>> embed_model = manager.get_embed_model()
        >>> llm = manager.get_llm()
    """
    
    _instance: Optional['ModelManager'] = None
    _embed_model: Optional[HuggingFaceEmbedding] = None
    _openai_mm_llm: Optional[OpenAIMultiModal] = None
    _llm: Optional[OpenAI] = None
    
    def __new__(cls) -> 'ModelManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """
        Get the singleton instance of ModelManager.
        
        Returns:
            ModelManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance and clear all loaded models.
        
        Useful for testing or when models need to be reloaded.
        """
        cls._instance = None
        cls._embed_model = None
        cls._openai_mm_llm = None
        cls._llm = None
        logger.info("ModelManager instance reset")
    
    def get_embed_model(self, force_reload: bool = False) -> HuggingFaceEmbedding:
        """
        Get or load the HuggingFace embedding model.
        
        Args:
            force_reload: If True, force reload the model even if cached
        
        Returns:
            HuggingFaceEmbedding: The embedding model instance
        
        Raises:
            ModelLoadError: If model loading fails
        
        Examples:
            >>> manager = ModelManager.get_instance()
            >>> embed_model = manager.get_embed_model()
            >>> embeddings = embed_model.get_text_embedding("Hello world")
        """
        if self._embed_model is None or force_reload:
            try:
                logger.info("Loading HuggingFace embedding model...")
                
                embed_model_path = os.getenv('EMBED_MODEL_PATH')
                if not embed_model_path:
                    raise ModelLoadError(
                        "EMBED_MODEL_PATH not found in environment variables. "
                        "Please set it in .env file."
                    )
                
                self._embed_model = HuggingFaceEmbedding(
                    model_name=embed_model_path
                )
                
                logger.info(f"Successfully loaded embedding model from: {embed_model_path}")
                
            except Exception as e:
                error_msg = f"Failed to load embedding model: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
        
        return self._embed_model
    
    def get_openai_mm_llm(self, force_reload: bool = False) -> OpenAIMultiModal:
        """
        Get or load the OpenAI multimodal LLM.
        
        Args:
            force_reload: If True, force reload the model even if cached
        
        Returns:
            OpenAIMultiModal: The multimodal LLM instance
        
        Raises:
            ModelLoadError: If model loading fails or API key is missing
        
        Examples:
            >>> manager = ModelManager.get_instance()
            >>> mm_llm = manager.get_openai_mm_llm()
            >>> response = mm_llm.complete("Describe this image")
        """
        if self._openai_mm_llm is None or force_reload:
            try:
                logger.info("Initializing OpenAI multimodal LLM...")
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ModelLoadError(
                        "OPENAI_API_KEY not found in environment variables. "
                        "Please set it in .env file."
                    )
                
                # Set environment variable for llama-index
                os.environ["OPENAI_API_KEY"] = api_key
                
                self._openai_mm_llm = OpenAIMultiModal(
                    model=OPENAI_MODEL_GPT4O_MINI,
                    api_key=api_key,
                    max_new_tokens=OPENAI_MAX_NEW_TOKENS
                )
                
                logger.info(f"Successfully initialized OpenAI multimodal LLM: {OPENAI_MODEL_GPT4O_MINI}")
                
            except Exception as e:
                error_msg = f"Failed to initialize OpenAI multimodal LLM: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
        
        return self._openai_mm_llm
    
    def get_llm(self, force_reload: bool = False) -> OpenAI:
        """
        Get or load the OpenAI LLM for retrieval.
        
        Args:
            force_reload: If True, force reload the model even if cached
        
        Returns:
            OpenAI: The LLM instance
        
        Raises:
            ModelLoadError: If model loading fails or API key is missing
        
        Examples:
            >>> manager = ModelManager.get_instance()
            >>> llm = manager.get_llm()
            >>> response = llm.complete("What is RAG?")
        """
        if self._llm is None or force_reload:
            try:
                logger.info("Initializing OpenAI LLM for retrieval...")
                
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ModelLoadError(
                        "OPENAI_API_KEY not found in environment variables. "
                        "Please set it in .env file."
                    )
                
                # Set environment variable for llama-index
                os.environ["OPENAI_API_KEY"] = api_key
                
                retriever_model = os.getenv('RETRIEVER_MODEL', DEFAULT_RETRIEVER_MODEL)
                
                self._llm = OpenAI(model=retriever_model)
                
                logger.info(f"Successfully initialized OpenAI LLM: {retriever_model}")
                
            except Exception as e:
                error_msg = f"Failed to initialize OpenAI LLM: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
        
        return self._llm
    
    def preload_all_models(self) -> dict:
        """
        Preload all models at once.
        
        Useful for warming up the application or testing.
        
        Returns:
            dict: Status of each model loading attempt
        
        Examples:
            >>> manager = ModelManager.get_instance()
            >>> status = manager.preload_all_models()
            >>> print(status)
            {'embed_model': 'success', 'openai_mm_llm': 'success', 'llm': 'success'}
        """
        status = {}
        
        try:
            self.get_embed_model()
            status['embed_model'] = 'success'
        except Exception as e:
            status['embed_model'] = f'failed: {str(e)}'
            logger.error(f"Failed to preload embed_model: {e}")
        
        try:
            self.get_openai_mm_llm()
            status['openai_mm_llm'] = 'success'
        except Exception as e:
            status['openai_mm_llm'] = f'failed: {str(e)}'
            logger.error(f"Failed to preload openai_mm_llm: {e}")
        
        try:
            self.get_llm()
            status['llm'] = 'success'
        except Exception as e:
            status['llm'] = f'failed: {str(e)}'
            logger.error(f"Failed to preload llm: {e}")
        
        return status


# Convenience functions for backward compatibility
def get_embed_model() -> HuggingFaceEmbedding:
    """
    Convenience function to get embedding model.
    
    Returns:
        HuggingFaceEmbedding: The embedding model instance
    """
    return ModelManager.get_instance().get_embed_model()


def get_openai_mm_llm() -> OpenAIMultiModal:
    """
    Convenience function to get OpenAI multimodal LLM.
    
    Returns:
        OpenAIMultiModal: The multimodal LLM instance
    """
    return ModelManager.get_instance().get_openai_mm_llm()


def get_llm() -> OpenAI:
    """
    Convenience function to get OpenAI LLM.
    
    Returns:
        OpenAI: The LLM instance
    """
    return ModelManager.get_instance().get_llm()

