"""
Unit tests for ModelManager.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from model_manager import ModelManager, ModelLoadError


@pytest.mark.unit
class TestModelManager:
    """Test ModelManager singleton and lazy loading."""
    
    def test_singleton_pattern(self):
        """Test that ModelManager implements singleton pattern."""
        manager1 = ModelManager.get_instance()
        manager2 = ModelManager.get_instance()
        
        assert manager1 is manager2, "ModelManager should return the same instance"
    
    def test_reset_clears_instance(self):
        """Test that reset() clears the singleton instance."""
        manager1 = ModelManager.get_instance()
        ModelManager.reset()
        manager2 = ModelManager.get_instance()
        
        assert manager1 is not manager2, "Reset should create a new instance"
    
    @patch('model_manager.HuggingFaceEmbedding')
    def test_lazy_loading_embed_model(self, mock_hf_embedding, mock_env_vars):
        """Test that embedding model is loaded lazily."""
        manager = ModelManager.get_instance()
        
        # Model should not be loaded yet
        assert manager._embed_model is None
        
        # Access model - should trigger loading
        with patch.dict('os.environ', {'EMBED_MODEL_PATH': 'test/path'}):
            model = manager.get_embed_model()
        
        # Model should now be loaded
        assert manager._embed_model is not None
        mock_hf_embedding.assert_called_once()
    
    @patch('model_manager.OpenAIMultiModal')
    def test_lazy_loading_openai_mm_llm(self, mock_openai_mm, mock_env_vars):
        """Test that OpenAI multimodal LLM is loaded lazily."""
        manager = ModelManager.get_instance()
        
        # Model should not be loaded yet
        assert manager._openai_mm_llm is None
        
        # Access model - should trigger loading
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            model = manager.get_openai_mm_llm()
        
        # Model should now be loaded
        assert manager._openai_mm_llm is not None
        mock_openai_mm.assert_called_once()
    
    def test_missing_env_var_raises_error(self):
        """Test that missing environment variables raise ModelLoadError."""
        manager = ModelManager.get_instance()
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ModelLoadError) as exc_info:
                manager.get_embed_model()
            
            assert "EMBED_MODEL_PATH" in str(exc_info.value)
    
    @patch('model_manager.HuggingFaceEmbedding')
    def test_force_reload(self, mock_hf_embedding, mock_env_vars):
        """Test that force_reload reloads the model."""
        manager = ModelManager.get_instance()
        
        with patch.dict('os.environ', {'EMBED_MODEL_PATH': 'test/path'}):
            # Load model first time
            model1 = manager.get_embed_model()
            assert mock_hf_embedding.call_count == 1
            
            # Get model again without force_reload - should not reload
            model2 = manager.get_embed_model(force_reload=False)
            assert mock_hf_embedding.call_count == 1
            
            # Force reload - should reload
            model3 = manager.get_embed_model(force_reload=True)
            assert mock_hf_embedding.call_count == 2
    
    @patch('model_manager.HuggingFaceEmbedding')
    @patch('model_manager.OpenAIMultiModal')
    @patch('model_manager.OpenAI')
    def test_preload_all_models(self, mock_openai, mock_openai_mm, mock_hf, mock_env_vars):
        """Test preloading all models at once."""
        manager = ModelManager.get_instance()
        
        with patch.dict('os.environ', {
            'EMBED_MODEL_PATH': 'test/path',
            'OPENAI_API_KEY': 'test_key',
            'RETRIEVER_MODEL': 'gpt-4o-mini'
        }):
            status = manager.preload_all_models()
        
        assert status['embed_model'] == 'success'
        assert status['openai_mm_llm'] == 'success'
        assert status['llm'] == 'success'
        
        mock_hf.assert_called_once()
        mock_openai_mm.assert_called_once()
        mock_openai.assert_called_once()
    
    def test_convenience_functions(self, mock_env_vars):
        """Test convenience functions for getting models."""
        from model_manager import get_embed_model, get_openai_mm_llm, get_llm
        
        with patch('model_manager.HuggingFaceEmbedding'):
            with patch.dict('os.environ', {'EMBED_MODEL_PATH': 'test/path'}):
                model = get_embed_model()
                assert model is not None

