# AdvancedRAG Architecture

## Overview

AdvancedRAG is an intelligent document retrieval system that supports multiple storage backends (MongoDB, Weaviate) and provides advanced features like multimodal retrieval, hybrid search, and document management.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   app.py     │  │   app2.py    │  │ui_components │      │
│  │  (MongoDB)   │  │  (Weaviate)  │  │    .py       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  retriever.py│  │  weaviate    │  │file_processing│      │
│  │              │  │  Retriever.py│  │   (future)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │model_manager │  │logging_config│  │error_handling│      │
│  │    .py       │  │     .py      │  │     .py      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  constants.py│  │  storage/    │                         │
│  │              │  │  (interface) │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                      Storage Backends                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   MongoDB    │  │   Weaviate   │  │   Models     │      │
│  │  (DocStore,  │  │  (Vector DB) │  │ (HuggingFace,│      │
│  │  VectorStore)│  │              │  │   OpenAI)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Description

### 1. UI Layer

#### app.py (MongoDB Frontend)
- Main Streamlit application for MongoDB backend
- Handles document upload, deletion, and retrieval
- Uses `ui_components.py` for shared UI elements

#### app2.py (Weaviate Frontend)
- Streamlit application for Weaviate backend
- Similar functionality to app.py but adapted for Weaviate
- Uses `ui_components.py` for shared UI elements

#### ui_components.py
- Shared UI components to eliminate code duplication
- Provides reusable functions for:
  - Document list rendering
  - File upload interface
  - Retrieval results display
  - Confirmation modals

### 2. Business Logic Layer

#### retriever.py (MongoDB Implementation)
- **MongoDBChecker**: Check database contents
- **MongoDBCollectionManager**: Manage documents and collections
- **IntelligentRetriever**: Build and use advanced retrievers
- Supports text, image, and multimodal retrieval

#### weaviateRetriever.py (Weaviate Implementation)
- Similar functionality to retriever.py but for Weaviate
- BM25 search implementation
- Collection management

#### pdf_processor.py
- PDF processing utilities
- Image extraction from PDFs
- OCR capabilities

### 3. Infrastructure Layer

#### model_manager.py
- **Singleton pattern** for ML model management
- **Lazy loading** - models loaded only when needed
- Manages:
  - HuggingFace embedding models
  - OpenAI multimodal LLM
  - OpenAI retrieval LLM
- Improves startup time significantly

#### logging_config.py
- Centralized logging configuration
- File and console handlers
- Log rotation
- Structured logging with timestamps and module info

#### error_handling.py
- Custom exception classes:
  - `StorageError`
  - `ModelLoadError`
  - `ProcessingError`
  - `RetrievalError`
  - `ValidationError`
  - `ConfigurationError`
- Retry decorators with exponential backoff
- Validation utilities
- Graceful degradation helpers

#### constants.py
- All magic numbers and strings centralized
- Configuration values:
  - Chunk sizes
  - Retrieval parameters
  - File processing settings
  - Retry configuration
  - UI messages

#### storage/ (Interface Layer)
- **storage_interface.py**: Abstract base class for storage backends
- **storage_factory.py**: Factory pattern for creating storage instances
- Enables easy addition of new storage backends

### 4. Storage Backends

#### MongoDB
- **Document Store**: Stores parsed documents and nodes
- **Vector Store**: Stores embeddings for similarity search
- **Index Store**: Manages multiple index types

#### Weaviate
- Vector database for semantic search
- BM25 search capabilities
- Built-in support for hybrid search

#### Models
- **HuggingFace Embeddings**: Local embedding model
- **OpenAI GPT-4o-mini**: For multimodal and text generation
- **OpenAI**: For retrieval tasks

## Data Flow

### Document Upload Flow

```
1. User uploads file(s) via Streamlit UI
   ↓
2. Files saved to temporary folder
   ↓
3. File type detection (text/PDF/image)
   ↓
4. Parse documents based on type
   - Text files: SimpleDirectoryReader
   - PDFs/DOCX: DoclingReader
   - Images: OpenAI multimodal extraction
   ↓
5. Chunk documents into nodes
   ↓
6. Generate embeddings
   ↓
7. Store nodes and embeddings in backend
   ↓
8. Create/update vector indexes
   ↓
9. Update index info JSON
   ↓
10. Clean up temporary files
```

### Query Flow

```
1. User enters query via Streamlit UI
   ↓
2. Extract file filters from query (if any)
   ↓
3. Build retriever (or use cached)
   - Load vector index
   - Load multimodal index (if available)
   - Create BM25 retriever
   - Combine with QueryFusionRetriever
   ↓
4. Route query to appropriate retriever
   - Text retriever for text queries
   - Multimodal retriever for image queries
   ↓
5. Execute hybrid search (vector + BM25)
   ↓
6. Rank and merge results
   ↓
7. Return top-k nodes
   ↓
8. Display results in UI
   - Text nodes as text
   - Image nodes as images
```

## Key Design Patterns

### 1. Singleton Pattern (model_manager.py)
- Ensures only one instance of each model
- Reduces memory usage
- Faster subsequent access

### 2. Lazy Loading
- Models loaded on first access, not at import
- Significantly improves application startup time
- Reduces initial memory footprint

### 3. Factory Pattern (storage/)
- Abstracts storage backend creation
- Enables easy swapping between MongoDB and Weaviate
- Facilitates testing with mock implementations

### 4. Strategy Pattern (Planned for file_processing/)
- Different strategies for different file types
- Easy to add new file type processors
- Separation of concerns

### 5. Decorator Pattern (error_handling.py)
- Retry logic applied via decorators
- Exception logging decorators
- Clean separation of error handling from business logic

## Performance Optimizations

### 1. Lazy Model Loading
- **Before**: All models loaded at module import (~30s startup)
- **After**: Models loaded on first use (~2s startup)

### 2. Retriever Caching
- Retriever instances cached in session state
- Rebuilt only when database changes
- Saves ~5-10s per query

### 3. Adaptive Chunking
- Different chunk sizes for different content types
- Optimized from 1,000,000 to 512 tokens
- Improves retrieval precision

### 4. Hybrid Search
- Combines vector search (semantic) with BM25 (keyword)
- Better recall and precision
- Weighted fusion for optimal results

### 5. File Upload Batch Processing
- Multiple files uploaded in single transaction
- Reduced database round trips
- Better error handling

## Security Considerations

### 1. Environment Variables
- All sensitive data (API keys, URIs) in .env
- Never hardcoded in source code

### 2. Input Validation
- File extension validation
- File size limits
- Path traversal prevention

### 3. Error Messages
- Don't expose internal details
- Generic messages for users
- Detailed logs for developers

### 4. Access Control
- MongoDB authentication required
- Weaviate connection secured
- API keys protected

## Scalability Considerations

### 1. Horizontal Scaling
- Stateless application design
- Can run multiple Streamlit instances
- Session state in client (Streamlit)

### 2. Database Scaling
- MongoDB sharding support
- Weaviate clustering support
- Separate read/write operations possible

### 3. Model Serving
- Can move to dedicated model server
- API-based embeddings (OpenAI)
- Batch embedding generation

### 4. Caching Strategy
- Embedding cache (planned)
- Query result cache
- Retriever cache

## Future Enhancements

### 1. Complete Storage Abstraction
- Finish storage interface implementations
- Add more backends (Pinecone, Qdrant)
- Unified configuration

### 2. File Processing Pipeline
- Extract to separate module
- Add more file type support
- Async processing with progress tracking

### 3. Advanced Retrieval
- Query expansion
- Re-ranking
- Multi-hop reasoning

### 4. Monitoring & Observability
- Performance metrics
- Query analytics
- Error tracking

### 5. API Layer
- RESTful API for programmatic access
- Authentication and rate limiting
- API documentation (OpenAPI/Swagger)

## Testing Strategy

### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Fast execution

### Integration Tests
- Test storage backend integration
- Test model loading
- Test file processing pipeline

### End-to-End Tests
- Test complete workflows
- Upload → Retrieve → Display
- UI testing with Selenium

## Deployment

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Run MongoDB app
streamlit run app.py

# Run Weaviate app
streamlit run app2.py
```

### Production
- Use Docker containers
- Separate model server
- Load balancer for multiple instances
- Persistent volume for data
- Monitoring and logging

## Maintenance

### Adding New Features
1. Follow existing patterns
2. Update tests
3. Update documentation
4. Follow code style (black, isort, pylint)

### Troubleshooting
- Check logs in `advancedrag.log`
- Verify environment variables
- Check database connectivity
- Verify model paths

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style
- Pull request process
- Testing requirements
- Documentation standards

