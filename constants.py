"""
Constants and configuration values for AdvancedRAG application.

This module centralizes all magic numbers, strings, and configuration values
to improve maintainability and prevent scattered hardcoded values.
"""

# ============================================================================
# Chunking and Text Processing
# ============================================================================

# Default chunk size for text splitting (optimized from 1000000 to 512 tokens)
# Smaller chunks improve retrieval precision and reduce memory usage
DEFAULT_CHUNK_SIZE = 512

# Overlap between consecutive chunks to maintain context
DEFAULT_CHUNK_OVERLAP = 50

# Adaptive chunk sizes for different content types
ADAPTIVE_CHUNK_SIZES = {
    'code': 300,           # Smaller chunks for code to maintain function boundaries
    'table': 400,          # Medium chunks for tabular data
    'text': 512,           # Default for general text
    'documentation': 600,  # Slightly larger for documentation with context
}

# Maximum chunk size for image processing (keep images as single chunks)
IMAGE_CHUNK_SIZE = 1000000

# Text splitter separator
TEXT_SPLITTER_SEPARATOR = '#'

# ============================================================================
# Retrieval Parameters
# ============================================================================

# Number of similar results to return in vector search
DEFAULT_SIMILARITY_TOP_K = 3

# Number of results for BM25 retrieval
BM25_TOP_K = 10

# Window size for context expansion in retrieval
RETRIEVAL_WINDOW_SIZE = 3

# Delta percentage for similarity threshold
SIMILARITY_DELTA_PERCENT = 0.005

# Query fusion parameters
QUERY_FUSION_NUM_QUERIES = 4
QUERY_FUSION_TOP_K = 5

# ============================================================================
# File Processing
# ============================================================================

# Batch size for parallel image processing
IMAGE_PROCESSING_BATCH_SIZE = 5

# Temporary upload folder
FILE_UPLOAD_FOLDER = 'uploads'

# Supported file extensions
SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.csv'}
SUPPORTED_DOCLING_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

# File size limits (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
STREAM_PROCESSING_THRESHOLD = 10 * 1024 * 1024  # 10 MB - files larger than this use streaming

# ============================================================================
# Model Configuration
# ============================================================================

# OpenAI model settings
OPENAI_MODEL_GPT4O_MINI = "gpt-4o-mini"
OPENAI_MAX_NEW_TOKENS = 1500

# Default retriever model
DEFAULT_RETRIEVER_MODEL = "gpt-4o-mini"

# ============================================================================
# Storage Configuration
# ============================================================================

# Storage backend types
STORAGE_TYPE_MONGODB = "mongodb"
STORAGE_TYPE_WEAVIATE = "weaviate"

# MongoDB collection suffixes
MONGODB_DATA_SUFFIX = "/data"
MONGODB_INDEX_NAMESPACE = "indexes"

# Index types
INDEX_TYPE_VECTOR = "vector_index"
INDEX_TYPE_MULTIMODAL = "multimodal_index"
INDEX_TYPE_KEYWORD = "keyword_index"

# ============================================================================
# Caching Configuration
# ============================================================================

# Embedding cache settings
EMBEDDING_CACHE_SIZE = 1000  # Number of embeddings to cache in memory
EMBEDDING_CACHE_FILE = ".embedding_cache.pkl"

# Retriever cache settings
RETRIEVER_CACHE_TTL = 3600  # Time to live in seconds (1 hour)

# ============================================================================
# Retry and Error Handling
# ============================================================================

# Retry configuration for transient failures
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff: 1s, 2s, 4s
RETRY_MIN_WAIT = 1  # Minimum wait time in seconds
RETRY_MAX_WAIT = 10  # Maximum wait time in seconds

# Timeout settings (in seconds)
DATABASE_CONNECTION_TIMEOUT = 30
MODEL_LOADING_TIMEOUT = 120
FILE_PROCESSING_TIMEOUT = 300

# ============================================================================
# Logging Configuration
# ============================================================================

# Log file settings
LOG_FILE = "advancedrag.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# UI Configuration
# ============================================================================

# Streamlit page configuration
PAGE_TITLE = "Intelligent Retriever"
PAGE_ICON = "🔍"

# UI Messages
MSG_DATABASE_EMPTY = "Database empty."
MSG_NO_RETRIEVER = "No retriever created."
MSG_NO_RELATED_INFO = "No related information. Please check whether there are relevant data in the collection."
MSG_UPLOAD_SUCCESS = "Files uploaded successfully!"
MSG_UPLOAD_SELECT_FILES = "Please select file(s) to upload."
MSG_DELETE_SUCCESS = "Deleted successfully!"
MSG_COLLECTION_DELETED = "Collection deleted successfully!"

# UI Emojis
EMOJI_FOLDER = "📁"
EMOJI_FILE = "📄"
EMOJI_DELETE = "❌"
EMOJI_DELETE_COLLECTION = "🗑️"
EMOJI_WARNING = "⚠️"
EMOJI_SUCCESS = "✅"

# ============================================================================
# Performance Optimization
# ============================================================================

# Thread pool settings for parallel processing
MAX_WORKER_THREADS = 4
MAX_WORKER_PROCESSES = 2

# Memory optimization
USE_GENERATORS = True  # Use generators instead of lists where possible
ENABLE_COMPRESSION = True  # Enable response compression

# Vector store optimization
VECTOR_STORE_HNSW_M = 16  # HNSW parameter for index
VECTOR_STORE_HNSW_EF_CONSTRUCTION = 200  # HNSW parameter for construction
VECTOR_STORE_HNSW_EF_SEARCH = 100  # HNSW parameter for search

# ============================================================================
# Image Processing
# ============================================================================

# Image prompt for description extraction
IMAGE_DESCRIPTION_PROMPT = "Extract the contents in the image. Do not add other things."

# Image save format
IMAGE_SAVE_FORMAT = "PNG"
IMAGE_QUALITY = 95

# ============================================================================
# Regular Expressions
# ============================================================================

# Pattern for extracting file filters from queries
FILE_FILTER_PATTERN = r'\[file:\s*([^\]]+)\]'

# Pattern for extracting dates
DATE_PATTERN = r'\d{4}-\d{2}-\d{2}'

