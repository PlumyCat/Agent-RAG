import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    # =====================================================
    # Azure OpenAI Configuration
    # =====================================================
    
    # Azure OpenAI settings (primary)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    # Fallback to standard OpenAI if Azure not configured
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")
    
    # Determine which API to use
    USE_AZURE_OPENAI = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)
    
    # Model settings
    MODEL_NAME = AZURE_OPENAI_DEPLOYMENT_NAME if USE_AZURE_OPENAI else OPENAI_MODEL_NAME
    BASE_URL = AZURE_OPENAI_ENDPOINT if USE_AZURE_OPENAI else OPENAI_BASE_URL
    API_KEY = AZURE_OPENAI_API_KEY if USE_AZURE_OPENAI else OPENAI_API_KEY

    # =====================================================
    # PostgreSQL Configuration
    # =====================================================
    
    # Database connection settings
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "mcp_rag")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "mcp_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mcp_password")
    POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")
    
    # Alternative connection string
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    
    # Async connection string for asyncpg
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    # Connection pool settings
    POSTGRES_MIN_CONNECTIONS = int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5"))
    POSTGRES_MAX_CONNECTIONS = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20"))
    POSTGRES_CONNECTION_TIMEOUT = int(os.getenv("POSTGRES_CONNECTION_TIMEOUT", "30"))

    # =====================================================
    # Redis Configuration (for caching)
    # =====================================================
    
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

    # =====================================================
    # Vector Search and Embeddings
    # =====================================================
    
    # Embedding settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    USE_OPENAI_EMBEDDINGS = bool(API_KEY)
    
    # Search settings
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    DEFAULT_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))

    # =====================================================
    # Document Processing Settings
    # =====================================================
    
    # File processing limits
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
    MEMORY_THRESHOLD_MB = int(os.getenv("MEMORY_THRESHOLD_MB", "30"))
    PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "600"))
    
    # Text chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    CHUNK_SIZE_BYTES = 8192  # For file reading
    
    # Batch processing
    PDF_BATCH_SIZE = 5
    POWERPOINT_BATCH_SIZE = 5
    IMAGE_BATCH_SIZE = 5

    # =====================================================
    # OCR and Image Processing
    # =====================================================
    
    # OCR settings
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "fra+eng")
    IMAGE_OCR_ENHANCEMENT_LEVEL = os.getenv("IMAGE_OCR_ENHANCEMENT_LEVEL", "standard")
    OCR_DPI_THRESHOLD = int(os.getenv("OCR_DPI_THRESHOLD", "300"))
    IMAGE_MIN_CONFIDENCE = 30
    
    # OCR preprocessing
    ENABLE_OCR_PREPROCESSING = os.getenv("ENABLE_OCR_PREPROCESSING", "true").lower() == "true"
    ENABLE_DESKEW = True
    ENABLE_NOISE_REMOVAL = True
    
    # PowerPoint extraction
    EXTRACT_SLIDE_NOTES = True
    EXTRACT_PRESENTATION_METADATA = True

    # =====================================================
    # NLP and Entity Extraction
    # =====================================================
    
    # spaCy model for French/English
    SPACY_MODEL = os.getenv("SPACY_MODEL", "fr_core_news_sm")
    ENABLE_ENTITY_EXTRACTION = os.getenv("ENABLE_ENTITY_EXTRACTION", "true").lower() == "true"
    ENABLE_RELATION_EXTRACTION = os.getenv("ENABLE_RELATION_EXTRACTION", "true").lower() == "true"
    ENTITY_CONFIDENCE_THRESHOLD = float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.8"))

    # =====================================================
    # Knowledge Graph Settings
    # =====================================================
    
    ENABLE_KNOWLEDGE_GRAPH = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
    MAX_RELATIONSHIP_DEPTH = int(os.getenv("MAX_RELATIONSHIP_DEPTH", "3"))
    GRAPH_UPDATE_BATCH_SIZE = int(os.getenv("GRAPH_UPDATE_BATCH_SIZE", "100"))

    # =====================================================
    # Caching and Performance
    # =====================================================
    
    # Semantic cache
    ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
    MAX_CACHE_SIZE_MB = int(os.getenv("MAX_CACHE_SIZE_MB", "500"))
    
    # API rate limiting
    API_RATE_LIMIT_RPM = int(os.getenv("API_RATE_LIMIT_RPM", "60"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", "5"))

    # =====================================================
    # Application Settings
    # =====================================================
    
    # Directories
    UPLOAD_FOLDER = "./data/uploads"
    DATA_FOLDER = "./data"
    LOGS_FOLDER = "./logs"
    
    # Streamlit
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    
    # Development
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    ENABLE_SQL_LOGGING = os.getenv("ENABLE_SQL_LOGGING", "false").lower() == "true"
    ENABLE_PERFORMANCE_METRICS = os.getenv("ENABLE_PERFORMANCE_METRICS", "true").lower() == "true"

    # =====================================================
    # Legacy Settings (for backward compatibility)
    # =====================================================
    
    # Keep old ChromaDB path for migration purposes
    VECTOR_DB_PATH = "./data/chroma_db"
    T_SYSTEMS_MAX_BATCH_SIZE = 128
    MAX_UPLOAD_SIZE_MB = MAX_FILE_SIZE_MB

    # =====================================================
    # Validation and Setup
    # =====================================================
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check API keys
        if not cls.API_KEY:
            errors.append("No API key configured (Azure OpenAI or OpenAI)")
        
        # Check database configuration
        if not all([cls.POSTGRES_HOST, cls.POSTGRES_USER, cls.POSTGRES_PASSWORD, cls.POSTGRES_DB]):
            errors.append("Incomplete PostgreSQL configuration")
        
        # Check embedding dimension consistency
        if cls.EMBEDDING_MODEL == "text-embedding-3-small" and cls.EMBEDDING_DIMENSION != 1536:
            errors.append("Embedding dimension mismatch for text-embedding-3-small (should be 1536)")
        
        return errors
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.DATA_FOLDER,
            cls.LOGS_FOLDER,
            cls.VECTOR_DB_PATH  # Keep for migration
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_database_config(cls) -> dict:
        """Get database configuration for SQLAlchemy"""
        return {
            "url": cls.DATABASE_URL,
            "async_url": cls.ASYNC_DATABASE_URL,
            "pool_size": cls.POSTGRES_MAX_CONNECTIONS,
            "max_overflow": 0,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "echo": cls.ENABLE_SQL_LOGGING
        }
    
    @classmethod
    def get_redis_config(cls) -> dict:
        """Get Redis configuration"""
        config = {
            "host": cls.REDIS_HOST,
            "port": cls.REDIS_PORT,
            "db": cls.REDIS_DB,
            "decode_responses": True
        }
        if cls.REDIS_PASSWORD:
            config["password"] = cls.REDIS_PASSWORD
        return config
    
    @classmethod
    def log_configuration(cls):
        """Log current configuration (without sensitive data)"""
        config_info = {
            "azure_openai_enabled": cls.USE_AZURE_OPENAI,
            "model_name": cls.MODEL_NAME,
            "embedding_model": cls.EMBEDDING_MODEL,
            "postgres_host": cls.POSTGRES_HOST,
            "postgres_db": cls.POSTGRES_DB,
            "redis_enabled": bool(cls.REDIS_HOST),
            "entity_extraction": cls.ENABLE_ENTITY_EXTRACTION,
            "knowledge_graph": cls.ENABLE_KNOWLEDGE_GRAPH,
            "semantic_cache": cls.ENABLE_SEMANTIC_CACHE
        }
        
        logger.info("MCP RAG Configuration:", extra=config_info)

# Initialize and validate configuration
Config.setup_directories()

# Validate configuration on import
config_errors = Config.validate_config()
if config_errors:
    error_msg = "Configuration errors found:\n" + "\n".join(f"- {error}" for error in config_errors)
    logger.error(error_msg)
    if not Config.DEBUG:
        raise ValueError(error_msg)
else:
    Config.log_configuration()
    logger.info("Configuration validated successfully")