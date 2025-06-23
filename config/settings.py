"""
Configuration settings for the Clerk legal AI system.
All sensitive values should be set via environment variables.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class BoxConfig:
    """Box API configuration"""
    client_id: str = os.getenv("BOX_CLIENT_ID", "")
    client_secret: str = os.getenv("BOX_CLIENT_SECRET", "")
    enterprise_id: str = os.getenv("BOX_ENTERPRISE_ID", "")
    jwt_key_id: str = os.getenv("BOX_JWT_KEY_ID", "")
    private_key: str = os.getenv("BOX_PRIVATE_KEY", "").replace("\\n", "\n")
    passphrase: str = os.getenv("BOX_PASSPHRASE", "")

@dataclass
class QdrantConfig:
    """Qdrant vector database configuration"""
    host: str = os.getenv("QDRANT_HOST", "qdrant")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    https: bool = os.getenv("QDRANT_HTTPS", "false").lower() == "true"
    timeout: int = int(os.getenv("QDRANT_TIMEOUT", "60"))
    prefer_grpc: bool = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
    
    # Collection configuration
    collection_name: str = "legal_documents"
    hybrid_collection_name: str = "legal_documents_hybrid"
    registry_collection_name: str = "document_registry"  # For deduplication
    
    # Performance settings
    batch_size: int = 500
    max_workers: int = 16
    
    # Vector configuration
    embedding_dimensions: int = 1536  # OpenAI text-embedding-3-small
    
    @property
    def url(self) -> str:
        """Build Qdrant URL"""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"

@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = "text-embedding-3-small"
    context_model: str = os.getenv("CONTEXT_LLM_MODEL", "gpt-3.5-turbo")

@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    target_chunk_size: int = 1200  # Increased for better context
    chunk_variance: int = 100  # +/- 100 characters
    overlap_size: int = 200
    min_chunk_size: int = 500
    max_chunk_size: int = 1500

@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    max_file_size_mb: int = 200
    supported_extensions: tuple = (".pdf",)
    
@dataclass
class VectorConfig:
    """Vector database configuration"""
    collection_name: str = "legal_documents"
    embedding_dimensions: int = 1536  # for text-embedding-3-small
    
    # Qdrant-specific settings
    distance_metric: str = "cosine"
    hnsw_m: int = 32  # Higher for better accuracy with legal documents
    hnsw_ef_construct: int = 200
    quantization_enabled: bool = True
    quantization_type: str = "scalar"  # or "binary" for extreme compression
    quantization_quantile: float = 0.95
    quantization: bool = True

@dataclass
class CostConfig:
    """API cost tracking configuration"""
    enable_tracking: bool = True
    save_reports: bool = True
    report_directory: str = "logs"
    # Custom pricing overrides (uses defaults if not specified)
    custom_pricing: Dict[str, Any] = None
    
class Settings:
    """Main settings class that aggregates all configurations"""
    
    def __init__(self):
        self.box = BoxConfig()
        self.qdrant = QdrantConfig()
        self.openai = OpenAIConfig()
        self.chunking = ChunkingConfig()
        self.processing = ProcessingConfig()
        self.vector = VectorConfig()
        self.cost = CostConfig()
        
        # Legal-specific settings
        self.legal = {
            "enable_case_isolation": True,
            "enable_citation_tracking": True,
            "enable_deadline_detection": True,
            "enable_hybrid_search": True,
            "hybrid_search_weights": {
                "vector": 0.7,
                "keyword": 0.2,
                "citation": 0.1
            }
        }
    
    def __str__(self):
        return (
            f"Settings(\n"
            f"  box={self.box},\n"
            f"  qdrant={self.qdrant},\n"
            f"  openai={self.openai},\n"
            f"  chunking={self.chunking},\n"
            f"  processing={self.processing},\n"
            f"  vector={self.vector},\n"
            f"  cost={self.cost},\n"
            f"  legal={self.legal}\n"
            f")"
        )
    
    def validate(self) -> bool:
        """Validate all required settings are present"""
        errors = []
        
        # Check Box settings
        if not all([self.box.client_id, self.box.client_secret]):
            errors.append("Box API credentials missing")
        
        # Check Qdrant settings
        if not self.qdrant.host:
            errors.append("Qdrant host not configured")
            
        # Check OpenAI settings
        if not self.openai.api_key:
            errors.append("OpenAI API key missing")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True

# Create singleton instance
settings = Settings()