"""
Configuration settings for the Clerk legal AI system.
All sensitive values should be set via environment variables.
"""

import os
from typing import Dict, Any
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
class DatabaseConfig:
    """Database configuration for document tracking and deduplication"""
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = "text-embedding-3-small"
    context_model: str = os.getenv("CONTEXT_LLM_MODEL", "gpt-3.5-turbo")

@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    target_chunk_size: int = 1100
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
    collection_name: str = "case_documents"
    embedding_dimensions: int = 1536  # for text-embedding-3-small

@dataclass
class CostConfig:
    """API cost tracking configuration"""
    enable_tracking: bool = True
    save_reports: bool = True
    report_directory: str = "logs"
    # Custom pricing overrides (uses defaults if not specified)
    custom_pricing: Dict[str, Any] = None
    
class Settings:
    """Central settings manager"""
    def __init__(self):
        self.box = BoxConfig()
        self.database = DatabaseConfig()
        self.openai = OpenAIConfig()
        self.chunking = ChunkingConfig()
        self.processing = ProcessingConfig()
        self.vector = VectorConfig()
        self.cost = CostConfig()
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate that all required settings are present"""
        required_settings = [
            (self.box.client_id, "BOX_CLIENT_ID"),
            (self.box.client_secret, "BOX_CLIENT_SECRET"),
            (self.database.supabase_url, "SUPABASE_URL"),
            (self.database.supabase_key, "SUPABASE_KEY"),
            (self.openai.api_key, "OPENAI_API_KEY"),
        ]
        
        missing = [name for value, name in required_settings if not value]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please check your .env file."
            )

# Global settings instance
settings = Settings()