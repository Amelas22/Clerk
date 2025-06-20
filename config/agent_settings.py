"""
Configuration settings for the Legal Document AI Agent.
Extends the main settings with agent-specific configurations.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from config.settings import settings

class AgentSettings(BaseModel):
    """Configuration for the Legal Document AI Agent"""
    
    # Core Agent Configuration
    agent_name: str = Field(
        default="Clerk Legal AI",
        description="Name of the AI agent"
    )
    
    agent_version: str = Field(
        default="1.0.0",
        description="Version of the agent"
    )
    
    # Case Access Control
    allowed_case_names: List[str] = Field(
        default=["Cerrtio v Test"],
        description="List of case names this agent is allowed to access"
    )
    
    default_case_name: str = Field(
        default="Cerrtio v Test",
        description="Default case name for the agent"
    )
    
    # AI Model Configuration
    primary_model: str = Field(
        default="gpt-4o-mini",
        description="Primary AI model for the agent"
    )
    
    fallback_model: str = Field(
        default="gpt-3.5-turbo",
        description="Fallback model if primary fails"
    )
    
    model_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for AI model responses"
    )
    
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens for AI responses"
    )
    
    # Search and Retrieval Configuration
    default_search_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of search results to retrieve"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for search results"
    )
    
    max_context_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of document chunks to include in AI context"
    )
    
    # Response Configuration
    min_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for responses"
    )
    
    max_sources_in_response: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum number of sources to include in responses"
    )
    
    include_confidence_scores: bool = Field(
        default=True,
        description="Whether to include confidence scores in responses"
    )
    
    enable_follow_up_questions: bool = Field(
        default=True,
        description="Whether to generate suggested follow-up questions"
    )
    
    # Security and Validation
    require_user_authentication: bool = Field(
        default=True,
        description="Whether to require user authentication for queries"
    )
    
    enable_case_isolation: bool = Field(
        default=True,
        description="Whether to enforce strict case isolation"
    )
    
    log_all_queries: bool = Field(
        default=True,
        description="Whether to log all user queries for audit"
    )
    
    # Performance Configuration
    query_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout for query processing in seconds"
    )
    
    max_concurrent_queries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent queries to process"
    )
    
    enable_query_caching: bool = Field(
        default=True,
        description="Whether to enable query result caching"
    )
    
    cache_ttl_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Cache time-to-live in minutes"
    )
    
    # OpenWebUI Integration
    openwebui_function_name: str = Field(
        default="legal_document_chat",
        description="Name of the OpenWebUI function"
    )
    
    enable_citations: bool = Field(
        default=True,
        description="Whether to emit citation events in OpenWebUI"
    )
    
    enable_status_updates: bool = Field(
        default=True,
        description="Whether to emit status update events"
    )
    
    # Legal-Specific Configuration
    citation_format: str = Field(
        default="bluebook",
        description="Legal citation format to use (bluebook, etc.)"
    )
    
    include_legal_disclaimers: bool = Field(
        default=True,
        description="Whether to include legal disclaimers in responses"
    )
    
    default_disclaimers: List[str] = Field(
        default=[
            "This AI analysis is for informational purposes only and does not constitute legal advice.",
            "All information is based solely on the documents in this specific case file.",
            "Please consult with a qualified attorney for legal guidance."
        ],
        description="Default legal disclaimers to include in responses"
    )
    
    # Error Handling
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts for failed operations"
    )
    
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay between retry attempts in seconds"
    )
    
    graceful_degradation: bool = Field(
        default=True,
        description="Whether to provide partial responses when some operations fail"
    )
    
    @validator('allowed_case_names')
    def validate_case_names(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("allowed_case_names must be a non-empty list")
        return v
    
    @validator('default_case_name')
    def validate_default_case(cls, v, values):
        if 'allowed_case_names' in values and v not in values['allowed_case_names']:
            raise ValueError("default_case_name must be in allowed_case_names")
        return v
    
    class Config:
        env_prefix = "AGENT_"
        case_sensitive = False

# Load agent settings from environment
def load_agent_settings() -> AgentSettings:
    """Load agent settings from environment variables"""
    
    # Override with environment variables if present
    env_overrides = {}
    
    # Case configuration
    if os.getenv("AGENT_ALLOWED_CASES"):
        env_overrides["allowed_case_names"] = os.getenv("AGENT_ALLOWED_CASES").split(",")
    
    if os.getenv("AGENT_DEFAULT_CASE"):
        env_overrides["default_case_name"] = os.getenv("AGENT_DEFAULT_CASE")
    
    # Model configuration
    if os.getenv("AGENT_PRIMARY_MODEL"):
        env_overrides["primary_model"] = os.getenv("AGENT_PRIMARY_MODEL")
    
    if os.getenv("AGENT_TEMPERATURE"):
        env_overrides["model_temperature"] = float(os.getenv("AGENT_TEMPERATURE"))
    
    # Security configuration
    if os.getenv("AGENT_REQUIRE_AUTH"):
        env_overrides["require_user_authentication"] = os.getenv("AGENT_REQUIRE_AUTH").lower() == "true"
    
    # Performance configuration
    if os.getenv("AGENT_TIMEOUT"):
        env_overrides["query_timeout_seconds"] = int(os.getenv("AGENT_TIMEOUT"))
    
    return AgentSettings(**env_overrides)

# Global agent settings instance
agent_settings = load_agent_settings()

# Validation function for runtime checks
def validate_agent_configuration() -> Dict[str, Any]:
    """
    Validate the agent configuration and return status.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "configuration": agent_settings.dict()
    }
    
    # Check OpenAI API key
    if not settings.openai.api_key:
        validation_results["errors"].append("OpenAI API key not configured")
        validation_results["valid"] = False
    
    # Check Qdrant configuration
    if not settings.qdrant.host or not settings.qdrant.port:
        validation_results["errors"].append("Qdrant configuration incomplete")
        validation_results["valid"] = False
    
    # Check case name configuration
    if not agent_settings.allowed_case_names:
        validation_results["errors"].append("No allowed case names configured")
        validation_results["valid"] = False
    
    # Warnings for performance
    if agent_settings.max_tokens > 4000:
        validation_results["warnings"].append("High max_tokens may impact performance")
    
    if agent_settings.similarity_threshold < 0.5:
        validation_results["warnings"].append("Low similarity threshold may return irrelevant results")

    return validation_results

# Example usage and testing
if __name__ == "__main__":
    print("Agent Configuration:")
    print(f"Agent Name: {agent_settings.agent_name}")
    print(f"Allowed Cases: {agent_settings.allowed_case_names}")
    print(f"Default Case: {agent_settings.default_case_name}")
    print(f"Primary Model: {agent_settings.primary_model}")
    print(f"Authentication Required: {agent_settings.require_user_authentication}")
    
    print("\nValidation Results:")
    validation = validate_agent_configuration()
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")