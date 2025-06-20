"""
Validation utilities for the Clerk legal AI system.
Ensures proper case isolation and access control.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

class CaseAccessError(Exception):
    """Raised when case access validation fails"""
    pass

class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

# Allowed case names for the agent (strict whitelist)
ALLOWED_CASE_NAMES = [
    "Cerrtio v Test",
    # Add more cases as needed, but for now only this one is allowed
]

def validate_case_access(case_name: str) -> bool:
    """
    Validate that the requested case name is allowed for access.
    
    Args:
        case_name: The case name to validate
        
    Returns:
        True if access is allowed, False otherwise
        
    Raises:
        CaseAccessError: If access is explicitly denied
    """
    if not case_name:
        logger.error("Empty case name provided for validation")
        raise CaseAccessError("Case name cannot be empty")
    
    # Strip whitespace and normalize
    normalized_case = case_name.strip()
    
    # Check against whitelist
    if normalized_case not in ALLOWED_CASE_NAMES:
        logger.warning(f"Access denied to case: '{normalized_case}'. Allowed cases: {ALLOWED_CASE_NAMES}")
        raise CaseAccessError(f"Access denied to case: '{normalized_case}'")
    
    logger.debug(f"Case access validated: '{normalized_case}'")
    return True

def validate_user_input(user_input: str, max_length: int = 1000) -> str:
    """
    Validate and sanitize user input.
    
    Args:
        user_input: The user's input to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input string
        
    Raises:
        ValidationError: If input is invalid
    """
    if not user_input:
        raise ValidationError("User input cannot be empty")
    
    # Remove potential injection attempts
    sanitized = user_input.strip()
    
    # Check length
    if len(sanitized) > max_length:
        raise ValidationError(f"Input too long. Maximum {max_length} characters allowed.")
    
    # Basic injection protection
    suspicious_patterns = [
        r'<script.*?>',
        r'javascript:',
        r'data:text/html',
        r'eval\s*\(',
        r'exec\s*\(',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(f"Suspicious input detected: {pattern}")
            raise ValidationError("Input contains potentially unsafe content")
    
    return sanitized

def validate_document_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate document metadata structure and content.
    
    Args:
        metadata: Document metadata dictionary
        
    Returns:
        Validated metadata
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary")
    
    required_fields = ['case_name', 'document_id']
    for field in required_fields:
        if field not in metadata:
            raise ValidationError(f"Required metadata field missing: {field}")
    
    # Validate case name
    try:
        validate_case_access(metadata['case_name'])
    except CaseAccessError as e:
        raise ValidationError(f"Invalid case name in metadata: {str(e)}")
    
    # Validate document ID format
    doc_id = metadata['document_id']
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValidationError("Document ID must be a non-empty string")
    
    # Sanitize string fields
    sanitized_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            sanitized_metadata[key] = value.strip()
        else:
            sanitized_metadata[key] = value
    
    return sanitized_metadata

def validate_search_parameters(
    case_name: str,
    limit: int = 10,
    threshold: float = 0.7,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate search parameters for vector queries.
    
    Args:
        case_name: Case name to search
        limit: Maximum results to return
        threshold: Similarity threshold
        filters: Additional search filters
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate case access
    validate_case_access(case_name)
    
    # Validate limit
    if not isinstance(limit, int) or limit < 1 or limit > 100:
        raise ValidationError("Limit must be an integer between 1 and 100")
    
    # Validate threshold
    if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
        raise ValidationError("Threshold must be a number between 0.0 and 1.0")
    
    # Validate filters
    validated_filters = {}
    if filters:
        if not isinstance(filters, dict):
            raise ValidationError("Filters must be a dictionary")
        
        # Sanitize filter values
        for key, value in filters.items():
            if isinstance(value, str):
                validated_filters[key] = value.strip()
            else:
                validated_filters[key] = value
    
    return {
        'case_name': case_name.strip(),
        'limit': limit,
        'threshold': threshold,
        'filters': validated_filters
    }

def validate_api_key(api_key: str, service_name: str = "API") -> bool:
    """
    Basic validation for API keys.
    
    Args:
        api_key: The API key to validate
        service_name: Name of the service for logging
        
    Returns:
        True if API key appears valid
        
    Raises:
        ValidationError: If API key is invalid
    """
    if not api_key:
        raise ValidationError(f"{service_name} API key cannot be empty")
    
    # Basic format checks
    if len(api_key) < 10:
        raise ValidationError(f"{service_name} API key appears too short")
    
    # Don't log actual API keys for security
    logger.debug(f"{service_name} API key validation passed")
    return True

def validate_embedding_vector(vector: List[float], expected_dimension: int = 1536) -> bool:
    """
    Validate embedding vector format and dimensions.
    
    Args:
        vector: The embedding vector to validate
        expected_dimension: Expected vector dimension
        
    Returns:
        True if vector is valid
        
    Raises:
        ValidationError: If vector is invalid
    """
    if not isinstance(vector, list):
        raise ValidationError("Embedding vector must be a list")
    
    if len(vector) != expected_dimension:
        raise ValidationError(f"Vector dimension mismatch. Expected {expected_dimension}, got {len(vector)}")
    
    # Check that all elements are numbers
    for i, value in enumerate(vector):
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Vector element at index {i} must be a number")
        
        # Check for invalid values
        if not (-1.0 <= value <= 1.0):
            logger.warning(f"Vector element at index {i} is outside normal range: {value}")
    
    return True

def validate_query_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate AI agent response structure.
    
    Args:
        response: The response dictionary to validate
        
    Returns:
        Validated response
        
    Raises:
        ValidationError: If response structure is invalid
    """
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")
    
    required_fields = ['answer', 'confidence']
    for field in required_fields:
        if field not in response:
            raise ValidationError(f"Required response field missing: {field}")
    
    # Validate confidence score
    confidence = response['confidence']
    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        raise ValidationError("Confidence must be a number between 0.0 and 1.0")
    
    # Validate answer text
    answer = response['answer']
    if not isinstance(answer, str) or not answer.strip():
        raise ValidationError("Answer must be a non-empty string")
    
    # Validate sources if present
    if 'sources' in response:
        sources = response['sources']
        if not isinstance(sources, list):
            raise ValidationError("Sources must be a list")
        
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ValidationError(f"Source {i} must be a dictionary")
            
            required_source_fields = ['document_id', 'excerpt', 'relevance_score']
            for field in required_source_fields:
                if field not in source:
                    raise ValidationError(f"Source {i} missing required field: {field}")
    
    return response

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem operations.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")
    
    # Remove path separators and dangerous characters
    dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(dangerous_chars, '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not too long
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    if not sanitized:
        raise ValidationError("Filename becomes empty after sanitization")
    
    return sanitized

def log_validation_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Log validation events for audit purposes.
    
    Args:
        event_type: Type of validation event
        details: Event details to log
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'details': details
    }
    
    logger.info(f"Validation event: {event_type}", extra=log_entry)