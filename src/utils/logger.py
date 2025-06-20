"""
Logging utilities for the Clerk legal AI system.
Provides structured logging with security considerations for legal data.
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Configuration
LOG_DIR = Path("logs")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# Sensitive fields that should never be logged
SENSITIVE_FIELDS = {
    'api_key', 'password', 'token', 'secret', 'credential',
    'social_security_number', 'ssn', 'medical_record_number',
    'patient_id', 'dob', 'date_of_birth'
}

def sanitize_log_data(data: Any) -> Any:
    """
    Recursively sanitize data to remove sensitive information.
    
    Args:
        data: Data to sanitize (dict, list, or primitive)
        
    Returns:
        Sanitized data with sensitive fields masked
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    
    elif isinstance(data, str):
        # Basic pattern matching for sensitive data
        if len(data) > 8 and any(pattern in data.lower() for pattern in ['password', 'secret', 'key']):
            return "***REDACTED***"
        return data
    
    else:
        return data

class LegalAIFormatter(logging.Formatter):
    """Custom formatter for legal AI logging with security considerations"""
    
    def format(self, record):
        # Create base log record
        log_record = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        
        if hasattr(record, 'case_name'):
            log_record['case_name'] = record.case_name
        
        if hasattr(record, 'query_id'):
            log_record['query_id'] = record.query_id
        
        if hasattr(record, 'execution_time'):
            log_record['execution_time'] = record.execution_time
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # Sanitize all data
        sanitized_record = sanitize_log_data(log_record)
        
        # Return formatted JSON
        return json.dumps(sanitized_record, ensure_ascii=False)

class CaseIsolationFilter(logging.Filter):
    """Filter to ensure case data doesn't leak between logs"""
    
    def __init__(self, allowed_case: Optional[str] = None):
        super().__init__()
        self.allowed_case = allowed_case
    
    def filter(self, record):
        # If a case name is specified in the log record, validate it
        if hasattr(record, 'case_name') and self.allowed_case:
            if record.case_name != self.allowed_case:
                # Log the violation but don't pass the original record
                security_logger = logging.getLogger('security')
                security_logger.warning(
                    f"Case isolation violation detected: "
                    f"Expected '{self.allowed_case}', got '{record.case_name}'"
                )
                return False
        return True

def setup_logging(
    app_name: str = "clerk_legal_ai",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    allowed_case: Optional[str] = None
) -> logging.Logger:
    """
    Set up comprehensive logging for the legal AI system.
    
    Args:
        app_name: Name of the application
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        allowed_case: Case name for isolation filtering
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    if log_dir is None:
        log_dir = LOG_DIR
    
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with simple format for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # Add case isolation filter if specified
    if allowed_case:
        case_filter = CaseIsolationFilter(allowed_case)
        console_handler.addFilter(case_filter)
    
    logger.addHandler(console_handler)
    
    # File handler with JSON format for production
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{app_name}.log",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = LegalAIFormatter()
    file_handler.setFormatter(file_formatter)
    
    # Add case isolation filter if specified
    if allowed_case:
        case_filter = CaseIsolationFilter(allowed_case)
        file_handler.addFilter(case_filter)
    
    logger.addHandler(file_handler)
    
    # Separate security log
    security_logger = logging.getLogger('security')
    security_handler = logging.handlers.RotatingFileHandler(
        log_dir / "security.log",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    security_handler.setLevel(logging.WARNING)
    security_handler.setFormatter(file_formatter)
    security_logger.addHandler(security_handler)
    
    # Performance log
    performance_logger = logging.getLogger('performance')
    performance_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(file_formatter)
    performance_logger.addHandler(performance_handler)
    
    logger.info(f"Logging initialized for {app_name} at level {log_level}")
    if allowed_case:
        logger.info(f"Case isolation enabled for: {allowed_case}")
    
    return logger

def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get a logger instance with optional extra context.
    
    Args:
        name: Logger name (usually __name__)
        **kwargs: Additional context to include in logs
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Create adapter if extra context provided
    if kwargs:
        adapter = logging.LoggerAdapter(logger, sanitize_log_data(kwargs))
        return adapter
    
    return logger

class QueryLogger:
    """Context manager for logging query execution"""
    
    def __init__(self, logger: logging.Logger, query_id: str, user_id: str, case_name: str):
        self.logger = logger
        self.query_id = query_id
        self.user_id = user_id
        self.case_name = case_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(
            f"Query started",
            extra={
                'query_id': self.query_id,
                'user_id': self.user_id,
                'case_name': self.case_name,
                'start_time': self.start_time.isoformat()
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                f"Query completed successfully",
                extra={
                    'query_id': self.query_id,
                    'user_id': self.user_id,
                    'case_name': self.case_name,
                    'execution_time': execution_time,
                    'end_time': end_time.isoformat()
                }
            )
        else:
            self.logger.error(
                f"Query failed: {exc_val}",
                extra={
                    'query_id': self.query_id,
                    'user_id': self.user_id,
                    'case_name': self.case_name,
                    'execution_time': execution_time,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                }
            )

def log_case_access(logger: logging.Logger, user_id: str, case_name: str, action: str):
    """
    Log case access events for audit purposes.
    
    Args:
        logger: Logger instance
        user_id: ID of the user accessing the case
        case_name: Name of the case being accessed
        action: Action being performed
    """
    logger.info(
        f"Case access: {action}",
        extra={
            'event_type': 'case_access',
            'user_id': user_id,
            'case_name': case_name,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }
    )

def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "WARNING"):
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event
        details: Event details
        severity: Log severity level
    """
    security_logger = logging.getLogger('security')
    log_method = getattr(security_logger, severity.lower())
    
    sanitized_details = sanitize_log_data(details)
    
    log_method(
        f"Security event: {event_type}",
        extra={
            'event_type': 'security',
            'security_event_type': event_type,
            'details': sanitized_details,
            'timestamp': datetime.now().isoformat()
        }
    )

def log_performance_metric(metric_name: str, value: float, unit: str, context: Dict[str, Any] = None):
    """
    Log performance metrics.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        context: Additional context
    """
    performance_logger = logging.getLogger('performance')
    
    performance_logger.info(
        f"Performance metric: {metric_name}",
        extra={
            'event_type': 'performance',
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'context': sanitize_log_data(context or {}),
            'timestamp': datetime.now().isoformat()
        }
    )

# Initialize default logger
default_logger = setup_logging()