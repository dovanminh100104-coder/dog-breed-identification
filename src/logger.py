"""
Logging configuration for Dog Breed Identification System
"""
import logging
import sys
from pathlib import Path
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE

def setup_logger(name: str = "dog_breed_classifier") -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    LOG_FILE.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger()
