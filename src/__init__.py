"""
Dog Breed Identification System - Source Package

A comprehensive deep learning system for classifying 120 dog breeds
using ensemble learning and modern software engineering practices.
"""

__version__ = "2.0.0"
__author__ = "Đỗ Văn Minh"
__email__ = "dovanminh100104@gmail.com"
__description__ = "Deep Learning system for classifying 120 dog breeds with ensemble learning"

# Core ML components
from .final_dog_breed_classifier import DogBreedClassifier
from .test_model import ModelTester

# Utilities
from .logger import logger, setup_logger

# Version and metadata
__all__ = [
    # Core classes
    "DogBreedClassifier",
    "ModelTester",
    
    # Utilities
    "logger",
    "setup_logger",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]

# Package metadata for setup tools
__title__ = "Dog Breed Identification System"
__url__ = "https://github.com/dovanminh100104-coder/dog-breed-identification"
__license__ = "MIT"

# Optional imports for extended functionality
try:
    from . import config
    __all__.append("config")
except ImportError:
    pass
