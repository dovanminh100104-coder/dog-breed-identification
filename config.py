"""
Configuration file for Dog Breed Identification System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data paths
TRAIN_CSV = DATA_DIR / "labels.csv"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Model paths
BEST_MODEL_PATH = MODEL_DIR / "best_dog_model.h5"
FINAL_MODEL_PATH = MODEL_DIR / "final_dog_breed_model.h5"
TFLITE_MODEL_PATH = MODEL_DIR / "dog_breed_model.tflite"

# Model Architecture
DENSENET121_INPUT_SHAPE = (224, 224, 3)
EFFICIENTNETB3_INPUT_SHAPE = (224, 224, 3)

# Training Parameters
TRAIN_EPOCHS = 25
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Accuracy Improvement Parameters
ADVANCED_TRAIN_EPOCHS = 35
WARMUP_EPOCHS = 5
COSINE_ANNEALING = True
LABEL_SMOOTHING = 0.1
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROBABILITY = 0.5
CUTMIX_PROBABILITY = 0.3
ADVANCED_BATCH_SIZE = 16  # Smaller batch for advanced techniques
PROGRESSIVE_RESIZING = True
START_IMAGE_SIZE = 128
END_IMAGE_SIZE = 224
RESIZING_EPOCHS = 10

# Ensemble Parameters
ENSEMBLE_WEIGHTS = {
    'densenet121': 0.4,
    'efficientnetb3': 0.4,
    'vision_transformer': 0.2
}
WEIGHTED_VOTING = True
DYNAMIC_WEIGHTS = True

# Data Augmentation Parameters
AUGMENTATION_STRENGTH = 'advanced'
AUTOAUGMENT_POLICY = 'v0'
RANDAUGMENT_N = 2
RANDAUGMENT_M = 9
GAUSSIAN_NOISE_STD = 0.1
BLUR_KERNEL_SIZE = 3
COARSE_DROPOUT_MAX_HOLES = 8
COARSE_DROPOUT_MAX_HEIGHT = 16
COARSE_DROPOUT_MAX_WIDTH = 16
COARSE_DROPOUT_PROBABILITY = 0.3

# Optimization Parameters
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MIXED_PRECISION = True
ACCUMULATION_STEPS = 2

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001
REDUCE_LR_PATIENCE = 8
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# Model hyperparameters
IMAGE_SIZE = 224
NUM_CLASSES = 120

# Data augmentation parameters
AUGMENTATION = {
    "rotation_range": 25,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
    "rescale": 1./255
}

# Model architecture parameters
DROPOUT_RATE = 0.4
FINE_TUNE_DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.01
DENSE_UNITS = [1024, 512]

# Callback parameters
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
FINE_TUNE_PATIENCE = 5
MIN_LR = 1e-7
FINE_TUNE_MIN_LR = 1e-8

# Prediction parameters
TOP_K_PREDICTIONS = 3
CONFIDENCE_THRESHOLD = 0.1

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "dog_breed_classifier.log"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# Docker configuration
DOCKER_IMAGE_NAME = "dog-breed-classifier"
DOCKER_TAG = "latest"

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Random seeds for reproducibility
RANDOM_SEED = 42
