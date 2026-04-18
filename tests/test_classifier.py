"""
Unit tests for Dog Breed Classifier
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.final_dog_breed_classifier import DogBreedClassifier
from src.logger import logger
from config import *

class TestDogBreedClassifier(unittest.TestCase):
    """Test cases for DogBreedClassifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.classifier = DogBreedClassifier(image_size=64, batch_size=2)  # Small for testing
        
        # Create mock CSV file
        self.mock_csv = self.temp_dir / "labels.csv"
        mock_data = pd.DataFrame({
            'id': ['001', '002', '003', '004'],
            'breed': ['labrador', 'german_shepherd', 'labrador', 'german_shepherd']
        })
        mock_data.to_csv(self.mock_csv, index=False)
        
        # Create mock image directory
        self.mock_train_dir = self.temp_dir / "train"
        self.mock_train_dir.mkdir()
        
        # Create mock images
        for img_id in ['001', '002', '003', '004']:
            img_path = self.mock_train_dir / f"{img_id}.jpg"
            # Create a simple 64x64 RGB image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertEqual(self.classifier.image_size, 64)
        self.assertEqual(self.classifier.batch_size, 2)
        self.assertIsNone(self.classifier.model)
        self.assertIsNone(self.classifier.history)
    
    def test_load_data_success(self):
        """Test successful data loading"""
        train_df, val_df = self.classifier.load_data(
            csv_path=str(self.mock_csv),
            train_dir=str(self.mock_train_dir)
        )
        
        self.assertEqual(len(train_df), 3)  # 80% of 4 = 3.2 -> 3
        self.assertEqual(len(val_df), 1)    # 20% of 4 = 0.8 -> 1
        self.assertEqual(self.classifier.num_classes, 2)
        self.assertIn('image_file', train_df.columns)
    
    def test_load_data_file_not_found(self):
        """Test data loading with missing CSV file"""
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_data(csv_path="nonexistent.csv")
    
    def test_load_data_invalid_csv(self):
        """Test data loading with invalid CSV structure"""
        invalid_csv = self.temp_dir / "invalid.csv"
        invalid_data = pd.DataFrame({'wrong': ['column']})
        invalid_data.to_csv(invalid_csv, index=False)
        
        with self.assertRaises(ValueError):
            self.classifier.load_data(csv_path=str(invalid_csv))
    
    def test_load_data_missing_train_dir(self):
        """Test data loading with missing train directory"""
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_data(
                csv_path=str(self.mock_csv),
                train_dir="nonexistent_dir"
            )
    
    def test_create_data_generators_success(self):
        """Test successful data generator creation"""
        # First load data
        self.classifier.load_data(
            csv_path=str(self.mock_csv),
            train_dir=str(self.mock_train_dir)
        )
        
        # Then create generators
        self.classifier.create_data_generators()
        
        self.assertIsNotNone(self.classifier.train_generator)
        self.assertIsNotNone(self.classifier.val_generator)
        self.assertEqual(self.classifier.train_generator.batch_size, 2)
        self.assertEqual(self.classifier.val_generator.batch_size, 2)
    
    def test_create_data_generators_no_data(self):
        """Test data generator creation without loading data first"""
        with self.assertRaises(ValueError):
            self.classifier.create_data_generators()
    
    def test_create_ensemble_model_success(self):
        """Test successful ensemble model creation"""
        # Set up required data
        self.classifier.load_data(
            csv_path=str(self.mock_csv),
            train_dir=str(self.mock_train_dir)
        )
        self.classifier.create_data_generators()
        
        # Create model
        model = self.classifier.create_ensemble_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 64, 64, 3))
        self.assertEqual(model.output_shape, (None, 2))  # 2 classes
    
    def test_create_ensemble_model_no_classes(self):
        """Test model creation without setting num_classes"""
        with self.assertRaises(ValueError):
            self.classifier.create_ensemble_model()
    
    def test_predict_breed_no_model(self):
        """Test prediction without loading model"""
        with self.assertRaises(ValueError):
            self.classifier.predict_breed("dummy_path.jpg")
    
    def test_predict_breed_no_generators(self):
        """Test prediction without creating data generators"""
        # Create a dummy model
        self.classifier.model = MagicMock()
        with self.assertRaises(ValueError):
            self.classifier.predict_breed("dummy_path.jpg")
    
    def test_predict_breed_file_not_found(self):
        """Test prediction with missing image file"""
        # Set up minimal requirements
        self.classifier.model = MagicMock()
        self.classifier.val_generator = MagicMock()
        self.classifier.val_generator.class_indices = {'labrador': 0, 'german_shepherd': 1}
        
        with self.assertRaises(FileNotFoundError):
            self.classifier.predict_breed("nonexistent.jpg")
    
    def test_predict_breed_invalid_format(self):
        """Test prediction with invalid image format"""
        # Set up minimal requirements
        self.classifier.model = MagicMock()
        self.classifier.val_generator = MagicMock()
        self.classifier.val_generator.class_indices = {'labrador': 0, 'german_shepherd': 1}
        
        # Create invalid file
        invalid_file = self.temp_dir / "invalid.txt"
        invalid_file.write_text("not an image")
        
        with self.assertRaises(ValueError):
            self.classifier.predict_breed(str(invalid_file))
    
    def test_save_model_no_model(self):
        """Test saving model without having one"""
        with self.assertRaises(ValueError):
            self.classifier.save_model()
    
    def test_load_model_file_not_found(self):
        """Test loading non-existent model"""
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_model("nonexistent.h5")
    
    @patch('tensorflow.keras.models.load_model')
    def test_load_model_success(self, mock_load):
        """Test successful model loading"""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        model_path = self.temp_dir / "test_model.h5"
        model_path.write_text("dummy")  # Create dummy file
        
        self.classifier.load_model(str(model_path))
        
        self.assertEqual(self.classifier.model, mock_model)
        mock_load.assert_called_once_with(str(model_path))

class TestModelUtilities(unittest.TestCase):
    """Test cases for utility functions and configurations"""
    
    def test_config_imports(self):
        """Test that config values are properly imported"""
        from config import IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES, SUPPORTED_FORMATS
        
        self.assertIsInstance(IMAGE_SIZE, int)
        self.assertIsInstance(BATCH_SIZE, int)
        self.assertIsInstance(NUM_CLASSES, int)
        self.assertIsInstance(SUPPORTED_FORMATS, set)
        self.assertIn('.jpg', SUPPORTED_FORMATS)
        self.assertIn('.png', SUPPORTED_FORMATS)
    
    def test_logger_setup(self):
        """Test logger setup"""
        from src.logger import setup_logger
        
        test_logger = setup_logger("test_logger")
        self.assertIsNotNone(test_logger)
        self.assertEqual(test_logger.name, "test_logger")

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.classifier = DogBreedClassifier()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_image_format_validation(self):
        """Test image format validation"""
        # Valid formats
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_file = self.temp_dir / f"test{ext}"
            test_file.write_text("dummy")
            self.assertTrue(test_file.suffix.lower() in SUPPORTED_FORMATS)
        
        # Invalid format
        invalid_file = self.temp_dir / "test.txt"
        invalid_file.write_text("dummy")
        self.assertFalse(invalid_file.suffix.lower() in SUPPORTED_FORMATS)
    
    def test_csv_structure_validation(self):
        """Test CSV structure validation"""
        # Valid CSV
        valid_csv = self.temp_dir / "valid.csv"
        valid_data = pd.DataFrame({
            'id': ['001', '002'],
            'breed': ['labrador', 'german_shepherd']
        })
        valid_data.to_csv(valid_csv, index=False)
        
        df = pd.read_csv(valid_csv)
        self.assertIn('id', df.columns)
        self.assertIn('breed', df.columns)
        
        # Invalid CSV
        invalid_csv = self.temp_dir / "invalid.csv"
        invalid_data = pd.DataFrame({'wrong': ['column']})
        invalid_data.to_csv(invalid_csv, index=False)
        
        df = pd.read_csv(invalid_csv)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('breed', df.columns)

if __name__ == '__main__':
    # Configure test environment
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
    
    # Run tests
    unittest.main(verbosity=2)
