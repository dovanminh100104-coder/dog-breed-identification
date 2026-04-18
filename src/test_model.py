import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.final_dog_breed_classifier import DogBreedClassifier
from config import *
from src.logger import logger

class ModelTester:
    """
    Utility class for testing trained dog breed classifier
    """
    
    def __init__(self, model_path: str = str(MODEL_DIR / "transfer_learning_best_model.h5")):
        """
        Initialize model tester
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.classifier = None
        self.model = None
        self.class_indices = None
        
        logger.info(f"Initialized ModelTester with model: {model_path}")
    
    def load_model_and_components(self) -> None:
        """
        Load trained model and necessary components
        
        Raises:
            FileNotFoundError: If model or data files don't exist
        """
        try:
            # Check if model exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Check if labels CSV exists
            if not TRAIN_CSV.exists():
                raise FileNotFoundError(f"Labels CSV not found: {TRAIN_CSV}")
            
            logger.info("Loading model and components...")
            
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Create a temporary classifier to get class indices
            self.classifier = DogBreedClassifier()
            train_df, val_df = self.classifier.load_data()
            self.classifier.create_data_generators()
            self.class_indices = self.classifier.val_generator.class_indices
            
            logger.info(f"Loaded {len(self.class_indices)} class indices")
            
        except Exception as e:
            logger.error(f"Error loading model and components: {str(e)}")
            raise
    
    def predict_single_image(self, image_path: str, top_k: int = TOP_K_PREDICTIONS) -> Optional[List[Tuple[str, float]]]:
        """
        Predict breed for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (breed, confidence) tuples or None if error
        """
        try:
            image_path = Path(image_path)
            
            # Validate inputs
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            if self.model is None or self.class_indices is None:
                raise ValueError("Model not loaded. Call load_model_and_components() first.")
            
            logger.info(f"Predicting breed for {image_path}")
            
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB and resize
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_batch)[0]
            
            # Get top-k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            index_to_breed = {v: k for k, v in self.class_indices.items()}
            
            results = []
            for idx in top_indices:
                breed = index_to_breed[idx]
                confidence = predictions[idx] * 100
                if confidence >= CONFIDENCE_THRESHOLD * 100:
                    results.append((breed, confidence))
            
            logger.info(f"Prediction completed - Top result: {results[0] if results else 'No confident prediction'}")
            return results
            
        except Exception as e:
            logger.error(f"Error predicting single image: {str(e)}")
            return None
    
    def batch_test_directory(self, test_dir: str, max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Test all images in a directory
        
        Args:
            test_dir: Path to test directory
            max_images: Maximum number of images to test (None for all)
            
        Returns:
            Dictionary with test results
        """
        try:
            test_dir = Path(test_dir)
            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
            
            # Get all image files
            image_files = []
            for ext in SUPPORTED_FORMATS:
                image_files.extend(test_dir.glob(f"*{ext}"))
                image_files.extend(test_dir.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.warning(f"No images found in {test_dir}")
                return {'results': [], 'total_images': 0, 'successful_predictions': 0}
            
            if max_images:
                image_files = image_files[:max_images]
            
            logger.info(f"Testing {len(image_files)} images in {test_dir}")
            
            results = []
            successful_predictions = 0
            
            for image_file in tqdm(image_files, desc="Processing images"):
                prediction = self.predict_single_image(str(image_file))
                if prediction:
                    results.append({
                        'image': str(image_file),
                        'predictions': prediction
                    })
                    successful_predictions += 1
                else:
                    results.append({
                        'image': str(image_file),
                        'predictions': [],
                        'error': 'Prediction failed'
                    })
            
            summary = {
                'results': results,
                'total_images': len(image_files),
                'successful_predictions': successful_predictions,
                'success_rate': successful_predictions / len(image_files) * 100
            }
            
            logger.info(f"Batch testing completed - Success rate: {summary['success_rate']:.2f}%")
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch testing: {str(e)}")
            return {'results': [], 'total_images': 0, 'successful_predictions': 0, 'error': str(e)}
    
    def visualize_predictions(self, image_path: str, save_path: Optional[str] = None) -> None:
        """
        Visualize prediction results for an image
        
        Args:
            image_path: Path to image file
            save_path: Path to save visualization (optional)
        """
        try:
            predictions = self.predict_single_image(image_path)
            if not predictions:
                print("No predictions to visualize")
                return
            
            # Load image for display
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Display image
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title('Input Image')
            plt.axis('off')
            
            # Display predictions
            plt.subplot(1, 2, 2)
            breeds = [pred[0].replace('_', ' ').title() for pred in predictions]
            confidences = [pred[1] for pred in predictions]
            
            colors = ['green' if conf > 70 else 'orange' if conf > 40 else 'red' for conf in confidences]
            bars = plt.barh(breeds, confidences, color=colors)
            plt.xlabel('Confidence (%)')
            plt.title('Top Predictions')
            plt.xlim(0, 100)
            
            # Add confidence values on bars
            for bar, conf in zip(bars, confidences):
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.1f}%', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing predictions: {str(e)}")
            print(f"Error visualizing predictions: {str(e)}")

def interactive_test_menu():
    """
    Interactive menu for testing model
    """
    try:
        print("\n" + "="*60)
        print("🐕 DOG BREED CLASSIFIER - INTERACTIVE TESTING")
        print("="*60)
        
        # Initialize tester
        tester = ModelTester()
        
        # Load model
        print("\n📁 Loading model...")
        tester.load_model_and_components()
        print("   ✓ Model loaded successfully")
        
        while True:
            print("\n" + "-"*40)
            print("Choose an option:")
            print("1. Test single image")
            print("2. Batch test directory")
            print("3. Visualize predictions")
            print("4. Exit")
            print("-"*40)
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                if image_path:
                    predictions = tester.predict_single_image(image_path)
                    if predictions:
                        print("\n🎯 Predictions:")
                        for i, (breed, confidence) in enumerate(predictions, 1):
                            print(f"   {i}. {breed.replace('_', ' ').title()}: {confidence:.2f}%")
                    else:
                        print("❌ Prediction failed")
                else:
                    print("❌ Please enter a valid image path")
            
            elif choice == '2':
                test_dir = input("Enter test directory path: ").strip()
                max_images = input("Enter max images (press Enter for all): ").strip()
                max_images = int(max_images) if max_images.isdigit() else None
                
                if test_dir:
                    results = tester.batch_test_directory(test_dir, max_images)
                    print(f"\n📊 Batch Test Results:")
                    print(f"   Total images: {results['total_images']}")
                    print(f"   Successful predictions: {results['successful_predictions']}")
                    print(f"   Success rate: {results['success_rate']:.2f}%")
                    
                    # Show first few results
                    for i, result in enumerate(results['results'][:5]):
                        if 'predictions' in result and result['predictions']:
                            top_pred = result['predictions'][0]
                            print(f"   {Path(result['image']).name}: {top_pred[0]} ({top_pred[1]:.1f}%)")
                else:
                    print("❌ Please enter a valid directory path")
            
            elif choice == '3':
                image_path = input("Enter image path: ").strip()
                if image_path:
                    save_path = input("Enter save path (optional, press Enter to skip): ").strip()
                    save_path = save_path if save_path else None
                    tester.visualize_predictions(image_path, save_path)
                else:
                    print("❌ Please enter a valid image path")
            
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Error in interactive test menu: {str(e)}")
        print(f"\n❌ Error: {str(e)}")

def main():
    """
    Main testing function
    """
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "interactive":
            interactive_test_menu()
        
        elif command == "single" and len(sys.argv) > 2:
            image_path = sys.argv[2]
            tester = ModelTester()
            tester.load_model_and_components()
            predictions = tester.predict_single_image(image_path)
            
            if predictions:
                print("\n🎯 Predictions:")
                for i, (breed, confidence) in enumerate(predictions, 1):
                    print(f"   {i}. {breed.replace('_', ' ').title()}: {confidence:.2f}%")
            else:
                print("❌ Prediction failed")
        
        elif command == "batch" and len(sys.argv) > 2:
            test_dir = sys.argv[2]
            max_images = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
            
            tester = ModelTester()
            tester.load_model_and_components()
            results = tester.batch_test_directory(test_dir, max_images)
            
            print(f"\n📊 Batch Test Results:")
            print(f"   Total images: {results['total_images']}")
            print(f"   Successful predictions: {results['successful_predictions']}")
            print(f"   Success rate: {results['success_rate']:.2f}%")
        
        else:
            print("Usage:")
            print("  python test_model.py interactive  # Interactive menu")
            print("  python test_model.py single <image_path>  # Test single image")
            print("  python test_model.py batch <test_dir> [max_images]  # Batch test")
    
    else:
        # Default to interactive mode
        interactive_test_menu()

if __name__ == "__main__":
    main()
    
    # Test with images from test directory
    test_dir = 'test'
    test_images = []
    
    # Get first 5 test images
    import os
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:5]
        test_images = [os.path.join(test_dir, f) for f in test_files]
    else:
        print("Test directory not found. Using training images for demo...")
        # Use some training images instead
        train_dir = 'train'
        train_files = [f for f in os.listdir(str(TRAIN_DIR)) if f.endswith('.jpg')][:5]
        test_images = [os.path.join(str(TRAIN_DIR), f) for f in train_files]
    
    # Test each image
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(test_images):
        results, original_img = predict_single_image(image_path, model, class_indices)
        
        if results:
            plt.subplot(2, 3, i+1)
            plt.imshow(original_img)
            plt.title(f'Top 3 Predictions:\n1. {results[0][0]} ({results[0][1]:.1f}%)\n'
                     f'2. {results[1][0]} ({results[1][1]:.1f}%)\n'
                     f'3. {results[2][0]} ({results[2][1]:.1f}%)')
            plt.axis('off')
            
            print(f"\nImage {i+1}: {os.path.basename(image_path)}")
            for j, (breed, confidence) in enumerate(results, 1):
                print(f"  {j}. {breed}: {confidence:.2f}%")
    
    plt.tight_layout()
    plt.show()

def test_custom_image(image_path):
    """Test with a custom image path"""
    print(f"Testing custom image: {image_path}")
    
    model, class_indices = load_trained_model()
    results, original_img = predict_single_image(image_path, model, class_indices)
    
    if results:
        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(f'Predictions for {os.path.basename(image_path)}')
        plt.axis('off')
        plt.show()
        
        print("\nTop Predictions:")
        for i, (breed, confidence) in enumerate(results, 1):
            print(f"{i}. {breed}: {confidence:.2f}%")
    else:
        print("Failed to process image")

def interactive_test():
    """Interactive testing mode"""
    print("=== Dog Breed Classifier - Interactive Test ===")
    print("1. Test with sample images")
    print("2. Test with custom image path")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            test_with_sample_images()
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if image_path:
                test_custom_image(image_path)
            else:
                print("Please enter a valid image path")
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def batch_test(test_dir='test'):
    """Test all images in test directory and show statistics"""
    print(f"Running batch test on {test_dir}...")
    
    model, class_indices = load_trained_model()
    
    import os
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found!")
        return
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"Found {len(test_files)} test images")
    
    all_predictions = []
    confidence_scores = []
    
    for i, filename in enumerate(test_files):
        image_path = os.path.join(test_dir, filename)
        results, _ = predict_single_image(image_path, model, class_indices)
        
        if results:
            top_breed, top_confidence = results[0]
            all_predictions.append((filename, top_breed, top_confidence))
            confidence_scores.append(top_confidence)
            
            if i < 10:  # Show first 10 results
                print(f"{filename}: {top_breed} ({top_confidence:.1f}%)")
    
    # Statistics
    if confidence_scores:
        avg_confidence = np.mean(confidence_scores)
        high_confidence = sum(1 for c in confidence_scores if c > 80)
        medium_confidence = sum(1 for c in confidence_scores if 60 <= c <= 80)
        low_confidence = sum(1 for c in confidence_scores if c < 60)
        
        print(f"\n=== Test Statistics ===")
        print(f"Total images tested: {len(all_predictions)}")
        print(f"Average confidence: {avg_confidence:.2f}%")
        print(f"High confidence (>80%): {high_confidence} ({high_confidence/len(all_predictions)*100:.1f}%)")
        print(f"Medium confidence (60-80%): {medium_confidence} ({medium_confidence/len(all_predictions)*100:.1f}%)")
        print(f"Low confidence (<60%): {low_confidence} ({low_confidence/len(all_predictions)*100:.1f}%)")

if __name__ == "__main__":
    print("Dog Breed Classifier - Testing Suite")
    print("====================================")
    
    # Check if model exists
    if not os.path.exists('final_dog_breed_model.h5'):
        print("Error: Model file 'final_dog_breed_model.h5' not found!")
        print("Please run the training script first.")
    else:
        interactive_test()
