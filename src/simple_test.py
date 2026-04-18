"""
Simple Test Script - Easy way to test the trained dog breed model
"""
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import TRAIN_CSV, MODEL_DIR

def load_model_and_labels():
    """Load the trained model and breed labels"""
    try:
        # Load the best model
        model_path = MODEL_DIR / "transfer_learning_best_model.h5"
        print(f"Loading model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Load breed labels
        labels_df = pd.read_csv(TRAIN_CSV)
        breed_labels = sorted(labels_df['breed'].unique())
        print(f"✅ Loaded {len(breed_labels)} breed labels")
        
        return model, breed_labels
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_breed(model, breed_labels, image_path):
    """Predict dog breed from image"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return None, None
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_breeds = [breed_labels[i] for i in top_3_indices]
        top_3_confidences = [predictions[0][i] for i in top_3_indices]
        
        return top_3_breeds, top_3_confidences
        
    except Exception as e:
        print(f"❌ Error predicting: {str(e)}")
        return None, None

def test_with_sample_images():
    """Test model with sample images from dataset"""
    print("\n=== Testing with Sample Images ===")
    
    model, breed_labels = load_model_and_labels()
    if model is None:
        return
    
    # Get some sample images from dataset
    train_dir = Path("data/train")
    sample_images = list(train_dir.glob("*.jpg"))[:5]  # Test first 5 images
    
    if not sample_images:
        print("❌ No images found in data/train/")
        return
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n--- Test {i}: {img_path.name} ---")
        
        # Predict
        breeds, confidences = predict_breed(model, breed_labels, img_path)
        
        if breeds is None:
            continue
        
        # Display results
        print("🐕 Top 3 Predictions:")
        for j, (breed, conf) in enumerate(zip(breeds, confidences), 1):
            print(f"   {j}. {breed}: {conf:.2%}")
        
        # Show image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {breeds[0]} ({confidences[0]:.1%})")
        plt.axis('off')
        plt.show()

def test_custom_image(image_path):
    """Test model with a custom image"""
    print(f"\n=== Testing Custom Image: {image_path} ===")
    
    model, breed_labels = load_model_and_labels()
    if model is None:
        return
    
    # Check if image exists
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Predict
    breeds, confidences = predict_breed(model, breed_labels, img_path)
    
    if breeds is None:
        return
    
    # Display results
    print("🐕 Top 3 Predictions:")
    for i, (breed, conf) in enumerate(zip(breeds, confidences), 1):
        print(f"   {i}. {breed}: {conf:.2%}")
    
    # Show image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {breeds[0]} ({confidences[0]:.1%})")
    plt.axis('off')
    plt.show()

def interactive_test():
    """Interactive testing mode"""
    print("\n=== Interactive Dog Breed Testing ===")
    print("1. Test with sample images from dataset")
    print("2. Test with custom image path")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            test_with_sample_images()
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            test_custom_image(image_path)
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def main():
    """Main function"""
    print("🐕 Dog Breed Classifier - Simple Test Tool")
    print("=" * 50)
    
    # Check if model exists
    model_path = MODEL_DIR / "transfer_learning_best_model.h5"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please make sure you have trained the model first!")
        return
    
    print(f"✅ Found model: {model_path}")
    print(f"📊 Model accuracy: 79.22%")
    print(f"🎯 Number of breeds: 120")
    
    # Start interactive test
    interactive_test()

if __name__ == "__main__":
    main()
