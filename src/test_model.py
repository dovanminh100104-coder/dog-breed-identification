import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from final_dog_breed_classifier import DogBreedClassifier
import matplotlib.pyplot as plt

def load_trained_model(model_path='final_dog_breed_model.h5'):
    """Load trained model and necessary components"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices from training data
    df = pd.read_csv('labels.csv')
    df['image_file'] = df['id'].apply(lambda x: x + ".jpg")
    
    # Create a temporary classifier to get class indices
    classifier = DogBreedClassifier()
    train_df, val_df = classifier.load_data()
    classifier.create_data_generators()
    
    return model, classifier.val_generator.class_indices

def predict_single_image(image_path, model, class_indices, top_k=3):
    """Predict breed for a single image"""
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert BGR to RGB and resize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    predictions = model.predict(img_batch)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    index_to_breed = {v: k for k, v in class_indices.items()}
    
    results = []
    for idx in top_indices:
        breed = index_to_breed[idx]
        confidence = predictions[idx] * 100
        results.append((breed, confidence))
    
    return results, img_rgb

def test_with_sample_images():
    """Test model with sample images from test directory"""
    print("Loading trained model...")
    model, class_indices = load_trained_model()
    
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
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')][:5]
        test_images = [os.path.join(train_dir, f) for f in train_files]
    
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
