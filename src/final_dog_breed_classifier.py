import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from config import *
from src.logger import logger

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class DogBreedClassifier:
    def __init__(self, image_size: int = IMAGE_SIZE, batch_size: int = BATCH_SIZE):
        """
        Initialize Dog Breed Classifier
        
        Args:
            image_size: Input image size
            batch_size: Batch size for training
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.train_df = None
        self.val_df = None
        self.num_classes = NUM_CLASSES
        
        logger.info(f"Initialized DogBreedClassifier with image_size={image_size}, batch_size={batch_size}")
        
    def load_data(self, csv_path: str = str(TRAIN_CSV), train_dir: str = str(TRAIN_DIR)) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split data for training
        
        Args:
            csv_path: Path to labels CSV file
            train_dir: Path to training images directory
            
        Returns:
            Tuple of (train_df, val_df)
            
        Raises:
            FileNotFoundError: If CSV file or train directory doesn't exist
            ValueError: If CSV file is malformed
        """
        try:
            # Validate inputs
            csv_path = Path(csv_path)
            train_dir = Path(train_dir)
            
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            if not train_dir.exists():
                raise FileNotFoundError(f"Train directory not found: {train_dir}")
            
            logger.info(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Validate CSV structure
            if 'id' not in df.columns or 'breed' not in df.columns:
                raise ValueError("CSV must contain 'id' and 'breed' columns")
            
            df['image_file'] = df['id'].apply(lambda x: f"{x}.jpg")
            
            # Check if images exist
            missing_images = []
            for img_file in df['image_file'].head(10):  # Check first 10 images
                img_path = train_dir / img_file
                if not img_path.exists():
                    missing_images.append(img_file)
            
            if missing_images:
                logger.warning(f"Some images not found: {missing_images[:5]}...")
            
            # Split data with stratification
            train_df, val_df = train_test_split(
                df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['breed']
            )
            
            self.train_df = train_df
            self.val_df = val_df
            self.num_classes = len(df['breed'].unique())
            
            logger.info(f"Loaded {len(df)} samples across {self.num_classes} breeds")
            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
            
            return train_df, val_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_data_generators(self) -> None:
        """
        Create data generators for training and validation
        
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        try:
            if self.train_df is None or self.val_df is None:
                raise ValueError("Data must be loaded before creating generators")
            
            logger.info("Creating data generators")
            
            # Training data generator with augmentation
            train_datagen = ImageDataGenerator(**AUGMENTATION)
            
            # Validation data generator (only rescaling)
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            self.train_generator = train_datagen.flow_from_dataframe(
                dataframe=self.train_df,
                directory=str(TRAIN_DIR),
                x_col='image_file',
                y_col='breed',
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=RANDOM_SEED
            )
            
            self.val_generator = val_datagen.flow_from_dataframe(
                dataframe=self.val_df,
                directory=str(TRAIN_DIR),
                x_col='image_file',
                y_col='breed',
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            logger.info(f"Created generators - Train: {self.train_generator.samples}, Val: {self.val_generator.samples}")
            
        except Exception as e:
            logger.error(f"Error creating data generators: {str(e)}")
            raise
    
    def create_ensemble_model(self) -> Model:
        """
        Create ensemble model with DenseNet121 and EfficientNetB3
        
        Returns:
            Compiled Keras model
            
        Raises:
            ValueError: If num_classes is not set
        """
        try:
            if self.num_classes is None:
                raise ValueError("Number of classes must be set before creating model")
            
            logger.info(f"Creating ensemble model for {self.num_classes} classes")
            
            input_shape = (self.image_size, self.image_size, 3)
            inputs = tf.keras.Input(shape=input_shape)
            
            # DenseNet121 branch
            densenet_base = DenseNet121(
                weights='imagenet', include_top=False, input_tensor=inputs
            )
            densenet_base.trainable = False
            densenet_output = GlobalAveragePooling2D()(densenet_base.output)
            densenet_output = BatchNormalization()(densenet_output)
            densenet_output = Dropout(DROPOUT_RATE)(densenet_output)
            
            # EfficientNetB3 branch
            efficientnet_base = EfficientNetB3(
                weights='imagenet', include_top=False, input_tensor=inputs
            )
            efficientnet_base.trainable = False
            efficientnet_output = GlobalAveragePooling2D()(efficientnet_base.output)
            efficientnet_output = BatchNormalization()(efficientnet_output)
            efficientnet_output = Dropout(DROPOUT_RATE)(efficientnet_output)
            
            # Combine branches
            combined = Concatenate()([densenet_output, efficientnet_output])
            
            # Dense layers
            x = Dense(DENSE_UNITS[0], activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(combined)
            x = BatchNormalization()(x)
            x = Dropout(FINE_TUNE_DROPOUT_RATE)(x)
            
            x = Dense(DENSE_UNITS[1], activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
            x = BatchNormalization()(x)
            x = Dropout(DROPOUT_RATE)(x)
            
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            logger.info("Ensemble model created successfully")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {str(e)}")
            raise
    
    def train_model(self, epochs: int = TRAIN_EPOCHS) -> tf.keras.callbacks.History:
        """
        Train the model with callbacks
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training history
            
        Raises:
            ValueError: If model or generators are not ready
        """
        try:
            if self.model is None:
                raise ValueError("Model must be created before training")
            if self.train_generator is None or self.val_generator is None:
                raise ValueError("Data generators must be created before training")
            
            logger.info(f"Starting training for {epochs} epochs")
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    patience=EARLY_STOPPING_PATIENCE, 
                    restore_best_weights=True, 
                    monitor='val_accuracy'
                ),
                ReduceLROnPlateau(
                    factor=0.5, 
                    patience=REDUCE_LR_PATIENCE, 
                    min_lr=MIN_LR, 
                    monitor='val_accuracy'
                ),
                ModelCheckpoint(
                    str(BEST_MODEL_PATH), 
                    save_best_only=True, 
                    monitor='val_accuracy'
                )
            ]
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            self.history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            return self.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def fine_tune(self, epochs: int = FINE_TUNE_EPOCHS) -> tf.keras.callbacks.History:
        """
        Fine-tune the model by unfreezing last layers
        
        Args:
            epochs: Number of fine-tuning epochs
            
        Returns:
            Fine-tuning history
            
        Raises:
            ValueError: If model is not trained yet
        """
        try:
            if self.model is None:
                raise ValueError("Model must be created before fine-tuning")
            if self.history is None:
                raise ValueError("Model must be trained before fine-tuning")
            
            logger.info(f"Starting fine-tuning for {epochs} epochs")
            
            # Unfreeze last 20 layers
            for layer in self.model.layers[-20:]:
                layer.trainable = True
            
            # Re-compile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fine-tune with callbacks
            self.history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=[
                    EarlyStopping(patience=FINE_TUNE_PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=FINE_TUNE_MIN_LR)
                ],
                verbose=1
            )
            
            logger.info("Fine-tuning completed successfully")
            return self.history
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def evaluate_model(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate model performance and generate metrics
        
        Args:
            save_plots: Whether to save evaluation plots
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If model or validation data is not available
        """
        try:
            if self.model is None:
                raise ValueError("Model must be trained before evaluation")
            if self.val_generator is None:
                raise ValueError("Validation generator must be created before evaluation")
            
            logger.info("Starting model evaluation")
            
            # Get predictions
            y_pred = self.model.predict(self.val_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = self.val_generator.classes
            
            # Calculate metrics
            accuracy = np.mean(y_pred_classes == y_true)
            
            # Classification report
            class_names = list(self.val_generator.class_indices.keys())
            report = classification_report(
                y_true, y_pred_classes, 
                target_names=class_names, 
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred_classes)
            
            # Print results
            print("\n" + "="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred_classes, target_names=class_names))
            print("="*50)
            
            # Plot results if requested
            if save_plots:
                self._plot_evaluation_results(cm)
            
            # Return metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'predicted_classes': y_pred_classes,
                'true_classes': y_true
            }
            
            logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _plot_evaluation_results(self, confusion_mat: np.ndarray) -> None:
        """
        Plot training history and confusion matrix
        
        Args:
            confusion_mat: Confusion matrix array
        """
        try:
            RESULTS_DIR.mkdir(exist_ok=True)
            
            if self.history is not None:
                # Plot training history
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(self.history.history['accuracy'], label='Train Accuracy')
                plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
                plt.title('Training History - Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 3, 2)
                plt.plot(self.history.history['loss'], label='Train Loss')
                plt.plot(self.history.history['val_loss'], label='Val Loss')
                plt.title('Training History - Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 3, 3)
                sns.heatmap(confusion_mat, annot=False, cmap='Blues', fmt='d')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            logger.info(f"Evaluation plots saved to {RESULTS_DIR}")
            
        except Exception as e:
            logger.warning(f"Error plotting evaluation results: {str(e)}")
    
    def predict_breed(self, image_path: str, top_k: int = TOP_K_PREDICTIONS) -> List[Tuple[str, float]]:
        """
        Predict dog breed from image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (breed, confidence) tuples
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If model is not loaded
        """
        try:
            if self.model is None:
                raise ValueError("Model must be loaded before prediction")
            if self.val_generator is None:
                raise ValueError("Data generators must be created before prediction")
            
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check file format
            if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            logger.info(f"Predicting breed for {image_path}")
            
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img)[0]
            
            # Get top-k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            class_indices = self.val_generator.class_indices
            index_to_breed = {v: k for k, v in class_indices.items()}
            
            results = []
            for idx in top_indices:
                breed = index_to_breed[idx]
                confidence = predictions[idx] * 100
                if confidence >= CONFIDENCE_THRESHOLD * 100:
                    results.append((breed, confidence))
            
            logger.info(f"Prediction completed - Top result: {results[0] if results else 'No confident prediction'}")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def save_model(self, model_path: str = str(FINAL_MODEL_PATH)) -> None:
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
            
        Raises:
            ValueError: If model is not trained
        """
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            model_path = Path(model_path)
            model_path.parent.mkdir(exist_ok=True)
            
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = str(FINAL_MODEL_PATH)) -> None:
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def main() -> DogBreedClassifier:
    """
    Main training pipeline
    
    Returns:
        Trained classifier instance
    """
    try:
        # Initialize classifier
        classifier = DogBreedClassifier(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
        
        print("\n" + "="*60)
        print("🐕 DOG BREED CLASSIFICATION TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        print("\n📁 Loading data...")
        train_df, val_df = classifier.load_data()
        print(f"   ✓ Training samples: {len(train_df):,}")
        print(f"   ✓ Validation samples: {len(val_df):,}")
        print(f"   ✓ Number of breeds: {classifier.num_classes}")
        
        # Create data generators
        print("\n🔄 Creating data generators...")
        classifier.create_data_generators()
        print("   ✓ Data generators created successfully")
        
        # Create model
        print("\n🏗️ Creating ensemble model...")
        classifier.create_ensemble_model()
        print(f"   ✓ Model created with {classifier.model.count_params():,} parameters")
        
        # Train model
        print(f"\n🚀 Training model for {TRAIN_EPOCHS} epochs...")
        classifier.train_model(epochs=TRAIN_EPOCHS)
        
        # Fine-tune
        print(f"\n🔧 Fine-tuning model for {FINE_TUNE_EPOCHS} epochs...")
        classifier.fine_tune(epochs=FINE_TUNE_EPOCHS)
        
        # Evaluate
        print("\n📊 Evaluating model...")
        metrics = classifier.evaluate_model(save_plots=True)
        
        # Save model
        print("\n💾 Saving model...")
        classifier.save_model()
        print(f"   ✓ Model saved to {FINAL_MODEL_PATH}")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📈 Final Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print("="*60)
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    classifier = main()
