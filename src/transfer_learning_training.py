"""
Transfer Learning Training - Using pre-trained models for better accuracy
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import time

from config import DATA_DIR, TRAIN_DIR, MODEL_DIR, TRAIN_CSV
from src.logger import logger

class TransferLearningTrainer:
    """Transfer learning trainer using pre-trained models"""
    
    def __init__(self, target_accuracy: float = 0.85):
        self.target_accuracy = target_accuracy
        self.img_size = 224  # Standard size for pre-trained models
        self.batch_size = 32
        self.epochs = 25
        
        logger.info(f"Initialized TransferLearningTrainer with target_accuracy={target_accuracy}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Load and prepare dataset"""
        try:
            logger.info("Loading dataset for transfer learning...")
            
            # Load all labels
            labels_df = pd.read_csv(TRAIN_CSV)
            logger.info(f"Loaded {len(labels_df)} samples")
            
            # Use all 120 breeds
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            
            all_breeds = labels_df['breed'].unique()
            self.label_encoder.fit(all_breeds)
            
            logger.info(f"Created label encoder for {len(all_breeds)} unique breeds")
            
            # Add filename column with .jpg extension
            labels_df['filename'] = labels_df['id'] + '.jpg'
            
            # Split data (80% train, 20% test)
            split_idx = int(len(labels_df) * 0.8)
            train_df = labels_df.iloc[:split_idx]
            test_df = labels_df.iloc[split_idx:]
            
            logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
            
            return train_df, test_df, self.label_encoder
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_transfer_model(self, num_classes: int) -> tf.keras.Model:
        """Create transfer learning model using EfficientNetB0"""
        try:
            logger.info("Creating transfer learning model with EfficientNetB0...")
            
            # Load pre-trained EfficientNetB0
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
            
            # Freeze base model
            base_model.trainable = False
            
            # Create new model
            inputs = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3))
            
            # Preprocessing for EfficientNet
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            
            # Add custom layers
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs, outputs)
            
            # Compile with appropriate optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Created transfer learning model with {model.count_params():,} parameters")
            logger.info(f"Base model trainable: {base_model.trainable}")
            
            return model, base_model
            
        except Exception as e:
            logger.error(f"Error creating transfer model: {str(e)}")
            raise
    
    def create_data_generators(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create data generators for transfer learning"""
        try:
            logger.info("Creating data generators for transfer learning...")
            
            # Data augmentation
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
                rotation_range=20,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
            )
            
            # Create generators
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                directory=TRAIN_DIR,
                x_col='filename',
                y_col='breed',
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                validate_filenames=True
            )
            
            val_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                directory=TRAIN_DIR,
                x_col='filename',
                y_col='breed',
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                validate_filenames=True
            )
            
            test_generator = test_datagen.flow_from_dataframe(
                dataframe=test_df,
                directory=TRAIN_DIR,
                x_col='filename',
                y_col='breed',
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False,
                validate_filenames=True
            )
            
            logger.info(f"Created generators - Train: {len(train_generator)}, Val: {len(val_generator)}, Test: {len(test_generator)}")
            
            return train_generator, val_generator, test_generator
            
        except Exception as e:
            logger.error(f"Error creating generators: {str(e)}")
            raise
    
    def train_model(self, model: tf.keras.Model, base_model, train_gen, val_gen) -> Dict[str, Any]:
        """Train transfer learning model"""
        try:
            logger.info(f"Starting transfer learning training for {self.epochs} epochs...")
            
            # Progress callback
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    self.epoch_start_time = time.time()
                    logger.info(f"\n=== Epoch {epoch + 1}/{self.params['epochs']} ===")
                    print(f"\nEpoch {epoch + 1}/{self.params['epochs']} - Starting...")
                
                def on_epoch_end(self, epoch, logs=None):
                    epoch_time = time.time() - self.epoch_start_time
                    total_time = time.time() - self.training_start_time
                    
                    avg_epoch_time = total_time / (epoch + 1)
                    remaining_epochs = self.params['epochs'] - (epoch + 1)
                    estimated_remaining = avg_epoch_time * remaining_epochs
                    
                    logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
                    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
                    logger.info(f"Estimated remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} minutes)")
                    
                    print(f"  - Time: {epoch_time:.1f}s")
                    print(f"  - Accuracy: {logs.get('accuracy', 0):.4f}")
                    print(f"  - Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
                    print(f"  - Loss: {logs.get('loss', 0):.4f}")
                    print(f"  - Val Loss: {logs.get('val_loss', 0):.4f}")
                    print(f"  - Total elapsed: {total_time/60:.1f} min")
                    print(f"  - ETA: {estimated_remaining/60:.1f} min")
                
                def on_train_begin(self, logs=None):
                    self.training_start_time = time.time()
                    logger.info("Transfer learning training started...")
                    print(f"Training {self.params['epochs']} epochs with transfer learning...")
                    print(f"Batch size: {self.params.get('batch_size', 'N/A')}")
            
            callbacks = [
                ProgressCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    MODEL_DIR / "transfer_learning_best_model.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Phase 1: Train only the top layers
            logger.info("Phase 1: Training top layers...")
            start_time = time.time()
            
            history_phase1 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=10,
                callbacks=callbacks,
                verbose=1
            )
            
            # Phase 2: Fine-tune the base model
            logger.info("Phase 2: Fine-tuning base model...")
            base_model.trainable = True
            
            # Unfreeze the top layers of the base model
            for layer in base_model.layers[-20:]:
                layer.trainable = True
            
            # Re-compile with lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history_phase2 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=self.epochs - 10,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Combine histories
            combined_history = {
                'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
                'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
                'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
                'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
            }
            
            # Get final metrics
            final_accuracy = max(combined_history['val_accuracy'])
            final_loss = min(combined_history['val_loss'])
            
            results = {
                'training_time': training_time,
                'final_accuracy': final_accuracy,
                'final_loss': final_loss,
                'epochs_trained': len(combined_history['accuracy']),
                'history': combined_history
            }
            
            logger.info(f"Training completed in {training_time:.2f}s ({training_time/60:.1f} minutes)")
            logger.info(f"Final validation accuracy: {final_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(self, model: tf.keras.Model, test_gen) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            logger.info("Evaluating model...")
            
            # Predictions
            predictions = model.predict(test_gen, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_gen.classes
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(true_classes, predicted_classes)
            
            # Get class names
            class_names = list(test_gen.class_indices.keys())
            
            # Classification report
            report = classification_report(
                true_classes, 
                predicted_classes, 
                target_names=class_names,
                output_dict=True
            )
            
            results = {
                'test_accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(true_classes, predicted_classes),
                'class_names': class_names
            }
            
            logger.info(f"Test accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model: tf.keras.Model, results: Dict[str, Any]) -> str:
        """Save trained model"""
        try:
            logger.info("Saving model...")
            
            # Create model filename
            model_path = MODEL_DIR / "transfer_learning_trained_model.h5"
            
            # Save model
            model.save(model_path)
            
            # Save results
            import json
            results_path = MODEL_DIR / "transfer_learning_results.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = value
                elif isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Results saved to {results_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def run_transfer_learning_training(self) -> Dict[str, Any]:
        """Run complete transfer learning training pipeline"""
        try:
            logger.info("Starting transfer learning training pipeline...")
            
            # Load data
            train_df, test_df, label_encoder = self.load_and_prepare_data()
            
            # Create model
            num_classes = len(label_encoder.classes_)
            model, base_model = self.create_transfer_model(num_classes)
            
            # Create data generators
            train_gen, val_gen, test_gen = self.create_data_generators(train_df, test_df)
            
            # Train model
            training_results = self.train_model(model, base_model, train_gen, val_gen)
            
            # Evaluate model
            evaluation_results = self.evaluate_model(model, test_gen)
            
            # Save model
            model_path = self.save_model(model, {**training_results, **evaluation_results})
            
            # Final results
            final_results = {
                'model_path': model_path,
                'training': training_results,
                'evaluation': evaluation_results,
                'target_achieved': training_results['final_accuracy'] >= self.target_accuracy
            }
            
            logger.info("Transfer learning training completed!")
            logger.info(f"Target accuracy ({self.target_accuracy}): {'ACHIEVED' if final_results['target_achieved'] else 'NOT ACHIEVED'}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in transfer learning training: {str(e)}")
            raise

def main():
    """Main function"""
    try:
        print("=== Transfer Learning Dog Breed Training ===")
        print("Using EfficientNetB0 pre-trained model for maximum accuracy...")
        print()
        
        # Initialize trainer
        trainer = TransferLearningTrainer(target_accuracy=0.85)
        
        # Run training
        print("Starting transfer learning training...")
        results = trainer.run_transfer_learning_training()
        
        print(f"\nTraining Results:")
        print(f"  Model path: {results['model_path']}")
        print(f"  Training time: {results['training']['training_time']:.2f}s ({results['training']['training_time']/60:.1f} minutes)")
        print(f"  Final accuracy: {results['training']['final_accuracy']:.4f}")
        print(f"  Test accuracy: {results['evaluation']['test_accuracy']:.4f}")
        print(f"  Target achieved: {results['target_achieved']}")
        
        if results['target_achieved']:
            print("\nTransfer learning training successful! Model ready for production.")
        else:
            print("\nTransfer learning training completed. Consider more epochs or different model for better accuracy.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    main()
