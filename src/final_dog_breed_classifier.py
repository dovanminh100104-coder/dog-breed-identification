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
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class DogBreedClassifier:
    def __init__(self, image_size=224, batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.encoder = LabelEncoder()
        self.model = None
        self.history = None
        
    def load_data(self, csv_path='labels.csv', train_dir='train'):
        df = pd.read_csv(csv_path)
        df['image_file'] = df['id'].apply(lambda x: x + ".jpg")
        
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['breed']
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.num_classes = len(df['breed'].unique())
        
        return train_df, val_df
    
    def create_data_generators(self):
        train_datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            rescale=1./255
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory='train',
            x_col='image_file',
            y_col='breed',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_datagen.flow_from_dataframe(
            dataframe=self.val_df,
            directory='train',
            x_col='image_file',
            y_col='breed',
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    def create_ensemble_model(self):
        input_shape = (self.image_size, self.image_size, 3)
        inputs = tf.keras.Input(shape=input_shape)
        
        densenet_base = DenseNet121(
            weights='imagenet', include_top=False, input_tensor=inputs
        )
        densenet_base.trainable = False
        densenet_output = GlobalAveragePooling2D()(densenet_base.output)
        densenet_output = BatchNormalization()(densenet_output)
        densenet_output = Dropout(0.4)(densenet_output)
        
        efficientnet_base = EfficientNetB3(
            weights='imagenet', include_top=False, input_tensor=inputs
        )
        efficientnet_base.trainable = False
        efficientnet_output = GlobalAveragePooling2D()(efficientnet_base.output)
        efficientnet_output = BatchNormalization()(efficientnet_output)
        efficientnet_output = Dropout(0.4)(efficientnet_output)
        
        combined = Concatenate()([densenet_output, efficientnet_output])
        
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def train_model(self, epochs=30):
        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor='val_accuracy'),
            ModelCheckpoint('best_dog_model.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, epochs=10):
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-8)
            ],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.val_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.val_generator.classes
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=self.val_generator.class_indices.keys()))
        
        print(f"Final Accuracy: {np.mean(y_pred_classes == y_true):.4f}")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Training History')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_true, y_pred_classes)
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
    
    def predict_breed(self, image_path, top_k=3):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        predictions = self.model.predict(img)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        class_indices = self.val_generator.class_indices
        index_to_breed = {v: k for k, v in class_indices.items()}
        
        results = []
        for idx in top_indices:
            breed = index_to_breed[idx]
            confidence = predictions[idx] * 100
            results.append((breed, confidence))
        
        return results

def main():
    classifier = DogBreedClassifier(image_size=224, batch_size=32)
    
    print("Loading data...")
    train_df, val_df = classifier.load_data()
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Number of breeds: {classifier.num_classes}")
    
    print("\nCreating data generators...")
    classifier.create_data_generators()
    
    print("\nCreating ensemble model...")
    classifier.create_ensemble_model()
    classifier.model.summary()
    
    print("\nTraining model...")
    classifier.train_model(epochs=25)
    
    print("\nFine-tuning...")
    classifier.fine_tune(epochs=10)
    
    print("\nEvaluating model...")
    classifier.evaluate_model()
    
    classifier.model.save('final_dog_breed_model.h5')
    print("Model saved as 'final_dog_breed_model.h5'")
    
    return classifier

if __name__ == "__main__":
    classifier = main()
