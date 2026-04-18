import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'models/transfer_learning_best_model.h5'
LABELS_PATH = 'data/labels.csv'
HISTORY_PATH = 'logs/prediction_history.csv'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

def init_history():
    """Initialize history CSV if it doesn't exist"""
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'breed', 'confidence', 'top_3'])

init_history()

def log_prediction(breed, confidence, top_3):
    """Log prediction result to CSV"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(HISTORY_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, breed, f"{confidence:.1f}%", str(top_3)])
    except Exception as e:
        print(f"Error logging prediction: {e}")

# Global variables for model and classes
model = None
breed_classes = []

def load_resources():
    global model, breed_classes
    print("Loading model and labels...")
    try:
        # Load Labels
        labels_df = pd.read_csv(LABELS_PATH)
        breed_classes = sorted(list(labels_df['breed'].unique()))
        
        # Load Model
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading resources: {e}")

# Initial load
load_resources()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    """Get the last 10 predictions from the history CSV"""
    try:
        if not os.path.exists(HISTORY_PATH):
            return jsonify([])
        
        df = pd.read_csv(HISTORY_PATH)
        # Get last 10 entries and reverse to show newest first
        recent = df.tail(10).iloc[::-1]
        return jsonify(recent.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload.jpg')
        file.save(filepath)
        
        try:
            # Preprocess
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            preds = model.predict(img_array, verbose=0)
            
            # Get Top 3 results
            top_indices = np.argsort(preds[0])[-3:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    'breed': breed_classes[idx].replace('_', ' ').title(),
                    'confidence': float(preds[0][idx] * 100)
                })
            
            # Log prediction
            log_prediction(results[0]['breed'], results[0]['confidence'], results)

            return jsonify({
                'success': True,
                'prediction': results[0]['breed'],
                'confidence': results[0]['confidence'],
                'top_3': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
