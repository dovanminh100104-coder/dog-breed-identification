# 🐕 Dog Breed Identification System

## 📖 Giới thiệu

Project xây dựng hệ thống nhận diện 120 giống chó khác nhau sử dụng Deep Learning và Ensemble Learning. Model đạt **76% accuracy** trên validation set với khả năng dự đoán confidence lên đến **99.5%** cho các breed rõ nét.

## 🎯 Mục tiêu

- Phân loại chính xác 120 giống chó từ ảnh
- Áp dụng kỹ thuật Ensemble Learning để tăng accuracy
- Xây dựng pipeline hoàn chỉnh từ training đến prediction
- Tạo interface dễ sử dụng cho người dùng

## 🛠 Công nghệ sử dụng

### Core Technologies
- **Python 3.8+**
- **TensorFlow 2.x** - Deep Learning Framework
- **OpenCV** - Image Processing
- **Scikit-learn** - Machine Learning Utilities

### Deep Learning Models
- **DenseNet121** - Pretrained CNN Architecture
- **EfficientNetB3** - State-of-the-art CNN
- **Ensemble Learning** - Combine multiple models

### Data Processing
- **Pandas** - Data Manipulation
- **NumPy** - Numerical Computing
- **Matplotlib/Seaborn** - Visualization

## 📁 Cấu trúc thư mục

```
Dog Breed Identification/
│
├── src/                          # Source code
│   ├── final_dog_breed_classifier.py    # Main training script
│   └── test_model.py                   # Testing utilities
│
├── models/                       # Trained models
│   ├── final_dog_breed_model.h5         # Best ensemble model
│   └── best_dog_model.h5               # Checkpoint model
│
├── data/                         # Dataset
│   ├── train/                           # Training images (10,222 images)
│   ├── test/                            # Test images
│   └── labels.csv                       # Breed labels
│
├── results/                      # Training results
│   ├── training_history.png              # Accuracy/Loss curves
│   ├── confusion_matrix.png              # Model evaluation
│   └── sample_predictions.png             # Sample predictions
│
├── docs/                         # Documentation
│   ├── README_TEST.md                   # Testing guide
│   └── model_architecture.png            # Model architecture
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## 🚀 Cài đặt và chạy

### 1. Clone repository
```bash
git clone https://github.com/dovanminh-coder/dog-breed-identification.git
cd dog-breed-identification
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Chạy training
```bash
cd src
python final_dog_breed_classifier.py
```

### 5. Test model
```bash
python test_model.py
```

## 📊 Kết quả đạt được

### Model Performance
- **Validation Accuracy**: **76%**
- **Number of Classes**: 120 dog breeds
- **Training Samples**: 8,177 images
- **Validation Samples**: 2,045 images

### Top Predictions Examples
- **Saint Bernard**: 96.82% confidence ✅
- **Great Pyrenees**: 99.52% confidence ✅
- **Bedlington Terrier**: 64.26% confidence

### Training Metrics
- **Best Validation Accuracy**: 76.2%
- **Training Epochs**: 35 (25 + 10 fine-tuning)
- **Model Size**: ~200MB (DenseNet121 + EfficientNetB3)

## 🎮 Cách sử dụng

### 1. Predict với ảnh đơn lẻ
```python
from src.final_dog_breed_classifier import DogBreedClassifier

# Load model
classifier = DogBreedClassifier()
classifier.load_data()
classifier.create_data_generators()
classifier.create_ensemble_model()
classifier.model.load_model('models/final_dog_breed_model.h5')

# Predict
results = classifier.predict_breed('path/to/image.jpg', top_k=3)
for breed, confidence in results:
    print(f"{breed}: {confidence:.2f}%")
```

### 2. Batch testing
```python
from src.test_model import batch_test

# Test toàn bộ thư mục
batch_test('data/test')
```

### 3. Interactive testing
```bash
python src/test_model.py
# Chọn option 1-3 để test
```

## 🔧 Technical Details

### Model Architecture
```
Input (224x224x3)
    ↓
┌─────────────────┬─────────────────┐
│   DenseNet121   │ EfficientNetB3  │
│   (Frozen)      │    (Frozen)     │
└─────────────────┴─────────────────┘
    ↓                    ↓
GlobalAveragePooling   GlobalAveragePooling
    ↓                    ↓
        Concatenate
            ↓
        Dense(1024) + Dropout(0.5)
            ↓
        Dense(512) + Dropout(0.4)
            ↓
        Dense(120) - Softmax
```

### Training Strategy
1. **Phase 1**: Freeze base layers, train classifier heads (25 epochs)
2. **Phase 2**: Fine-tune last 20 layers with lower learning rate (10 epochs)

### Data Augmentation
- Rotation: ±25°
- Width/Height shift: ±20%
- Shear: ±20%
- Zoom: ±20%
- Horizontal flip
- Brightness adjustment: [0.8, 1.2]

## 📈 Evaluation Metrics

### Classification Report (Sample)
```
              precision    recall  f1-score   support

saint_bernard    0.96      1.00      0.98        16
great_pyrenees   0.94      1.00      0.97        22
komondor         0.93      1.00      0.96        13

accuracy                           0.76      2045
macro avg       0.76      0.75      0.75      2045
weighted avg    0.76      0.76      0.75      2045
```

## 🎯 Challenges & Solutions

### Challenges
1. **Imbalanced Dataset**: Some breeds have only 66 samples vs 126 samples
2. **Visual Similarity**: Many breeds look similar (e.g., different terriers)
3. **Image Quality**: Various lighting conditions and angles

### Solutions
1. **Stratified Sampling**: Ensure balanced train/validation split
2. **Ensemble Learning**: Combine multiple CNN architectures
3. **Advanced Data Augmentation**: Increase dataset diversity
4. **Regularization**: L2 regularization and Dropout to prevent overfitting

## 🔮 Future Improvements

1. **More Architectures**: Add ResNet50V2, MobileNetV3 to ensemble
2. **Transfer Learning**: Use breed-specific pretrained models
3. **Attention Mechanisms**: Add attention layers for better feature extraction
4. **Web Interface**: Build Flask/FastAPI web app
5. **Mobile Deployment**: Convert to TensorFlow Lite for mobile

## 👨‍💻 Author

- **Name**: Đỗ Văn Minh
- **Email**: dovanminh100104@gmail.com
- **GitHub**: github.com/dovanminh100104-coder

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset from [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)
- TensorFlow and Keras teams for excellent deep learning frameworks
- OpenCV community for image processing tools

---

**⭐ If you find this project helpful, please give it a star on GitHub!**
