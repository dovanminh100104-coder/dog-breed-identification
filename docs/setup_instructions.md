# 🚀 Quick Setup Guide

## For Recruiters/Hiring Managers

Want to quickly test this project? Follow these steps:

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/dog-breed-identification.git
cd dog-breed-identification
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Quick Test (2 minutes)
```bash
cd src
python test_model.py
# Choose option 1 to test with sample images
```

### 3. Train Model (Optional - 30 minutes)
```bash
python final_dog_breed_classifier.py
```

## Expected Results
- **Sample test**: Should show predictions with confidence scores
- **Training**: Should reach ~76% validation accuracy
- **Model size**: ~200MB (DenseNet121 + EfficientNetB3)

## Troubleshooting
- **TensorFlow errors**: Try `pip install tensorflow==2.13.0`
- **Memory issues**: Reduce batch_size in the script
- **Image loading errors**: Check dataset paths

## Project Highlights
✅ **Ensemble Learning**: DenseNet121 + EfficientNetB3  
✅ **76% Accuracy**: On 120 dog breeds  
✅ **Clean Code**: Well-structured, documented  
✅ **Reproducible**: Fixed random seeds  
✅ **Professional**: README, requirements, .gitignore
