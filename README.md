# 🐕 Dog Breed AI: Professional Classifier

[![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen)](https://github.com/dovanminh100104-coder/dog-breed-identification)
[![Model](https://img.shields.io/badge/Model-EfficientNetB0-blue)](https://github.com/dovanminh100104-coder/dog-breed-identification)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange)](https://github.com/dovanminh100104-coder/dog-breed-identification)
[![WebApp](https://img.shields.io/badge/Web_App-Flask-lightgrey)](https://github.com/dovanminh100104-coder/dog-breed-identification)
[![Docker](https://img.shields.io/badge/Docker-Supported-cyan)](https://github.com/dovanminh100104-coder/dog-breed-identification)

> **Tags:** `deep-learning`, `dog-breed-identification`, `flask-web-app`, `computer-vision`, `efficientnet`, `tensorflow-keras`, `image-classification`, `python-ai`, `dockerized-app`.

A production-grade dog breed classification project featuring a highly optimized **EfficientNetB0** model achieving **~90% accuracy** across 120 breeds. This repository includes a full-stack Flask web application with a modern Glassmorphism UI, real-time inference, and Docker containerization.

## 🚀 Key Features

- **High Precision**: Reached 90.14% validation accuracy using Transfer Learning with EfficientNetB0.
- **Premium Web Interface**: Interactive Flask-based UI with drag-and-drop support and sleek glassmorphism design.
- **Real-time Prediction**: Instant feedback showing Top 3 likely breeds with confidence scores.
- **Persistent Logging**: Automatically tracks prediction history in `logs/prediction_history.csv` and displays it on the dashboard.
- **Docker Ready**: Fully containerized environment for consistent deployment.

## 🛠 Tech Stack

- **Deep Learning**: TensorFlow, Keras, EfficientNetB0.
- **Backend**: Python, Flask, Flask-CORS.
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (Async/Await).
- **DevOps**: Docker, multi-layer containerization.
- **Data**: Pandas, NumPy, OpenCV.

## 📂 Project Structure

```text
├── data/               # Dataset labels
├── docs/               # Technical documentation & reports
├── logs/               # Prediction history and system logs
├── models/             # Production-ready .h5 model
├── notebooks/          # Analysis and training notebooks
├── src/                # Core training and evaluation scripts
├── static/             # CSS and JS assets
├── templates/          # HTML frontend
├── Dockerfile          # Container configuration
└── web_app.py          # Main application entry point
```

## 🌐 Getting Started

### 1. Local Setup
```bash
pip install -r requirements.txt
python web_app.py
```
Visit `http://127.0.0.1:5000`

### 2. Docker Setup
```bash
docker build -t dog-breed-app .
docker run -p 5000:5000 dog-breed-app
```

## 📊 Model Performance

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **90.14%** |
| **Architectue** | EfficientNetB0 (Transfer Learning) |
| **Classes** | 120 Dog Breeds |
| **Image Size** | 224 x 224 |

---
*Developed by dovanminh100104-coder*
