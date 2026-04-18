# Development Guide

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup and Installation](#setup-and-installation)
3. [Code Architecture](#code-architecture)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [API Development](#api-development)
7. [Model Training](#model-training)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

## Project Structure

```
dog-breed-identification-main/
├── src/                          # Source code
│   ├── final_dog_breed_classifier.py    # Main training script
│   ├── test_model.py                   # Testing utilities
│   ├── api_server.py                   # FastAPI web server
│   ├── model_optimizer.py              # Model optimization utilities
│   ├── evaluation_metrics.py           # Advanced evaluation metrics
│   ├── data_validator.py               # Data validation pipeline
│   └── logger.py                       # Logging configuration
├── tests/                       # Unit tests
│   ├── test_classifier.py              # Main test suite
│   └── __init__.py                     # Test package
├── config.py                    # Configuration file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker compose setup
├── run_tests.py                 # Test runner
└── README.md                    # Project documentation
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerized deployment)

### Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/dovanminh100104-coder/dog-breed-identification.git
cd dog-breed-identification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create necessary directories**
```bash
mkdir -p data/train data/test models results logs
```

5. **Download dataset**
- Download the Kaggle Dog Breed Identification dataset
- Extract training images to `data/train/`
- Place `labels.csv` in `data/`

### Docker Setup

1. **Build Docker image**
```bash
docker build -t dog-breed-classifier .
```

2. **Run with Docker Compose**
```bash
docker-compose up -d
```

## Code Architecture

### Core Components

#### 1. DogBreedClassifier (`src/final_dog_breed_classifier.py`)
Main training class with the following key methods:
- `load_data()`: Load and validate training data
- `create_data_generators()`: Create data generators with augmentation
- `create_ensemble_model()`: Build DenseNet121 + EfficientNetB3 ensemble
- `train_model()`: Train with callbacks and early stopping
- `fine_tune()`: Fine-tune last layers with lower learning rate
- `evaluate_model()`: Comprehensive evaluation with metrics
- `predict_breed()`: Single image prediction

#### 2. ModelTester (`src/test_model.py`)
Testing utilities with:
- Single image prediction
- Batch testing
- Visualization of results
- Interactive testing menu

#### 3. APIServer (`src/api_server.py`)
FastAPI web server providing:
- RESTful API endpoints
- File upload handling
- Model health checks
- Batch prediction support

#### 4. ModelOptimizer (`src/model_optimizer.py`)
Model optimization utilities:
- TensorFlow Lite conversion
- Model quantization
- Pruning (optional)
- Performance benchmarking

#### 5. AdvancedEvaluator (`src/evaluation_metrics.py`)
Comprehensive evaluation metrics:
- Per-class precision/recall/F1
- ROC curves and AUC
- Error analysis
- Confidence analysis
- Visualization tools

#### 6. DataValidator (`src/data_validator.py`)
Data validation pipeline:
- Dataset structure validation
- CSV validation
- Image file validation
- Consistency checking
- Quality analysis

### Configuration System

All configuration is centralized in `config.py`:
- Model hyperparameters
- File paths
- Training parameters
- API settings
- Logging configuration

### Logging System

Comprehensive logging setup in `src/logger.py`:
- File and console logging
- Configurable log levels
- Structured log format
- Error tracking

## Development Workflow

### 1. Model Development

```bash
# Train new model
python src/final_dog_breed_classifier.py

# Test model interactively
python src/test_model.py

# Run comprehensive evaluation
python src/evaluation_metrics.py
```

### 2. API Development

```bash
# Start API server
python src/api_server.py

# Or with uvicorn directly
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Model Optimization

```bash
# Optimize and quantize model
python src/model_optimizer.py
```

### 4. Data Validation

```bash
# Validate dataset
python src/data_validator.py
```

### 5. Testing

```bash
# Run all tests
python run_tests.py

# Run specific test
python -m pytest tests/test_classifier.py -v
```

## Testing

### Test Structure

Tests are organized in the `tests/` directory:
- `test_classifier.py`: Unit tests for core classifier functionality
- Mock data generation for isolated testing
- Error handling validation
- Configuration testing

### Running Tests

```bash
# Run all tests with verbose output
python run_tests.py

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_classifier.py -v
```

### Test Coverage

The test suite covers:
- Model initialization
- Data loading and validation
- Model creation and training
- Prediction functionality
- Error handling
- Configuration loading

## API Development

### Available Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `GET /model/info`: Model information
- `POST /predict`: Single image prediction
- `POST /predict/batch`: Batch prediction
- `POST /model/reload`: Reload model
- `GET /logs`: View recent logs
- `GET /metrics`: API metrics

### API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Testing API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

## Model Training

### Training Pipeline

1. **Data Preparation**
   - Validate dataset integrity
   - Check image quality
   - Ensure proper labeling

2. **Model Configuration**
   - Adjust hyperparameters in `config.py`
   - Set up data augmentation
   - Configure callbacks

3. **Training Execution**
   - Run training script
   - Monitor progress via logs
   - Check evaluation metrics

4. **Model Evaluation**
   - Comprehensive metrics analysis
   - Error analysis
   - Performance visualization

### Hyperparameter Tuning

Key hyperparameters in `config.py`:
- `IMAGE_SIZE`: Input image dimensions
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Initial learning rate
- `TRAIN_EPOCHS`: Number of training epochs
- `DROPOUT_RATE`: Dropout regularization
- `AUGMENTATION`: Data augmentation parameters

### Monitoring Training

Training progress is logged to:
- Console output
- Log files in `logs/`
- Training history plots in `results/`

## Deployment

### Local Deployment

```bash
# Start API server
python src/api_server.py

# Access API at http://localhost:8000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale the service
docker-compose up -d --scale dog-breed-classifier=3
```

### Production Considerations

1. **Model Optimization**
   - Use TensorFlow Lite for mobile
   - Apply quantization
   - Consider model pruning

2. **API Security**
   - Add authentication
   - Rate limiting
   - Input validation

3. **Monitoring**
   - Health checks
   - Performance metrics
   - Error tracking

4. **Scalability**
   - Load balancing
   - Horizontal scaling
   - Caching strategies

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```
Error: Model file not found
```
**Solution**: Ensure model is trained and saved to `models/final_dog_breed_model.h5`

#### 2. Data Loading Issues
```
Error: CSV file not found or malformed
```
**Solution**: Check `data/labels.csv` exists and has proper format

#### 3. Memory Issues
```
Error: Resource exhausted
```
**Solution**: Reduce batch size in `config.py` or use smaller image size

#### 4. API Connection Issues
```
Error: Connection refused
```
**Solution**: Check if API server is running on correct port

#### 5. Docker Issues
```
Error: Build failed
```
**Solution**: Check Dockerfile and ensure all dependencies are properly installed

### Debug Mode

Enable debug logging by setting in `config.py`:
```python
LOG_LEVEL = "DEBUG"
```

### Performance Issues

1. **Slow Training**
   - Reduce image size
   - Use smaller batch size
   - Enable mixed precision training

2. **Slow API Response**
   - Use optimized model
   - Enable model caching
   - Consider GPU acceleration

### Getting Help

1. Check logs in `logs/` directory
2. Run data validation to check dataset integrity
3. Review API documentation at `/docs`
4. Check test results for potential issues

## Best Practices

### Code Quality
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write comprehensive docstrings
- Add error handling for all operations

### Model Development
- Validate data before training
- Use proper train/validation split
- Monitor for overfitting
- Save model checkpoints

### API Development
- Validate all inputs
- Handle errors gracefully
- Provide meaningful error messages
- Include proper HTTP status codes

### Deployment
- Use environment variables for configuration
- Implement health checks
- Monitor performance metrics
- Use containerization for consistency

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Test API endpoints manually

## License

This project is licensed under the MIT License - see the LICENSE file for details.
