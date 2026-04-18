# API Documentation

## Overview

The Dog Breed Classification API provides RESTful endpoints for classifying dog breeds from images using deep learning models. The API is built with FastAPI and supports single image prediction, batch processing, and model management.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production use, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

## Endpoints

### 1. Root Endpoint

**GET /**

Returns basic API information and available endpoints.

**Response:**
```json
{
  "message": "Dog Breed Classification API",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. Health Check

**GET /health**

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 3. Model Information

**GET /model/info**

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_loaded": true,
  "num_classes": 120,
  "image_size": 224,
  "model_path": "models/final_dog_breed_model.h5"
}
```

### 4. Single Image Prediction

**POST /predict**

Classify a single dog breed image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "breed": "Golden Retriever",
      "confidence": 85.67
    },
    {
      "breed": "Labrador Retriever", 
      "confidence": 12.34
    },
    {
      "breed": "Flat-Coated Retriever",
      "confidence": 1.99
    }
  ]
}
```

**Error Response:**
```json
{
  "detail": "File must be an image"
}
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog_image.jpg"
```

### 5. Batch Prediction

**POST /predict/batch**

Classify multiple images in a single request.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `files[]` (array of image files, max 10 files)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "breed": "German Shepherd",
      "confidence": 92.45
    },
    {
      "filename": "image2.jpg",
      "success": true,
      "breed": "Siberian Husky",
      "confidence": 87.23
    }
  ],
  "total_files": 2,
  "successful_predictions": 2
}
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 6. Model Reload

**POST /model/reload**

Reload the model (useful after retraining).

**Response:**
```json
{
  "message": "Model reload initiated"
}
```

### 7. View Logs

**GET /logs?lines=100**

Get recent log entries for debugging.

**Query Parameters:**
- `lines` (optional): Number of recent log lines to retrieve (default: 100)

**Response:**
```json
{
  "logs": ["2024-01-01 10:00:00 - INFO - Model loaded successfully"],
  "total_lines": 1500,
  "showing": 100
}
```

### 8. API Metrics

**GET /metrics**

Get basic API performance metrics.

**Response:**
```json
{
  "model_loaded": true,
  "api_version": "1.0.0",
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
  "max_batch_size": 10,
  "confidence_threshold": 0.1
}
```

## Response Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input, wrong file type) |
| 404 | Not Found (model file missing) |
| 422 | Unprocessable Entity (validation error) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

## Error Handling

All errors return a JSON response with a `detail` field:

```json
{
  "detail": "Error description"
}
```

Common error scenarios:
- Invalid file format
- Corrupted image file
- Model not loaded
- File too large
- Server overload

## Rate Limiting

Currently not implemented. Consider adding:
- Requests per minute per IP
- Concurrent request limits
- Queue management for high load

## Performance Considerations

### Request Size Limits
- Maximum file size: 10MB per image
- Maximum batch size: 10 images per request
- Timeout: 30 seconds per request

### Optimization Tips
1. Use appropriate image sizes (224x224 recommended)
2. Compress images before upload
3. Use batch processing for multiple images
4. Implement client-side caching for repeated predictions

## SDK Examples

### Python

```python
import requests

# Single prediction
def predict_dog_breed(image_path):
    url = "http://localhost:8000/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    return response.json()

# Batch prediction
def predict_batch(image_paths):
    url = "http://localhost:8000/predict/batch"
    
    files = [('files', open(path, 'rb')) for path in image_paths]
    response = requests.post(url, files=files)
    
    # Close files
    for _, file in files:
        file.close()
    
    return response.json()
```

### JavaScript

```javascript
// Single prediction
async function predictDogBreed(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Batch prediction
async function predictBatch(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch('/predict/batch', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}
```

## Web Interface

The API includes an interactive web interface:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These provide:
- Interactive API documentation
- Request/response examples
- Test interface for all endpoints

## Deployment

### Docker

```bash
# Build image
docker build -t dog-breed-classifier .

# Run container
docker run -p 8000:8000 dog-breed-classifier
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# Scale service
docker-compose up -d --scale dog-breed-classifier=3
```

### Production Configuration

For production deployment, consider:

1. **Environment Variables**
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=INFO
export WORKERS=4
```

2. **Reverse Proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **SSL Configuration**
```bash
# Use certbot for Let's Encrypt
certbot --nginx -d your-domain.com
```

## Monitoring

### Health Monitoring

Monitor these endpoints:
- `/health` - Service health
- `/metrics` - Performance metrics
- `/logs` - Error logs

### Key Metrics

Track:
- Request latency
- Error rates
- Model prediction confidence
- Memory usage
- CPU usage

### Alerting

Set up alerts for:
- High error rates (>5%)
- Slow response times (>2 seconds)
- Model not loaded
- Service downtime

## Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Check if model file exists in `models/`
   - Verify model format compatibility
   - Check logs for detailed error messages

2. **Memory Issues**
   - Reduce batch size
   - Use smaller images
   - Increase available memory

3. **Slow Response Times**
   - Use optimized model
   - Enable GPU acceleration
   - Implement caching

4. **File Upload Errors**
   - Check file format support
   - Verify file size limits
   - Ensure proper multipart form data

### Debug Mode

Enable debug logging:
```python
# In config.py
LOG_LEVEL = "DEBUG"
```

### Log Analysis

Check logs for:
- Error patterns
- Performance bottlenecks
- Usage statistics

## Versioning

API versioning follows semantic versioning:
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

Current version: `1.0.0`

## Security Considerations

### Current Limitations
- No authentication
- No rate limiting
- No input sanitization beyond image validation

### Recommended Improvements
1. **Authentication**
   - API keys
   - JWT tokens
   - OAuth 2.0

2. **Rate Limiting**
   - Per-IP limits
   - Per-user limits
   - Global limits

3. **Input Validation**
   - File type verification
   - Size limits
   - Malware scanning

4. **HTTPS**
   - SSL/TLS encryption
   - Secure headers
   - Certificate management

## Support

For issues and questions:
1. Check the logs at `/logs`
2. Review API documentation at `/docs`
3. Test with the interactive interface
4. Check the health endpoint status

## License

This API is part of the Dog Breed Identification project, licensed under the MIT License.
