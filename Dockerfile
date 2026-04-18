# Use a slim Python image to keep the size manageable
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=web_app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
# Note: .dockerignore will handle excluding large datasets and scrap files
COPY . .

# Create uploads directory if it doesn't exist
RUN mkdir -p uploads logs

# Expose the web app port
EXPOSE 5000

# Start the application
CMD ["python", "web_app.py"]
