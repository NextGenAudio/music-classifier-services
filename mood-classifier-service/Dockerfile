# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and ML libraries
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libfftw3-dev \
    pkg-config \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for sample tracks if it doesn't exist
RUN mkdir -p sampletracks

# Expose the port the app runs on
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/docs || exit 1

# Run the application
CMD ["python", "main.py"]