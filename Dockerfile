# Dockerfile for Potato Disease Classifier
# Single-stage build for Render.com compatibility

FROM python:3.12-slim

WORKDIR /app

# Install ONLY system dependencies needed by TensorFlow (CPU)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Copy application code
COPY . .

# Create directories for models, uploads and training data
RUN mkdir -p models uploads training_data/early_blight training_data/late_blight training_data/healthy

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
