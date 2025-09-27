# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (removed libatlas-base-dev)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libblas-dev \
        liblapack-dev \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies with retries to avoid BrokenPipeError
RUN pip install --default-timeout=100 --retries=5 -r requirements.txt

# Copy application code
COPY . .

# Expose port (if using FastAPI/Uvicorn)
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
