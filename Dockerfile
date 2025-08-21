# QKD Failure Detection System Docker Image
FROM python:3.13.7-slim

# Set metadata
LABEL maintainer="Arnav <arnav@example.com>"
LABEL description="QKD Failure Detection System - Research Platform for Quantum Cryptography"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r qkduser && useradd -r -g qkduser qkduser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY demos/ ./demos/
COPY notebooks/ ./notebooks/
COPY plots/ ./plots/
COPY resources/ ./resources/
COPY README.md LICENSE Makefile ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/results

# Set correct permissions
RUN chown -R qkduser:qkduser /app

# Switch to non-root user
USER qkduser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.qkd_simulator; print('QKD System OK')" || exit 1

# Expose port for Jupyter notebooks (if needed)
EXPOSE 8888

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Alternative entry points (can be overridden)
# CMD ["python", "demos/demo_anomaly_detection.py"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
