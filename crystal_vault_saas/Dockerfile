# Crystal Vault SaaS - Docker Configuration
# 7D mH-Q Quantum-Secure Password Manager

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/vaults

# Expose port
EXPOSE 8000

# Run the application - use shell form to expand $PORT
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
