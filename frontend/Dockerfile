FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for Pillow and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "front.py", "--server.port=8501", "--server.address=0.0.0.0"]
