FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp1 is required for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
