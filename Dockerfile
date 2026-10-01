FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required for FAISS or similar
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (ignoring errors for certain problematic torch versions in slim, but we let pip resolve it)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start server using standard uvicorn command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
