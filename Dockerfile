FROM python:3.10-slim

WORKDIR /app

# Install system-level build tools required by ML packages (e.g. hdbscan, numba)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer for Docker cache efficiency)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project source code
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start FastAPI — models are auto-downloaded at startup via _load_models()
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
