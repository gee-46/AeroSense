FROM python:3.12-slim

# libsndfile1 is required by soundfile/librosa to decode audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared preprocessing config (used by both training/ and backend/)
COPY common/ ./common/
COPY backend/ ./backend/

# Real trained checkpoints + registry + feature stats (small: <2MB total).
# The full dataset in data/raw is intentionally NOT copied — see .dockerignore.
COPY models/ ./models/

RUN mkdir -p /app/backend/uploads /app/backend/outputs

ENV PYTHONPATH=/app:/app/backend
ENV MODEL_DIR=/app/models
ENV UPLOAD_DIR=/app/backend/uploads
ENV OUTPUT_DIR=/app/backend/outputs
ENV DATABASE_URL=sqlite:////app/aerosense.db

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
