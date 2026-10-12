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

# Run as an unprivileged user; /data holds the SQLite database (mount a volume there).
RUN useradd --system --create-home --uid 10001 aerosense \
    && mkdir -p /app/backend/uploads /app/backend/outputs /data \
    && chown -R aerosense /app/backend/uploads /app/backend/outputs /data

ENV PYTHONPATH=/app:/app/backend
ENV MODEL_DIR=/app/models
ENV UPLOAD_DIR=/app/backend/uploads
ENV OUTPUT_DIR=/app/backend/outputs
ENV DATABASE_URL=sqlite:////data/aerosense.db
ENV CORS_ORIGINS=http://localhost:5173

EXPOSE 8000
VOLUME ["/data"]
USER aerosense

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
