from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import AeroSenseError
from app.db.database import init_db
from app.ml.model_loader import get_model_registry
from app.api.routes import health, audio, predictions, models, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting AeroSense API (env={settings.APP_ENV})")
    init_db()
    # Load models ONCE at startup (singleton) rather than per-request.
    registry = get_model_registry()
    logger.info(
        f"Startup complete. classifier_loaded={registry.classifier_loaded} "
        f"autoencoder_loaded={registry.autoencoder_loaded} device={registry.device}"
    )
    yield


app = FastAPI(
    title="AeroSense API",
    description=(
        "Audio-based acoustic anomaly detection for UAV propeller faults. "
        "Classifier and anomaly detector are real PyTorch models trained on the "
        "TII UAV Propeller Anomaly Audio Dataset (ICASSP 2022). See /api/v1/models "
        "and /api/v1/health/model for real training metadata."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Grad-CAM overlay images (and any other generated output artifacts)
# so the frontend can load them directly as <img src="{API_BASE}{heatmap_path}">.
# Uploaded audio (settings.UPLOAD_DIR) is intentionally NOT mounted here —
# there's no product need to serve raw uploads back, and not exposing them
# avoids turning the upload endpoint into an arbitrary file host.
import os as _os
_os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")


@app.exception_handler(AeroSenseError)
async def aerosense_error_handler(request: Request, exc: AeroSenseError):
    logger.error(f"{exc.code}: {exc.message} (path={request.url.path})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces to clients; log server-side.
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}},
    )


app.include_router(health.router)
app.include_router(audio.router)
app.include_router(predictions.router)
app.include_router(models.router)
app.include_router(analytics.router)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "AeroSense API", "docs": "/docs"}
