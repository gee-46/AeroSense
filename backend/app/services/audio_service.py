import os
import uuid

import soundfile as sf

from app.core.config import settings
from app.core.exceptions import InvalidAudioError, UnsupportedFormatError, FileTooLargeError, CorruptAudioError
from app.core.logging import logger


def validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file extension '{ext}'. Allowed: {sorted(settings.ALLOWED_EXTENSIONS)}"
        )
    return ext


def validate_size(size_bytes: int) -> None:
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"File is {size_bytes / (1024*1024):.1f}MB, exceeds limit of {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    if size_bytes == 0:
        raise InvalidAudioError("Uploaded file is empty")


def safe_filename(original_filename: str) -> str:
    """Generate a collision-free, path-traversal-safe filename for storage."""
    ext = os.path.splitext(original_filename)[1].lower()
    return f"{uuid.uuid4().hex}{ext}"


def save_upload(file_bytes: bytes, original_filename: str) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    fname = safe_filename(original_filename)
    dest = os.path.join(settings.UPLOAD_DIR, fname)
    # os.path.basename + fixed upload dir root out any path traversal attempt
    dest = os.path.join(settings.UPLOAD_DIR, os.path.basename(fname))
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return dest


def probe_audio(path: str) -> dict:
    """Read audio metadata without fully decoding via librosa (fast, and
    catches corrupt files early with a clear error)."""
    try:
        info = sf.info(path)
    except Exception as e:
        raise CorruptAudioError(f"Unable to decode audio file: {e}")
    return {
        "duration_seconds": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
    }


def cleanup_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Failed to clean up temp file {path}: {e}")
