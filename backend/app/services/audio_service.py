import math
import os
import re
import uuid

import soundfile as sf
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import InvalidAudioError, UnsupportedFormatError, FileTooLargeError, CorruptAudioError
from app.core.logging import logger

_READ_CHUNK = 1024 * 1024  # 1 MiB
_MAX_DISPLAY_NAME = 255
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def display_filename(original_filename: str | None) -> str:
    """The client's filename, reduced to something safe to store and echo back:
    no directory components, no control characters, bounded length. It is used
    for display only — files are always stored under a server-generated name."""
    name = (original_filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL_CHARS.sub("", name).strip()
    return name[:_MAX_DISPLAY_NAME] or "upload"


def validate_extension(filename: str | None) -> str:
    ext = os.path.splitext(display_filename(filename))[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file extension '{ext}'. Allowed: {sorted(settings.ALLOWED_EXTENSIONS)}"
        )
    return ext


def max_upload_bytes() -> int:
    return int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)


def validate_size(size_bytes: int) -> None:
    if size_bytes > max_upload_bytes():
        raise FileTooLargeError(f"File exceeds the upload limit of {settings.MAX_UPLOAD_SIZE_MB:g}MB")
    if size_bytes == 0:
        raise InvalidAudioError("Uploaded file is empty")


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload without ever buffering more than the limit (+1 chunk).

    `await file.read()` would pull an arbitrarily large body into memory before
    the size check ran."""
    limit = max_upload_bytes()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise FileTooLargeError(f"File exceeds the upload limit of {settings.MAX_UPLOAD_SIZE_MB:g}MB")
        chunks.append(chunk)
    validate_size(total)
    return b"".join(chunks)


def safe_filename(original_filename: str) -> str:
    """Generate a collision-free, path-traversal-safe filename for storage.
    Only the (allow-listed) extension of the client's name is kept."""
    ext = os.path.splitext(display_filename(original_filename))[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        ext = ""
    return f"{uuid.uuid4().hex}{ext}"


def save_upload(file_bytes: bytes, original_filename: str) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(settings.UPLOAD_DIR, safe_filename(original_filename))
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return dest


def probe_audio(path: str) -> dict:
    """Read audio metadata without fully decoding (fast, and catches corrupt
    files early with a clear error). Also enforces the maximum duration."""
    try:
        info = sf.info(path)
    except Exception as e:
        logger.info(f"Could not decode upload: {e}")
        raise CorruptAudioError("Unable to decode audio file; it may be corrupt or in an unsupported format")
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0:
        raise InvalidAudioError("Audio contains no samples")
    if duration > settings.MAX_AUDIO_DURATION_SECONDS:
        raise InvalidAudioError(
            f"Audio is {duration:.0f}s long; the maximum is {settings.MAX_AUDIO_DURATION_SECONDS:g}s"
        )
    return {
        "duration_seconds": duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
    }


def cleanup_file(path: str | None) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Failed to clean up temp file {path}: {e}")
