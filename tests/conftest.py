import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

# --- Isolate test environment BEFORE importing app modules ---
TEST_DIR = tempfile.mkdtemp(prefix="aerosense_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DIR}/test_aerosense.db"
os.environ["UPLOAD_DIR"] = os.path.join(TEST_DIR, "uploads")
os.environ["OUTPUT_DIR"] = os.path.join(TEST_DIR, "outputs")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BACKEND_ROOT)

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.db.database import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _setup_db():
    init_db()
    yield
    shutil.rmtree(TEST_DIR, ignore_errors=True)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def synthetic_wav_path(tmp_path):
    """A short, valid, deterministic sine-wave WAV — used for API-mechanics
    tests (upload validation, error handling, persistence) where the exact
    predicted class doesn't matter."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    path = tmp_path / "synthetic.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture()
def too_short_wav_path(tmp_path):
    sr = 16000
    audio = np.zeros(int(sr * 0.2), dtype=np.float32)  # 0.2s, below min segment
    path = tmp_path / "tooshort.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture()
def corrupt_audio_path(tmp_path):
    path = tmp_path / "corrupt.wav"
    with open(path, "wb") as f:
        f.write(b"RIFF....not really a wav file....")
    return str(path)


def _find_real_sample(label: str):
    """Locate one real dataset segment source file for a given label, if the
    dataset has been downloaded (data/raw/...). Returns None if unavailable
    so tests can skip gracefully rather than fail in environments without
    the dataset."""
    manifest = os.path.join(REPO_ROOT, "data", "processed", "manifest.csv")
    if not os.path.exists(manifest):
        return None
    import csv
    with open(manifest) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["label"] == label:
                if os.path.exists(row["path"]):
                    return row["path"]
    return None


@pytest.fixture()
def real_normal_wav():
    path = _find_real_sample("normal")
    if path is None:
        pytest.skip("real dataset not available in this environment")
    return path


@pytest.fixture()
def real_fault_wav():
    path = _find_real_sample("3_4_broken")
    if path is None:
        pytest.skip("real dataset not available in this environment")
    return path
