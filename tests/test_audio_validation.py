import io


def test_upload_rejects_unsupported_extension(client):
    r = client.post(
        "/api/v1/audio/upload",
        files={"file": ("notaudio.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_upload_rejects_empty_file(client):
    r = client.post(
        "/api/v1/audio/upload",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_AUDIO"


def test_upload_rejects_corrupt_audio(client, corrupt_audio_path):
    with open(corrupt_audio_path, "rb") as f:
        r = client.post(
            "/api/v1/audio/upload",
            files={"file": ("corrupt.wav", f, "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "CORRUPT_AUDIO"


def test_upload_accepts_valid_wav(client, synthetic_wav_path):
    with open(synthetic_wav_path, "rb") as f:
        r = client.post(
            "/api/v1/audio/upload",
            files={"file": ("synthetic.wav", f, "audio/wav")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["sample_rate"] == 16000
    assert body["channels"] == 1
    assert 2.9 < body["duration_seconds"] < 3.1


def test_analyze_rejects_oversized_file(client, monkeypatch, synthetic_wav_path):
    from app.core.config import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.001)  # ~1KB limit
    with open(synthetic_wav_path, "rb") as f:
        r = client.post(
            "/api/v1/audio/analyze",
            files={"file": ("synthetic.wav", f, "audio/wav")},
        )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_analyze_rejects_too_short_audio(client, too_short_wav_path):
    with open(too_short_wav_path, "rb") as f:
        r = client.post(
            "/api/v1/audio/analyze",
            files={"file": ("tooshort.wav", f, "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_AUDIO"


def test_analyze_malformed_request_missing_file(client):
    r = client.post("/api/v1/audio/analyze", data={"device_id": "x"})
    assert r.status_code == 422  # FastAPI/Pydantic validation error
