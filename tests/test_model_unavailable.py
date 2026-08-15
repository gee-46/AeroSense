from app.ml.model_loader import get_model_registry


def test_analyze_returns_503_when_classifier_not_loaded(client, synthetic_wav_path, monkeypatch):
    registry = get_model_registry()
    monkeypatch.setattr(registry, "classifier_loaded", False)

    with open(synthetic_wav_path, "rb") as f:
        r = client.post("/api/v1/audio/analyze", files={"file": ("synthetic.wav", f, "audio/wav")})

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_analyze_returns_503_when_autoencoder_not_loaded(client, synthetic_wav_path, monkeypatch):
    registry = get_model_registry()
    monkeypatch.setattr(registry, "autoencoder_loaded", False)

    with open(synthetic_wav_path, "rb") as f:
        r = client.post("/api/v1/audio/analyze", files={"file": ("synthetic.wav", f, "audio/wav")})

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_model_health_reflects_unloaded_state(client, monkeypatch):
    registry = get_model_registry()
    monkeypatch.setattr(registry, "classifier_loaded", False)
    r = client.get("/api/v1/health/model")
    assert r.status_code == 200
    assert r.json()["classifier_loaded"] is False
