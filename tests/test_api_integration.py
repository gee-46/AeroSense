"""End-to-end API tests that run the REAL trained checkpoints.

They use small synthetic audio purely to exercise the pipeline mechanics
(finite outputs, persistence, Grad-CAM serving, cleanup, error envelopes).
They deliberately make no claim about classification accuracy: accuracy is
measured only by evaluation/evaluate.py on the held-out dataset split.
"""
import io
import math
import os
import threading

import pytest

from conftest import post_audio, list_dir
from app.core.config import settings
from app.ml.model_loader import get_model_registry
from common.config import CLASS_NAMES


# ---------------------------------------------------------------- inference
def test_analyze_runs_real_models_and_returns_valid_outputs(client, noise_wav_path):
    r = post_audio(client, noise_wav_path, data={"device_id": "dev-1"})
    assert r.status_code == 200, r.text
    b = r.json()
    registry = get_model_registry()

    assert list(b["class_probabilities"]) == CLASS_NAMES          # label order preserved
    probs = list(b["class_probabilities"].values())
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-4)
    assert math.isfinite(b["anomaly_score"]) and b["anomaly_score"] >= 0
    # Threshold must come from registry.json, not the 0.5 fallback default.
    assert registry.anomaly_threshold_source == "registry.json"
    assert b["anomaly_threshold"] == pytest.approx(registry.registry_meta["autoencoder"]["anomaly_threshold"])
    assert b["anomaly_threshold"] != settings.ANOMALY_THRESHOLD
    assert b["status"] in ("NORMAL", "ANOMALY", "UNKNOWN_ANOMALY")
    assert b["model_version"] == "aerosense_cnn_v1"
    assert b["autoencoder_version"] == "aerosense_autoencoder_v1"
    # Inference is deterministic in eval mode.
    r2 = post_audio(client, noise_wav_path)
    assert r2.json()["class_probabilities"] == b["class_probabilities"]
    assert r2.json()["anomaly_score"] == b["anomaly_score"]


def test_anomaly_decision_is_consistent_with_threshold(client, noise_wav_path):
    b = post_audio(client, noise_wav_path).json()
    assert b["status"] != "NORMAL"
    assert b["anomaly_score"] >= b["anomaly_threshold"]
    if b["status"] == "ANOMALY":
        assert b["fault_type"] in CLASS_NAMES[1:]
    else:
        assert b["fault_type"] is None


# ------------------------------------------------------------- persistence
def test_get_matches_post_and_keeps_real_audio_metadata(client, noise_wav_path):
    posted = post_audio(client, noise_wav_path, name="mission 7.wav",
                        data={"device_id": "dev-2", "environment": "indoor", "rpm": "4200"}).json()
    got = client.get(f"/api/v1/predictions/{posted['prediction_id']}").json()
    assert got == posted                      # same mapping for both endpoints
    assert got["audio"]["filename"] == "mission 7.wav"
    assert got["audio"]["format"] == "WAV"
    assert got["audio"]["channels"] == 1
    assert got["audio"]["size_bytes"] > 0
    assert got["rpm"] == 4200 and got["environment"] == "indoor"
    assert got["explanation"]["important_frequency_range_hz"] is not None
    assert got["explanation"]["notes"]


def test_timestamps_are_timezone_aware_utc(client, noise_wav_path):
    posted = post_audio(client, noise_wav_path).json()
    # A naive ISO string would be parsed as *local* time by browsers.
    assert posted["timestamp"].endswith(("Z", "+00:00"))
    trend = client.get("/api/v1/analytics/anomaly-trend").json()
    assert all(p["timestamp"].endswith(("Z", "+00:00")) for p in trend)


def test_database_survives_application_restart(noise_wav_path):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c1:
        pid = post_audio(c1, noise_wav_path).json()["prediction_id"]
    # Leaving the context runs shutdown (engine disposed); entering again runs startup.
    with TestClient(app) as c2:
        assert c2.get(f"/api/v1/predictions/{pid}").status_code == 200
        items = c2.get("/api/v1/predictions?limit=200").json()["items"]
        assert any(p["prediction_id"] == pid for p in items)


def test_old_database_without_new_columns_is_migrated(tmp_path):
    from sqlalchemy import create_engine, inspect, text
    from app.db.database import _add_missing_columns
    eng = create_engine(f"sqlite:///{tmp_path}/old.db")
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE predictions (prediction_id VARCHAR PRIMARY KEY, filename VARCHAR NOT NULL, "
            "duration_seconds FLOAT NOT NULL, sample_rate INTEGER NOT NULL, model_version VARCHAR NOT NULL, "
            "autoencoder_version VARCHAR NOT NULL, status VARCHAR NOT NULL, anomaly_score FLOAT NOT NULL, "
            "anomaly_threshold FLOAT NOT NULL, processing_time_ms FLOAT NOT NULL)"))
    _add_missing_columns(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("predictions")}
    assert {"channels", "audio_format", "size_bytes", "explanation", "explanation_path"} <= cols


# -------------------------------------------------------------- Grad-CAM
def test_gradcam_image_is_served_and_removed_on_delete(client, noise_wav_path):
    b = post_audio(client, noise_wav_path).json()
    url = b["explanation"]["heatmap_path"]
    assert url.startswith("/static/outputs/gradcam_") and ".." not in url
    assert settings.OUTPUT_DIR not in str(b)                       # no filesystem path leaks
    img = client.get(url)
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.content[:8] == b"\x89PNG\r\n\x1a\n"

    assert client.delete(f"/api/v1/predictions/{b['prediction_id']}").status_code == 204
    assert client.get(url).status_code == 404                     # no orphaned image
    assert client.get(f"/api/v1/predictions/{b['prediction_id']}").status_code == 404
    assert client.delete(f"/api/v1/predictions/{b['prediction_id']}").status_code == 404


def test_gradcam_hooks_do_not_accumulate_across_requests(client, noise_wav_path):
    conv = get_model_registry().classifier.features[-1].conv
    before = (len(conv._forward_hooks), len(conv._backward_hooks))
    for _ in range(3):
        assert post_audio(client, noise_wav_path).json()["explanation"] is not None
    assert (len(conv._forward_hooks), len(conv._backward_hooks)) == before


def test_concurrent_analyses_all_succeed(client, noise_wav_path):
    results = []

    def work():
        results.append(post_audio(client, noise_wav_path).status_code)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 4


# ---------------------------------------------------------- upload cleanup
def test_upload_dir_is_empty_after_success_and_failures(client, noise_wav_path, corrupt_audio_path,
                                                        too_short_wav_path, upload_dir, monkeypatch):
    assert post_audio(client, noise_wav_path).status_code == 200
    assert post_audio(client, corrupt_audio_path).status_code == 422
    assert post_audio(client, too_short_wav_path).status_code == 422
    with open(noise_wav_path, "rb") as f:
        assert client.post("/api/v1/audio/upload", files={"file": ("a.wav", f, "audio/wav")}).status_code == 200
    monkeypatch.setattr(get_model_registry(), "autoencoder_loaded", False)
    assert post_audio(client, noise_wav_path).status_code == 503
    assert list_dir(upload_dir) == []


def test_no_orphaned_gradcam_images(client, noise_wav_path):
    ids = [post_audio(client, noise_wav_path).json()["prediction_id"] for _ in range(2)]
    on_disk = {f for f in os.listdir(settings.OUTPUT_DIR) if f.startswith("gradcam_")}
    assert {f"gradcam_{i}.png" for i in ids} <= on_disk
    for i in ids:
        client.delete(f"/api/v1/predictions/{i}")
    remaining = {f for f in os.listdir(settings.OUTPUT_DIR) if f.startswith("gradcam_")}
    assert not ({f"gradcam_{i}.png" for i in ids} & remaining)


def test_persistence_failure_removes_generated_image(client, noise_wav_path, monkeypatch):
    from app.core.exceptions import DatabaseError
    from app.db import repository

    def boom(*a, **k):
        raise DatabaseError("Failed to persist prediction")

    before = set(os.listdir(settings.OUTPUT_DIR))
    monkeypatch.setattr(repository, "create_prediction", boom)
    r = post_audio(client, noise_wav_path)
    assert r.status_code == 500 and r.json()["error"]["code"] == "DATABASE_ERROR"
    assert set(os.listdir(settings.OUTPUT_DIR)) == before


# -------------------------------------------------------- input validation
@pytest.mark.parametrize("rpm", ["-5", "nan", "inf", "99999999"])
def test_invalid_rpm_rejected_with_standard_envelope(client, noise_wav_path, rpm):
    r = post_audio(client, noise_wav_path, data={"rpm": rpm})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "detail" not in r.json()


def test_m4a_is_not_accepted(client):
    r = client.post("/api/v1/audio/upload", files={"file": ("x.m4a", io.BytesIO(b"abc"), "audio/mp4")})
    assert r.status_code == 415


def test_too_long_audio_rejected(client, noise_wav_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_AUDIO_DURATION_SECONDS", 1.0)
    r = post_audio(client, noise_wav_path)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_AUDIO"


def test_path_traversal_filename_is_neutralised(client, noise_wav_path, upload_dir):
    r = post_audio(client, noise_wav_path, name="../../../etc/evil.wav")
    assert r.status_code == 200
    assert r.json()["audio"]["filename"] == "evil.wav"
    assert list_dir(upload_dir) == []


def test_oversize_body_rejected_before_full_read(client, noise_wav_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.01)
    r = post_audio(client, noise_wav_path)
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_unknown_route_and_method_use_error_envelope(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    r = client.put("/api/v1/predictions")
    assert r.status_code == 405 and "error" in r.json()


# ----------------------------------------------- listing / filter / analytics
def test_listing_filter_pagination_and_analytics(client, noise_wav_path):
    for dev in ("fleet-a", "fleet-a", "fleet-b"):
        assert post_audio(client, noise_wav_path, data={"device_id": dev}).status_code == 200

    a = client.get("/api/v1/predictions", params={"device_id": "fleet-a"}).json()
    assert a["total"] == 2 and len(a["items"]) == 2
    page = client.get("/api/v1/predictions", params={"device_id": "fleet-a", "limit": 1, "offset": 1}).json()
    assert page["total"] == 2 and len(page["items"]) == 1
    stamps = [p["timestamp"] for p in a["items"]]
    assert stamps == sorted(stamps, reverse=True)                      # newest first

    anomalies = client.get("/api/v1/predictions", params={"status": "ANOMALY"}).json()
    assert all(p["status"] == "ANOMALY" for p in anomalies["items"])
    assert client.get("/api/v1/predictions", params={"status": "BOGUS"}).status_code == 422
    assert client.get("/api/v1/predictions", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/predictions", params={"limit": 500}).status_code == 422
    assert client.get("/api/v1/predictions", params={"offset": -1}).status_code == 422

    everything = client.get("/api/v1/predictions", params={"limit": 200}).json()
    summary = client.get("/api/v1/analytics/summary").json()
    assert summary["total_predictions"] == everything["total"]
    assert sum(summary["status_distribution"].values()) == summary["total_predictions"]
    faults = client.get("/api/v1/analytics/fault-distribution").json()
    expected = {}
    for p in everything["items"]:
        if p["status"] == "ANOMALY":
            expected[p["fault_type"]] = expected.get(p["fault_type"], 0) + 1
    assert faults == expected
    trend = client.get("/api/v1/analytics/anomaly-trend").json()
    assert len(trend) == min(everything["total"], 200)
    assert [t["timestamp"] for t in trend] == sorted(t["timestamp"] for t in trend)   # oldest first


# ------------------------------------------------------- models / health
def test_models_endpoint_distinguishes_configured_loaded_trained(client):
    body = client.get("/api/v1/models").json()
    for m in body["models"]:
        assert (m["configured"], m["loaded"], m["trained"]) == (True, True, True)
        assert m["meta"]["n_params"] > 0
    flat = str(body)
    assert "models/aerosense" not in flat and settings.MODEL_DIR not in flat   # no server paths
    assert "CRNN" in str(body["registry"]["limitations"])                      # honest limitation kept
    active = client.get("/api/v1/models/active").json()
    assert active["classifier"] == "aerosense_cnn_v1" and active["anomaly_threshold_source"] == "registry.json"


def test_model_health_trained_flag_is_not_just_loaded(client, monkeypatch):
    registry = get_model_registry()
    h = client.get("/api/v1/health/model").json()
    assert h["classifier_is_trained"] and h["feature_stats_loaded"] and h["consistency_warnings"] == []
    monkeypatch.setattr(registry, "registry_meta", {})            # loaded, but nothing declares it trained
    h = client.get("/api/v1/health/model").json()
    assert h["classifier_loaded"] is True and h["classifier_is_trained"] is False


def test_analyze_503_when_feature_stats_unavailable(client, noise_wav_path, monkeypatch):
    monkeypatch.setattr(get_model_registry(), "feature_stats_loaded", False)
    r = post_audio(client, noise_wav_path)
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_openapi_documents_every_frontend_endpoint(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    expected = {
        ("get", "/api/v1/health"), ("get", "/api/v1/health/model"),
        ("post", "/api/v1/audio/upload"), ("post", "/api/v1/audio/analyze"),
        ("get", "/api/v1/predictions"), ("get", "/api/v1/predictions/{prediction_id}"),
        ("delete", "/api/v1/predictions/{prediction_id}"),
        ("get", "/api/v1/models"), ("get", "/api/v1/models/active"),
        ("get", "/api/v1/analytics/summary"), ("get", "/api/v1/analytics/fault-distribution"),
        ("get", "/api/v1/analytics/anomaly-trend"),
    }
    for method, path in expected:
        assert method in paths.get(path, {}), f"{method.upper()} {path} missing from OpenAPI"
    pred = spec["components"]["schemas"]["PredictionResult"]["properties"]
    assert set(pred) == {
        "prediction_id", "status", "fault_type", "fault_confidence", "anomaly_score", "anomaly_threshold",
        "severity", "model_version", "autoencoder_version", "processing_time_ms", "timestamp", "audio",
        "class_probabilities", "explanation", "device_id", "environment", "operating_condition", "rpm",
    }
