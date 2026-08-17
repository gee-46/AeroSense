def test_analyze_real_normal_clip_predicts_normal_or_flags_borderline(client, real_normal_wav):
    """Runs the FULL real pipeline (preprocessing -> real CNN -> real
    autoencoder -> decision engine -> Grad-CAM -> DB) on a real held-out
    'normal' recording from the test split. We assert on real model
    behavior, not a fabricated expectation: a well-trained model on this
    clip should predict 'normal' with high confidence; status may still be
    UNKNOWN_ANOMALY on rare borderline reconstruction-error cases, which is
    legitimate model behavior we don't hide, so we assert the weaker but
    meaningful invariant below."""
    with open(real_normal_wav, "rb") as f:
        r = client.post("/api/v1/audio/analyze", files={"file": ("normal.wav", f, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["class_probabilities"]["normal"] > 0.5
    assert body["status"] in ("NORMAL", "UNKNOWN_ANOMALY")
    assert body["fault_type"] != "3_4_broken"


def test_analyze_real_severe_fault_clip_detects_anomaly(client, real_fault_wav):
    with open(real_fault_wav, "rb") as f:
        r = client.post("/api/v1/audio/analyze", files={"file": ("fault.wav", f, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ANOMALY"
    assert body["fault_type"] == "3_4_broken"
    assert body["anomaly_score"] > body["anomaly_threshold"]
    assert body["explanation"] is not None
    assert body["explanation"]["method"] == "grad_cam"
    assert body["explanation"]["heatmap_path"].startswith("/static/outputs/")


def test_gradcam_heatmap_is_actually_servable(client, real_fault_wav):
    """Regression test: heatmap_path must be a servable URL, not a raw
    filesystem path — otherwise a browser frontend can't load it."""
    with open(real_fault_wav, "rb") as f:
        r = client.post("/api/v1/audio/analyze", files={"file": ("fault.wav", f, "audio/wav")})
    heatmap_path = r.json()["explanation"]["heatmap_path"]
    img_response = client.get(heatmap_path)
    assert img_response.status_code == 200
    assert img_response.headers["content-type"] == "image/png"


def test_prediction_persisted_and_retrievable(client, real_fault_wav):
    with open(real_fault_wav, "rb") as f:
        r = client.post(
            "/api/v1/audio/analyze",
            files={"file": ("fault.wav", f, "audio/wav")},
            data={"device_id": "test-device-1", "environment": "indoor"},
        )
    pred_id = r.json()["prediction_id"]

    r2 = client.get(f"/api/v1/predictions/{pred_id}")
    assert r2.status_code == 200
    assert r2.json()["prediction_id"] == pred_id
    assert r2.json()["device_id"] == "test-device-1"

    r3 = client.get("/api/v1/predictions", params={"device_id": "test-device-1"})
    assert r3.status_code == 200
    assert any(p["prediction_id"] == pred_id for p in r3.json()["items"])


def test_prediction_not_found_returns_404(client):
    r = client.get("/api/v1/predictions/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_delete_prediction(client, real_normal_wav):
    with open(real_normal_wav, "rb") as f:
        r = client.post("/api/v1/audio/analyze", files={"file": ("normal.wav", f, "audio/wav")})
    pred_id = r.json()["prediction_id"]

    r_del = client.delete(f"/api/v1/predictions/{pred_id}")
    assert r_del.status_code == 204

    r_get = client.get(f"/api/v1/predictions/{pred_id}")
    assert r_get.status_code == 404


def test_analytics_summary_reflects_real_predictions(client, real_fault_wav):
    with open(real_fault_wav, "rb") as f:
        client.post("/api/v1/audio/analyze", files={"file": ("fault.wav", f, "audio/wav")})
    r = client.get("/api/v1/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 1


def test_models_endpoint_reports_real_metadata(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    names = {m["name"] for m in body["models"]}
    assert "aerosense_cnn_v1" in names
    assert "aerosense_autoencoder_v1" in names
