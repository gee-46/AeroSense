def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_model_health_reports_real_trained_models(client):
    r = client.get("/api/v1/health/model")
    assert r.status_code == 200
    body = r.json()
    # These assertions only pass if REAL checkpoints were loaded — if you
    # run this test suite before training, it will fail loudly rather than
    # silently pretending models exist.
    assert body["classifier_loaded"] is True
    assert body["classifier_is_trained"] is True
    assert body["autoencoder_loaded"] is True
    assert body["autoencoder_is_trained"] is True
    assert body["anomaly_threshold"] is not None
    assert 0 < body["anomaly_threshold"] < 1
