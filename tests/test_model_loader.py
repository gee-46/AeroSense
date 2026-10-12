"""Model-loading safety: wrong class order, missing/invalid stats and registry
problems must never silently produce a 'working' but wrong model."""
import json
import os
import shutil

import pytest
import torch

from app.core.config import settings
from app.core.exceptions import ModelUnavailableError
from app.ml.model_loader import ModelRegistry


def _fresh_registry():
    inst = object.__new__(ModelRegistry)      # bypass the process-wide singleton
    inst._initialized = False
    inst.__init__()
    return inst


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    src = settings.MODEL_DIR
    for f in ("aerosense_cnn_v1.pt", "aerosense_autoencoder_v1.pt", "feature_stats.json", "registry.json"):
        shutil.copy(os.path.join(src, f), tmp_path / f)
    monkeypatch.setattr(settings, "CLASSIFIER_MODEL_PATH", str(tmp_path / "aerosense_cnn_v1.pt"))
    monkeypatch.setattr(settings, "AUTOENCODER_MODEL_PATH", str(tmp_path / "aerosense_autoencoder_v1.pt"))
    monkeypatch.setattr(settings, "FEATURE_STATS_PATH", str(tmp_path / "feature_stats.json"))
    monkeypatch.setattr(settings, "REGISTRY_PATH", str(tmp_path / "registry.json"))
    return tmp_path


def test_pristine_checkpoints_load_cleanly(model_dir):
    r = _fresh_registry()
    assert r.classifier_loaded and r.autoencoder_loaded and r.feature_stats_loaded
    assert r.consistency_warnings == []
    assert r.classifier_is_trained() and r.autoencoder_is_trained()
    assert r.anomaly_threshold_source == "registry.json"
    assert r.anomaly_threshold == pytest.approx(0.024940771710633032)
    assert r.classifier_meta["class_names"] == ["normal", "1_broken", "2_broken", "3_4_broken"]


def test_checkpoint_with_different_class_order_is_refused(model_dir):
    p = model_dir / "aerosense_cnn_v1.pt"
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
    ckpt["class_names"] = ["1_broken", "normal", "2_broken", "3_4_broken"]
    torch.save(ckpt, p)
    r = _fresh_registry()
    assert r.classifier_loaded is False and r.autoencoder_loaded is True


def test_missing_checkpoints_leave_models_unloaded(model_dir):
    (model_dir / "aerosense_cnn_v1.pt").unlink()
    (model_dir / "aerosense_autoencoder_v1.pt").unlink()
    r = _fresh_registry()
    assert not r.classifier_loaded and not r.autoencoder_loaded
    assert not r.classifier_is_trained() and not r.autoencoder_is_trained()


def test_corrupt_checkpoint_is_not_loaded_and_does_not_crash(model_dir):
    (model_dir / "aerosense_cnn_v1.pt").write_bytes(b"not a checkpoint")
    assert _fresh_registry().classifier_loaded is False


@pytest.mark.parametrize("content", [None, "{not json",
                                     json.dumps({"log_mel_mean": 0.0, "log_mel_std": 0.0}),
                                     json.dumps({"log_mel_mean": "x", "log_mel_std": 1}),
                                     json.dumps({"log_mel_mean": 1.0})])
def test_bad_feature_stats_disable_inference(model_dir, content):
    path = model_dir / "feature_stats.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    r = _fresh_registry()
    assert r.feature_stats_loaded is False
    with pytest.raises(ModelUnavailableError):
        r.require_classifier()


def test_feature_stats_from_other_audio_config_are_rejected(model_dir):
    path = model_dir / "feature_stats.json"
    stats = json.loads(path.read_text())
    stats["audio_config"]["n_mels"] = 128
    path.write_text(json.dumps(stats))
    assert _fresh_registry().feature_stats_loaded is False


def test_registry_missing_falls_back_to_default_threshold_and_is_labelled(model_dir):
    (model_dir / "registry.json").unlink()
    r = _fresh_registry()
    assert r.anomaly_threshold_source == "default"
    assert r.anomaly_threshold == settings.ANOMALY_THRESHOLD
    assert not r.classifier_is_trained()          # loaded, but nothing declares it trained


@pytest.mark.parametrize("bad", [0, -1, "x"])
def test_invalid_registry_threshold_is_ignored(model_dir, bad):
    path = model_dir / "registry.json"
    reg = json.loads(path.read_text())
    reg["autoencoder"]["anomaly_threshold"] = bad
    path.write_text(json.dumps(reg))
    r = _fresh_registry()
    assert r.anomaly_threshold_source == "default"


def test_registry_disagreeing_with_checkpoint_is_reported(model_dir):
    path = model_dir / "registry.json"
    reg = json.loads(path.read_text())
    reg["classifier"]["trained_epoch"] = 99
    reg["autoencoder"]["n_params"] = 1
    path.write_text(json.dumps(reg))
    r = _fresh_registry()
    assert any("trained_epoch" in w for w in r.consistency_warnings)
    assert any("n_params" in w for w in r.consistency_warnings)
