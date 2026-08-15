import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from backend.app.ml.models.cnn import AeroSenseCNN
from backend.app.ml.models.autoencoder import ConvAutoencoder
from common.config import AUDIO_CONFIG, NUM_CLASSES, MODEL_DIR


def test_cnn_forward_pass_shape():
    model = AeroSenseCNN(num_classes=NUM_CLASSES)
    x = torch.randn(4, 1, AUDIO_CONFIG.n_mels, AUDIO_CONFIG.n_frames)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_cnn_embed_shape():
    model = AeroSenseCNN(num_classes=NUM_CLASSES)
    x = torch.randn(2, 1, AUDIO_CONFIG.n_mels, AUDIO_CONFIG.n_frames)
    emb = model.embed(x)
    assert emb.shape == (2, 128)


def test_autoencoder_forward_pass_preserves_shape():
    model = ConvAutoencoder()
    x = torch.randn(3, 1, AUDIO_CONFIG.n_mels, AUDIO_CONFIG.n_frames)
    recon = model(x)
    assert recon.shape == x.shape


def test_autoencoder_reconstruction_error_is_nonnegative():
    model = ConvAutoencoder()
    x = torch.randn(5, 1, AUDIO_CONFIG.n_mels, AUDIO_CONFIG.n_frames)
    err = model.reconstruction_error(x)
    assert err.shape == (5,)
    assert torch.all(err >= 0)


def test_real_cnn_checkpoint_loads_and_is_deterministic():
    ckpt_path = os.path.join(MODEL_DIR, "aerosense_cnn_v1.pt")
    if not os.path.exists(ckpt_path):
        import pytest
        pytest.skip("CNN checkpoint not trained in this environment")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = AeroSenseCNN(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.randn(1, 1, AUDIO_CONFIG.n_mels, AUDIO_CONFIG.n_frames)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "eval-mode forward pass must be deterministic"
    assert ckpt["val_macro_f1"] > 0.5, "sanity check: trained checkpoint should beat random guessing"


def test_real_autoencoder_checkpoint_loads():
    ckpt_path = os.path.join(MODEL_DIR, "aerosense_autoencoder_v1.pt")
    if not os.path.exists(ckpt_path):
        import pytest
        pytest.skip("autoencoder checkpoint not trained in this environment")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ConvAutoencoder()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = torch.randn(1, 1, AUDIO_CONFIG.n_mels, AUDIO_CONFIG.n_frames)
    with torch.no_grad():
        err = model.reconstruction_error(x)
    assert err.item() >= 0
