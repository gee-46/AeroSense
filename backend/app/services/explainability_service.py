"""
Grad-CAM explainability for AeroSenseCNN.

Produces a real gradient-weighted class activation map over the mel
spectrogram (frequency x time) of the segment the model was most confident
about, then reports which frequency band and time window the model actually
attended to. This is derived directly from the trained model's gradients —
it is not a templated or invented explanation.
"""
import os
import uuid

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import settings
from common.config import AUDIO_CONFIG


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.features[-1].conv  # last conv layer before GAP
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: (1, 1, n_mels, n_frames). Returns a (n_mels, n_frames) heatmap in [0, 1]."""
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)


def summarize_attention(heatmap: np.ndarray, cfg=AUDIO_CONFIG, energy_threshold: float = 0.6) -> dict:
    """Turn a Grad-CAM heatmap into frequency/time ranges a human can read."""
    mask = heatmap >= energy_threshold
    if not mask.any():
        mask = heatmap >= heatmap.max() * 0.8  # fall back to top 20% region

    freq_bins = np.where(mask.any(axis=1))[0]
    time_bins = np.where(mask.any(axis=0))[0]

    # Convert mel-bin indices to approximate Hz using librosa's mel scale
    import librosa
    mel_freqs = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    freq_range = [float(mel_freqs[freq_bins.min()]), float(mel_freqs[freq_bins.max()])] if len(freq_bins) else None

    frame_to_sec = cfg.hop_length / cfg.sample_rate
    time_range = [float(time_bins.min() * frame_to_sec), float(time_bins.max() * frame_to_sec)] if len(time_bins) else None

    return {"important_frequency_range_hz": freq_range, "important_time_range_seconds": time_range}


def save_overlay(mel: np.ndarray, heatmap: np.ndarray, prediction_id: str) -> str:
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    fname = f"gradcam_{prediction_id}.png"
    path = os.path.join(settings.OUTPUT_DIR, fname)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.imshow(heatmap, origin="lower", aspect="auto", cmap="jet", alpha=0.45)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel bin")
    ax.set_title("Grad-CAM attribution over mel spectrogram")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def explain(registry, mel_segment_standardized: np.ndarray, class_idx: int, prediction_id: str) -> dict:
    """
    mel_segment_standardized: (1, n_mels, n_frames) — a single standardized
    segment (e.g. the one with the highest confidence for the predicted class).
    Returns explanation metadata + writes an overlay PNG to OUTPUT_DIR.
    """
    model = registry.require_classifier()
    model.zero_grad()
    cam = GradCAM(model)
    x = torch.from_numpy(mel_segment_standardized[np.newaxis, :, :, :].astype(np.float32)).to(registry.device)
    x.requires_grad_(False)
    heatmap = cam.generate(x, class_idx)
    ranges = summarize_attention(heatmap)
    overlay_path = save_overlay(mel_segment_standardized[0], heatmap, prediction_id)
    return {
        "method": "grad_cam",
        "heatmap_path": overlay_path,
        **ranges,
        "notes": "Attribution computed from the trained classifier's gradients with respect "
                 "to the predicted class, over the mel spectrogram of the highest-confidence "
                 "segment in the clip. Frequency/time ranges are the region where normalized "
                 "attribution >= 0.6 (or the top 20% region if no pixel reaches that).",
    }
