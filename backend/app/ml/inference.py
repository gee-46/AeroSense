"""Low-level inference: tensor in, model outputs out. No I/O, no business logic
(that lives in app/services/*). Uses torch.no_grad() throughout per performance
requirements."""
import numpy as np
import torch

from app.ml.model_loader import ModelRegistry


def classify(registry: ModelRegistry, batch: np.ndarray) -> dict:
    """
    batch: (N, 1, n_mels, n_frames) float32 array of standardized log-mel segments.
    Returns per-segment softmax probabilities and the aggregated (mean) probability
    vector across all segments of the clip.
    """
    model = registry.require_classifier()
    x = torch.from_numpy(batch.astype(np.float32)).to(registry.device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    mean_probs = probs.mean(axis=0)
    return {
        "segment_probs": probs,          # (N, num_classes)
        "mean_probs": mean_probs,        # (num_classes,)
    }


def autoencoder_scores(registry: ModelRegistry, batch: np.ndarray) -> np.ndarray:
    """Return per-segment reconstruction error (anomaly score components)."""
    model = registry.require_autoencoder()
    x = torch.from_numpy(batch.astype(np.float32)).to(registry.device)
    with torch.no_grad():
        errors = model.reconstruction_error(x).cpu().numpy()
    return errors


def embed(registry: ModelRegistry, batch: np.ndarray) -> np.ndarray:
    """CNN penultimate-layer embeddings, used for embedding-space distance
    as a secondary signal for unknown-anomaly detection."""
    model = registry.require_classifier()
    x = torch.from_numpy(batch.astype(np.float32)).to(registry.device)
    with torch.no_grad():
        emb = model.embed(x).cpu().numpy()
    return emb
