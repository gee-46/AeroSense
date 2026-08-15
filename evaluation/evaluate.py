"""
Evaluate the trained CNN classifier and autoencoder on real held-out data,
and fit the anomaly-detection threshold from validation data (never guessed).

Produces:
    models/registry.json                    (model metadata + fitted threshold, source of truth for the backend)
    evaluation/results/cnn_confusion_matrix.png
    evaluation/results/cnn_test_report.json
    evaluation/results/autoencoder_threshold_analysis.json
    evaluation/results/autoencoder_roc.png     (unknown-anomaly detection ROC: normal vs faulty)

Usage:
    python -m evaluation.evaluate
"""
import json
import os
import sys
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import CLASS_NAMES, MODEL_DIR, DATA_PROCESSED_DIR
from training.dataset import AeroSenseSegmentDataset, CLASS_TO_IDX
from backend.app.ml.models.cnn import AeroSenseCNN
from backend.app.ml.models.autoencoder import ConvAutoencoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_cnn(device):
    ckpt = torch.load(os.path.join(MODEL_DIR, "aerosense_cnn_v1.pt"), map_location=device)
    model = AeroSenseCNN(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_autoencoder(device):
    ckpt = torch.load(os.path.join(MODEL_DIR, "aerosense_autoencoder_v1.pt"), map_location=device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_cnn_on_test(model, device, manifest_path):
    test_ds = AeroSenseSegmentDataset(manifest_path, "test", augment=False)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    t0 = None
    import time
    latencies = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            t_start = time.time()
            logits = model(x)
            latencies.append((time.time() - t_start) / x.size(0) * 1000)  # ms/sample
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_latency_ms = float(np.mean(latencies))

    return {
        "n_test_segments": len(test_ds),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "avg_inference_latency_ms_per_segment": avg_latency_ms,
    }, cm, all_labels, all_preds


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("AeroSense CNN — Test Set Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compute_reconstruction_errors(model, device, manifest_path, split):
    """Return (errors, is_anomaly) for every segment in `split`, using the
    real class label to define ground truth: normal=0, any fault=1."""
    ds = AeroSenseSegmentDataset(manifest_path, split, augment=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    errors, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            err = model.reconstruction_error(x)
            errors.extend(err.cpu().numpy().tolist())
            labels.extend((y.numpy() != CLASS_TO_IDX["normal"]).astype(int).tolist())
    return np.array(errors), np.array(labels)


def fit_anomaly_threshold(val_errors, val_labels):
    """
    Choose the reconstruction-error threshold on the VALIDATION split that
    maximizes F1 for the binary normal-vs-anomaly decision. This replaces
    any hand-picked constant with a threshold actually fit to data, and the
    search is done only on validation data (test set stays untouched until
    final reporting).
    """
    thresholds = np.linspace(val_errors.min(), val_errors.max(), 200)
    best_f1, best_t = -1, float(np.median(val_errors))
    for t in thresholds:
        preds = (val_errors >= t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = os.path.join(DATA_PROCESSED_DIR, "manifest.csv")

    print("=== CNN classifier: test-set evaluation ===")
    cnn_model, cnn_ckpt = load_cnn(device)
    cnn_report, cm, y_true, y_pred = evaluate_cnn_on_test(cnn_model, device, manifest_path)
    print(json.dumps({k: v for k, v in cnn_report.items() if k != "classification_report"}, indent=2))
    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png"))
    with open(os.path.join(RESULTS_DIR, "cnn_test_report.json"), "w") as f:
        json.dump(cnn_report, f, indent=2)

    print("\n=== Autoencoder: fitting anomaly threshold on VALIDATION data ===")
    ae_model, ae_ckpt = load_autoencoder(device)
    val_errors, val_labels = compute_reconstruction_errors(ae_model, device, manifest_path, "val")
    threshold, val_f1_at_threshold = fit_anomaly_threshold(val_errors, val_labels)
    print(f"Fitted threshold={threshold:.5f} (val F1={val_f1_at_threshold:.4f}) "
          f"from {len(val_errors)} val segments "
          f"({int(val_labels.sum())} anomalous, {int((1-val_labels).sum())} normal)")

    print("\n=== Autoencoder: evaluating fitted threshold on TEST data (held out) ===")
    test_errors, test_labels = compute_reconstruction_errors(ae_model, device, manifest_path, "test")
    test_preds = (test_errors >= threshold).astype(int)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0
    )
    test_acc = accuracy_score(test_labels, test_preds)
    try:
        test_auc = roc_auc_score(test_labels, test_errors)
    except ValueError:
        test_auc = None

    fpr, tpr, _ = roc_curve(test_labels, test_errors)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={test_auc:.3f}" if test_auc else "AUC=n/a")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Autoencoder anomaly detection ROC (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "autoencoder_roc.png"), dpi=120)
    plt.close(fig)

    ae_results = {
        "threshold": threshold,
        "threshold_fit_method": "max-F1 sweep over validation-set reconstruction errors "
                                 "(200-point linspace between observed min/max)",
        "validation": {
            "n_segments": len(val_errors),
            "n_anomalous": int(val_labels.sum()),
            "n_normal": int((1 - val_labels).sum()),
            "f1_at_threshold": val_f1_at_threshold,
        },
        "test": {
            "n_segments": len(test_errors),
            "n_anomalous": int(test_labels.sum()),
            "n_normal": int((1 - test_labels).sum()),
            "accuracy": test_acc,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_auc,
        },
        "note": "Ground truth for this binary anomaly evaluation is derived from the "
                "dataset's real class labels (normal=0, any broken-propeller class=1). "
                "The autoencoder itself was trained ONLY on normal segments and never "
                "saw a faulty segment during training.",
    }
    with open(os.path.join(RESULTS_DIR, "autoencoder_threshold_analysis.json"), "w") as f:
        json.dump(ae_results, f, indent=2)

    print(f"TEST SET  precision={test_precision:.4f}  recall={test_recall:.4f}  "
          f"f1={test_f1:.4f}  acc={test_acc:.4f}  auc={test_auc}")

    # --- Write the model registry: single source of truth consumed by the backend ---
    registry = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "classifier": {
            "name": "aerosense_cnn_v1",
            "path": "models/aerosense_cnn_v1.pt",
            "architecture": "AeroSenseCNN (4 conv blocks + GAP + FC head)",
            "class_names": CLASS_NAMES,
            "dataset": "tiiuae/UAV-Propeller-Anomaly-Audio-Dataset (ICASSP 2022)",
            "trained_epoch": cnn_ckpt["epoch"],
            "val_macro_f1": cnn_ckpt["val_macro_f1"],
            "val_accuracy": cnn_ckpt["val_accuracy"],
            "test_accuracy": cnn_report["accuracy"],
            "test_macro_f1": cnn_report["macro_f1"],
            "test_avg_latency_ms_per_segment": cnn_report["avg_inference_latency_ms_per_segment"],
            "n_params": sum(p.numel() for p in cnn_model.parameters()),
            "is_trained": True,
        },
        "autoencoder": {
            "name": "aerosense_autoencoder_v1",
            "path": "models/aerosense_autoencoder_v1.pt",
            "architecture": "ConvAutoencoder (3-layer conv encoder/decoder, latent_dim=32)",
            "trained_on": "normal-class segments only (train split)",
            "trained_epoch": ae_ckpt["epoch"],
            "val_loss": ae_ckpt["val_loss"],
            "anomaly_threshold": threshold,
            "anomaly_threshold_fit_method": ae_results["threshold_fit_method"],
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_roc_auc": test_auc,
            "n_params": sum(p.numel() for p in ae_model.parameters()),
            "is_trained": True,
        },
        "feature_stats_path": "models/feature_stats.json",
        "manifest_path": "data/processed/manifest.csv",
        "limitations": [
            "Classes are propeller-damage-severity classes from a single real dataset "
            "(normal / 1_broken / 2_broken / 3_4_broken), not a general bearing/motor/"
            "aircraft-engine fault taxonomy — no reachable dataset covered those.",
            "Only a CNN was trained to convergence; CRNN/Transformer architectures were "
            "not trained due to single-CPU-core compute constraints in this environment "
            "(see training/ for untrained-but-implemented alternatives if applicable).",
            "Segments are 2-second, 50%-overlap windows from 293 real indoor flight "
            "recordings; the model has not been validated on outdoor or fixed-wing audio.",
        ],
    }
    with open(os.path.join(MODEL_DIR, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nWrote model registry: {os.path.join(MODEL_DIR, 'registry.json')}")


if __name__ == "__main__":
    main()
