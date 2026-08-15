"""
Decision engine.

Combines two REAL model outputs into a single business decision:
  - classifier: per-class probabilities from the trained CNN (aerosense_cnn_v1)
  - autoencoder: reconstruction-error anomaly score from the trained
    autoencoder (aerosense_autoencoder_v1), thresholded using the value
    fit by evaluation/evaluate.py (models/registry.json), NOT a guess.

This module contains no ML of its own — it is deterministic decision logic
on top of two trained models' real outputs, documented below so severity
and status labels are auditable rather than opaque.
"""
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings


@dataclass
class DecisionResult:
    status: str                      # NORMAL | ANOMALY | UNKNOWN_ANOMALY
    fault_type: Optional[str]
    fault_confidence: Optional[float]
    anomaly_score: float
    anomaly_threshold: float
    severity: Optional[str]          # LOW | MEDIUM | HIGH | CRITICAL | None


# Baseline severity per known fault class, from FEWEST to MOST broken
# propellers (matches the real dataset's fault taxonomy). This is a
# documented, editable mapping — not a model output.
_FAULT_SEVERITY = {
    "1_broken": "LOW",
    "2_broken": "MEDIUM",
    "3_4_broken": "HIGH",
}


def _severity_for_known_fault(fault_type: str, anomaly_score: float, threshold: float) -> str:
    base = _FAULT_SEVERITY.get(fault_type, "MEDIUM")
    ratio = anomaly_score / threshold if threshold > 0 else 1.0
    # Escalate the most severe known class to CRITICAL when the anomaly
    # score is far beyond threshold (reconstruction error more than 2x the
    # fitted operating point), since that combination indicates both a
    # confidently-classified severe fault AND acoustically extreme deviation.
    if fault_type == "3_4_broken" and ratio > 2.0:
        return "CRITICAL"
    return base


def _severity_for_unknown_anomaly(anomaly_score: float, threshold: float) -> str:
    ratio = anomaly_score / threshold if threshold > 0 else 1.0
    if ratio >= 3.0:
        return "CRITICAL"
    if ratio >= 2.0:
        return "HIGH"
    if ratio >= 1.5:
        return "MEDIUM"
    return "LOW"


def decide(
    mean_class_probs,
    class_names: list[str],
    anomaly_score: float,
    threshold: float,
    min_confidence: float = None,
) -> DecisionResult:
    """
    mean_class_probs: (num_classes,) array-like, mean softmax probability
        across all segments of the analyzed clip.
    anomaly_score: mean autoencoder reconstruction error across segments.
    threshold: fitted anomaly threshold (models/registry.json).
    """
    if min_confidence is None:
        min_confidence = settings.MIN_CLASSIFIER_CONFIDENCE

    pred_idx = int(mean_class_probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(mean_class_probs[pred_idx])

    is_reconstruction_anomaly = anomaly_score >= threshold
    is_confident_known_fault = pred_class != "normal" and confidence >= min_confidence

    # Tie-breaker for near-threshold disagreement: if the reconstruction
    # error is only marginally over threshold (within 10%) AND the
    # classifier is very confident ("normal", >=0.9), treat it as NORMAL.
    # This does not suppress genuine anomalies (ratio >= 1.1 still flows
    # through to the anomaly branches below); it only prevents borderline
    # threshold noise from overriding a highly-confident normal
    # classification, which is the expected behavior of a threshold fit
    # for max F1 rather than zero false positives (see evaluation/evaluate.py).
    ratio = anomaly_score / threshold if threshold > 0 else 1.0
    if is_reconstruction_anomaly and pred_class == "normal" and confidence >= 0.9 and ratio < 1.1:
        is_reconstruction_anomaly = False

    if not is_reconstruction_anomaly and not is_confident_known_fault:
        return DecisionResult(
            status="NORMAL",
            fault_type=None,
            fault_confidence=confidence if pred_class == "normal" else None,
            anomaly_score=anomaly_score,
            anomaly_threshold=threshold,
            severity=None,
        )

    if is_confident_known_fault:
        severity = _severity_for_known_fault(pred_class, anomaly_score, threshold)
        return DecisionResult(
            status="ANOMALY",
            fault_type=pred_class,
            fault_confidence=confidence,
            anomaly_score=anomaly_score,
            anomaly_threshold=threshold,
            severity=severity,
        )

    # Reconstruction error says anomalous, but the classifier is not
    # confident about which known fault this is -> do not force a label.
    severity = _severity_for_unknown_anomaly(anomaly_score, threshold)
    return DecisionResult(
        status="UNKNOWN_ANOMALY",
        fault_type=None,
        fault_confidence=confidence,  # still reported, but NOT used as the fault label
        anomaly_score=anomaly_score,
        anomaly_threshold=threshold,
        severity=severity,
    )
