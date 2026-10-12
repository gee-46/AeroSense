import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.services.anomaly_service import decide

CLASS_NAMES = ["normal", "1_broken", "2_broken", "3_4_broken"]


def test_decide_normal_when_below_threshold_and_predicted_normal():
    probs = np.array([0.95, 0.02, 0.02, 0.01])
    result = decide(probs, CLASS_NAMES, anomaly_score=0.01, threshold=0.05)
    assert result.status == "NORMAL"
    assert result.fault_type is None
    assert result.severity is None


def test_decide_anomaly_with_confident_known_fault():
    probs = np.array([0.01, 0.02, 0.02, 0.95])
    result = decide(probs, CLASS_NAMES, anomaly_score=0.20, threshold=0.05, min_confidence=0.5)
    assert result.status == "ANOMALY"
    assert result.fault_type == "3_4_broken"
    assert result.severity in ("HIGH", "CRITICAL")


def test_decide_unknown_anomaly_when_reconstruction_flags_but_classifier_unsure():
    probs = np.array([0.3, 0.3, 0.25, 0.15])  # no class confident
    result = decide(probs, CLASS_NAMES, anomaly_score=0.20, threshold=0.05, min_confidence=0.5)
    assert result.status == "UNKNOWN_ANOMALY"
    assert result.fault_type is None
    assert result.severity is not None


def test_decide_does_not_force_known_category_on_unknown_anomaly():
    """Explicit regression test for spec requirement: unknown anomalies must
    never be assigned a fault_type just because SOME class has the highest
    (but not confident) probability."""
    probs = np.array([0.28, 0.27, 0.26, 0.19])
    result = decide(probs, CLASS_NAMES, anomaly_score=1.0, threshold=0.1, min_confidence=0.5)
    assert result.fault_type is None


def test_severity_escalates_to_critical_for_severe_fault_far_over_threshold():
    probs = np.array([0.01, 0.01, 0.01, 0.97])
    result = decide(probs, CLASS_NAMES, anomaly_score=0.5, threshold=0.05, min_confidence=0.5)
    assert result.status == "ANOMALY"
    assert result.severity == "CRITICAL"


def test_borderline_normal_tiebreaker_does_not_override_genuine_anomaly():
    """The near-threshold NORMAL tie-breaker should only apply when the
    reconstruction error is marginally (< 10%) over threshold — a clearly
    anomalous score must still be flagged even if the classifier guesses
    'normal' with high confidence (models can disagree; that disagreement
    should surface, not be hidden)."""
    probs = np.array([0.92, 0.03, 0.03, 0.02])
    result = decide(probs, CLASS_NAMES, anomaly_score=0.20, threshold=0.05, min_confidence=0.5)
    assert result.status != "NORMAL"


def test_threshold_boundary_is_inclusive_and_deterministic():
    """score == threshold counts as a reconstruction anomaly (>=); a confident
    'normal' only overrides it inside the documented <10% tolerance band."""
    thr = 0.05
    sure_normal = np.array([0.95, 0.02, 0.02, 0.01])
    assert decide(sure_normal, CLASS_NAMES, thr * 0.999, thr).status == "NORMAL"
    assert decide(sure_normal, CLASS_NAMES, thr, thr).status == "NORMAL"          # inside tolerance band
    assert decide(sure_normal, CLASS_NAMES, thr * 1.0999, thr).status == "NORMAL"
    assert decide(sure_normal, CLASS_NAMES, thr * 1.1, thr).status == "UNKNOWN_ANOMALY"
    unsure = np.array([0.6, 0.2, 0.1, 0.1])                                        # normal but < 0.9
    assert decide(unsure, CLASS_NAMES, thr, thr).status == "UNKNOWN_ANOMALY"
    assert decide(unsure, CLASS_NAMES, thr * 0.999, thr).status == "NORMAL"


def test_confidence_boundary_for_known_fault():
    probs = np.array([0.1, 0.5, 0.2, 0.2])
    assert decide(probs, CLASS_NAMES, 0.0, 0.05, min_confidence=0.5).status == "ANOMALY"
    assert decide(probs, CLASS_NAMES, 0.0, 0.05, min_confidence=0.5001).status == "NORMAL"
