import time

import numpy as np

from app.core.config import settings
from app.core.exceptions import InferenceError, InvalidAudioError, AeroSenseError
from app.ml.model_loader import get_model_registry
from app.ml.inference import classify, autoencoder_scores
from app.services.preprocessing_service import load_and_segment, segments_to_model_batch
from app.services.anomaly_service import decide
from app.services.explainability_service import explain
from app.core.logging import logger


def run_full_analysis(audio_path: str, prediction_id: str, with_explanation: bool = True) -> dict:
    """
    Full pipeline for POST /api/v1/audio/analyze.
    Returns a dict matching app.schemas.prediction.PredictionResult (minus
    the request-context fields the route layer fills in) plus internal
    timing. Every number in the return value comes from a real forward
    pass through a trained checkpoint — nothing here is fabricated.
    """
    t0 = time.perf_counter()
    registry = get_model_registry()

    segments = load_and_segment(audio_path)
    batch = segments_to_model_batch(segments, registry.feature_mean, registry.feature_std)

    try:
        cls_result = classify(registry, batch)
    except AeroSenseError:
        raise  # e.g. ModelUnavailableError — propagate its real status code (503), don't mask it
    except Exception as e:
        logger.exception(f"Classifier inference failed for {audio_path}")
        raise InferenceError(f"Classifier inference failed: {e}")

    try:
        ae_errors = autoencoder_scores(registry, batch)
    except AeroSenseError:
        raise
    except Exception as e:
        logger.exception(f"Autoencoder inference failed for {audio_path}")
        raise InferenceError(f"Autoencoder inference failed: {e}")

    mean_probs = cls_result["mean_probs"]
    mean_anomaly_score = float(np.mean(ae_errors))

    decision = decide(
        mean_class_probs=mean_probs,
        class_names=registry.class_names,
        anomaly_score=mean_anomaly_score,
        threshold=registry.anomaly_threshold,
    )

    class_probabilities = {c: float(p) for c, p in zip(registry.class_names, mean_probs)}

    explanation = None
    if with_explanation and decision.status != "NORMAL":
        try:
            pred_idx = int(mean_probs.argmax())
            # Use the segment whose class-probability for the predicted
            # class is highest -> the clearest single example to attribute.
            per_seg_target_prob = cls_result["segment_probs"][:, pred_idx]
            best_seg_idx = int(np.argmax(per_seg_target_prob))
            explanation = explain(registry, batch[best_seg_idx], pred_idx, prediction_id)
        except Exception as e:
            logger.warning(f"Explainability generation failed (non-fatal): {e}")
            explanation = None

    processing_time_ms = (time.perf_counter() - t0) * 1000

    return {
        "status": decision.status,
        "fault_type": decision.fault_type,
        "fault_confidence": decision.fault_confidence,
        "anomaly_score": decision.anomaly_score,
        "anomaly_threshold": decision.anomaly_threshold,
        "severity": decision.severity,
        "class_probabilities": class_probabilities,
        "explanation": explanation,
        "processing_time_ms": processing_time_ms,
        "model_version": registry.classifier_meta.get("version", "unknown"),
        "autoencoder_version": registry.autoencoder_meta.get("version", "unknown"),
        "n_segments_analyzed": len(segments),
    }
