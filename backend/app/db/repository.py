from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import Prediction
from app.core.exceptions import DatabaseError, NotFoundError
from app.core.logging import logger


def create_prediction(db: Session, **fields) -> Prediction:
    try:
        pred = Prediction(**fields)
        db.add(pred)
        db.commit()
        db.refresh(pred)
        return pred
    except Exception as e:
        db.rollback()
        # Do not echo driver/SQL text to clients; it is logged by the caller.
        logger.exception("Failed to persist prediction")
        raise DatabaseError("Failed to persist prediction")


def get_prediction(db: Session, prediction_id: str) -> Prediction:
    pred = db.query(Prediction).filter(Prediction.prediction_id == prediction_id).first()
    if pred is None:
        raise NotFoundError(f"Prediction '{prediction_id}' not found")
    return pred


def list_predictions(
    db: Session, limit: int = 50, offset: int = 0,
    status: Optional[str] = None, device_id: Optional[str] = None,
):
    query = db.query(Prediction)
    if status:
        query = query.filter(Prediction.status == status)
    if device_id:
        query = query.filter(Prediction.device_id == device_id)
    total = query.count()
    items = query.order_by(Prediction.timestamp.desc()).offset(offset).limit(limit).all()
    return total, items


def delete_prediction(db: Session, prediction_id: str) -> Optional[str]:
    """Delete a record. Returns its Grad-CAM URL path (if any) so the caller can
    remove the generated image too, instead of orphaning it on disk."""
    pred = get_prediction(db, prediction_id)
    explanation_path = pred.explanation_path
    try:
        db.delete(pred)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("Failed to delete prediction")
        raise DatabaseError("Failed to delete prediction")
    return explanation_path


def fault_distribution(db: Session) -> dict:
    rows = (
        db.query(Prediction.fault_type, func.count(Prediction.prediction_id))
        .filter(Prediction.status.in_(["ANOMALY"]))
        .group_by(Prediction.fault_type)
        .all()
    )
    return {ft or "unspecified": count for ft, count in rows}


def status_distribution(db: Session) -> dict:
    rows = db.query(Prediction.status, func.count(Prediction.prediction_id)).group_by(Prediction.status).all()
    return {status: count for status, count in rows}


def anomaly_trend(db: Session, limit: int = 200):
    rows = (
        db.query(Prediction.timestamp, Prediction.anomaly_score, Prediction.status)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [{"timestamp": t, "anomaly_score": s, "status": st} for t, s, st in reversed(rows)]


def summary_stats(db: Session) -> dict:
    total = db.query(func.count(Prediction.prediction_id)).scalar() or 0
    avg_latency = db.query(func.avg(Prediction.processing_time_ms)).scalar() or 0.0
    avg_anomaly_score = db.query(func.avg(Prediction.anomaly_score)).scalar() or 0.0
    return {
        "total_predictions": total,
        "avg_processing_time_ms": round(float(avg_latency), 2),
        "avg_anomaly_score": round(float(avg_anomaly_score), 4),
        "status_distribution": status_distribution(db),
    }
