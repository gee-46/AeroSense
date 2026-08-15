import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text

from app.db.database import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String, primary_key=True, default=_new_id)
    filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    device_id = Column(String, nullable=True)
    environment = Column(String, nullable=True)
    operating_condition = Column(String, nullable=True)
    rpm = Column(Float, nullable=True)

    duration_seconds = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)

    model_version = Column(String, nullable=False)
    autoencoder_version = Column(String, nullable=False)

    status = Column(String, nullable=False, index=True)  # NORMAL | ANOMALY | UNKNOWN_ANOMALY | ERROR
    fault_type = Column(String, nullable=True)
    fault_confidence = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=False)
    anomaly_threshold = Column(Float, nullable=False)
    severity = Column(String, nullable=True)

    class_probabilities = Column(JSON, nullable=True)
    explanation_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    processing_time_ms = Column(Float, nullable=False)
