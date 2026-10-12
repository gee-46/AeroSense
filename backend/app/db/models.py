import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.types import TypeDecorator

from app.db.database import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on SQLite.

    SQLite has no timezone-aware type: values written as UTC come back naive,
    and Pydantic then serializes them without a "Z"/offset. Browsers parse such
    strings as *local* time, which shifts every timestamp by the viewer's UTC
    offset. Normalize to UTC on write and re-attach UTC on read.
    """
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String, primary_key=True, default=_new_id)
    filename = Column(String, nullable=False)
    # Uploaded audio is deleted after analysis (see audio route), so this is
    # kept only for schema compatibility and is always NULL for new rows.
    stored_path = Column(String, nullable=True)
    timestamp = Column(UTCDateTime, default=lambda: datetime.now(timezone.utc), index=True)

    device_id = Column(String, nullable=True)
    environment = Column(String, nullable=True)
    operating_condition = Column(String, nullable=True)
    rpm = Column(Float, nullable=True)

    duration_seconds = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=True)
    audio_format = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)

    model_version = Column(String, nullable=False)
    autoencoder_version = Column(String, nullable=False)

    status = Column(String, nullable=False, index=True)  # NORMAL | ANOMALY | UNKNOWN_ANOMALY | ERROR
    fault_type = Column(String, nullable=True)
    fault_confidence = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=False)
    anomaly_threshold = Column(Float, nullable=False)
    severity = Column(String, nullable=True)

    class_probabilities = Column(JSON, nullable=True)
    explanation_path = Column(String, nullable=True)   # URL path of the Grad-CAM overlay
    explanation = Column(JSON, nullable=True)          # full explanation (freq/time ranges, notes)
    error_message = Column(Text, nullable=True)

    processing_time_ms = Column(Float, nullable=False)
