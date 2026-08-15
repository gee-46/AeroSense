from sqlalchemy.orm import Session

from app.db import repository


def get_summary(db: Session) -> dict:
    return repository.summary_stats(db)


def get_fault_distribution(db: Session) -> dict:
    return repository.fault_distribution(db)


def get_anomaly_trend(db: Session, limit: int = 200) -> list:
    return repository.anomaly_trend(db, limit=limit)
