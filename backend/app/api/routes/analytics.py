from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services import analytics_service

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary", summary="Aggregate stats over all persisted predictions")
def summary(db: Session = Depends(get_db)):
    return analytics_service.get_summary(db)


@router.get("/fault-distribution", summary="Count of predictions per fault type")
def fault_distribution(db: Session = Depends(get_db)):
    return analytics_service.get_fault_distribution(db)


@router.get("/anomaly-trend", summary="Recent anomaly scores over time")
def anomaly_trend(db: Session = Depends(get_db)):
    return analytics_service.get_anomaly_trend(db)
