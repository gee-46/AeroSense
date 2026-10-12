from typing import Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.analytics import AnalyticsSummary, AnomalyTrendPoint
from app.services import analytics_service

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary, summary="Aggregate stats over all persisted predictions")
def summary(db: Session = Depends(get_db)):
    return analytics_service.get_summary(db)


@router.get("/fault-distribution", response_model=Dict[str, int],
            summary="Count of ANOMALY predictions per fault type")
def fault_distribution(db: Session = Depends(get_db)):
    return analytics_service.get_fault_distribution(db)


@router.get("/anomaly-trend", response_model=List[AnomalyTrendPoint],
            summary="Most recent anomaly scores over time (oldest first, max 200)")
def anomaly_trend(db: Session = Depends(get_db)):
    return analytics_service.get_anomaly_trend(db)
