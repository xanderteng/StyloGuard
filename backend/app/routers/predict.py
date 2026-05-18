from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.model.predictor import predict_authorship
from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> dict:
    return predict_authorship(db, request.claimed_author, request.text)
