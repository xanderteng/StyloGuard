from fastapi import APIRouter

from app.model.predictor import predict_authorship
from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    return predict_authorship(request.claimed_author, request.text)
