from fastapi import APIRouter, HTTPException
from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel
from src.metrics import Metrics
from time import time
import logging
from src.data_model import Response, Request, HTTPError
from src.exceptions import UserNotFoundException, PredictionException

logging.basicConfig(level=logging.DEBUG)

feature_store = FeatureStore()
model = BasketModel()
metrics = Metrics()  # Prometeus' object


router = APIRouter(prefix="/predict")


@router.post(
    "/",
    response_model=Response,  # Estructura de los datos que queremos devolver
    responses={  # Respuestas esperadas
        200: {"model": Response},
        500: {"model": HTTPError, "description": "Problems processing the text"},
    },
)
async def predict(request: Request) -> Response:
    metrics.increase_requests()
    try:
        start_time_predict = time()
        features = feature_store.get_features(request.user_id)
        pred = model.predict(features.values)
        print("pred")

    except UserNotFoundException as exception:
        metrics.increase_user_not_found_errors()
        logging.error("User: %s, Message: %s", request.user_id, exception.message)
        raise HTTPException(status_code=500, detail="User not found") from exception

    except PredictionException as exception:
        metrics.increase_unknown_errors()
        logging.error("User: %s, Message: %s", request.user_id, exception.message)
        raise HTTPException(status_code=500, detail="Prediction not completed")

    except Exception as exception:
        metrics.increase_unknown_errors()
        logging.error("User: %s, Unknown exception", request.user_id)
        raise HTTPException(status_code=500, detail="Unknown exception") from exception

    metrics.observe_predict_duration(start_time_predict)
    print("pred", pred)
    return Response(basket_price=pred.mean())
