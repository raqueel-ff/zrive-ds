import os
import joblib
import numpy as np

from exceptions import PredictionException


MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "bin", "model.joblib")
)


class BasketModel:
    def __init__(self):
        self.model = joblib.load(MODEL)
        print('init well')

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            print('intento hacer predict')
            pred = self.model.predict(features)
        except Exception as exception:
            raise PredictionException("Error during model inference") from exception
        return pred
