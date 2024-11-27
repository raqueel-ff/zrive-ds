import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time
from feature_store import FeatureStore
# from basket_model import BasketModel
# from exceptions import UserNotFoundException
# from exceptions import PredictionException

# Create an instance of FastAPI
app = FastAPI()

logging.basicConfig(filename="metrics.txt", level=logging.INFO)
logger = logging.getLogger()

#Feature Store and model inicialization
feature_store = FeatureStore()
# basket_model = BasketModel()

users = ['a','b']
@app.get("/")
def status():
    return {"status": "OK"}

@app.post("/predict/")
def create_item(user: str):
    
    #Escribir en el metrics.txt el usuario pedido y el start time
    start_time = time.time()

    try:
        features = feature_store.get_features(user)
        logger.info(f"User ID: {user}, Status: 200 OK, Start time: {start_time}, Features: {features}")
        return {"message": "User found", "user": user, "features": features}
        #features = feature_store.get_features(user)
        # prediction = basket_model.predict(features)
        # prediction_price = prediction[0]
        # latency = time.time() - start_time
    except HTTPException as e:
        logger.error(f"User ID: {user}, Error: User not found, Status Code: 404, Start time: {start_time}")
        raise HTTPException(status_code=404, detail=f'User {user} not found')
    
        
    #     return {"predicted_price": prediction_price}

    # except UserNotFoundException as e:
    #     logger.error(f"User not found: {str(e)}")
    #     raise HTTPException(status_code=404, detail="User not found")
    # except PredictionException as e:
    #     logger.error(f"Prediction error: {str(e)}")
    #     raise HTTPException(status_code=500, detail="Error during prediction")
    # except Exception as e:
    #     logger.error(f"General error: {str(e)}")
    #     raise HTTPException(status_code=500, detail="Internal Server Error")

# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)