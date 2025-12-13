from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import MODEL_VERSION, predict_output, model
import pickle
import pandas as pd

app = FastAPI()

#Human readable
@app.get("/")
def Home():
    return {
        'message':'Insurance Premium Prediction API'
    }

#machine readable
@app.get("/health")
def health_check():
    return {
        'status':'OK',
        'version': MODEL_VERSION,
        'model_loaded': model is not True
    }

@app.post("/predict",response_model=PredictionResponse)
def predict_premium(data: UserInput):

    user_input = {
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation
    }
    try:
        prediction = predict_output(user_input)
        return JSONResponse(status_code=200, content={'predicted_output':prediction})
    except Exception as e:
        return JSONResponse(status_code=500,content=str(e))
