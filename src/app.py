import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from config import Config
import sklearn

# Initialize FastAPI app
app = FastAPI(version="1.0.0")

# Load the pre-trained model and scaler
MODEL_PATH = os.path.join(Config.MODEL_DIR, Config.MODEL_NAME)
SCALER_PATH = os.path.join(Config.MODEL_DIR, Config.SCALER_NAME)

model = None
scaler = None

class TitanicPassenger(BaseModel):
    pclass: int
    age: float
    fare: float
    sex: str
    embarked: str

    class Config:
        schema_extra = {
            "example": {
                "pclass": 1,
                "age": 29.0,
                "fare": 211.33,
                "sex": "female",
                "embarked": "S"
            },
            "description": {
                "pclass": "Passenger Class (1, 2, 3)",
                "age": "Age in years (29.0)",
                "fare": "fare (211.33)",
                "sex": "Gender ('male' or 'female')",
                "embarked": "Port of Embarkation ('C','Q','S')"
            }
        }

def load_models():
    """Load the pre-trained model and scaler"""
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def preprocess_input(passenger: TitanicPassenger) -> np.ndarray:
    """Preprocess the input data to match the model's requirements"""
    # create a dataFrame with the input data
    df = pd.DataFrame([passenger.model_dump()])
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['sex', 'embarked'])
    
    # ensure all necessary columns exist
    required_columns = Config.FEATURES
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
            
    # reorder columns to match training data
    df = df[required_columns]
    
    # scale the features
    scaled_features = scaler.transform(df)
    
    return scaled_features

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
def read_root():

    return {"message": "Welcome to the Titanic Survival Prediction api :) \n Go to /docs to see the documentation"}

@app.post("/predict")
def predict_survival(passenger: TitanicPassenger):
    """Predict survival probability for a Titanic passenger"""
    try:
        # pre process input
        processed_input = preprocess_input(passenger)
        
        # Make prediction
        survival_prob = model.predict_proba(processed_input)[0][1]
        survival_prediction = model.predict(processed_input)[0]
        
        return {
            "survival_prediction": bool(survival_prediction),
            "survival_probability": float(survival_prob),
            "message": "Passenger would likely survive" if survival_prediction else "Passenger would likely not survive"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=4222, reload=True)