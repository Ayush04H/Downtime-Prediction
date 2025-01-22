from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import pandas as pd
import joblib
import os
from logger import Logger
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Manufacturing Downtime Prediction API",
    description="API for predicting manufacturing equipment downtime",
    version="1.0.0"
)

# Initialize logger
log = Logger()

class ManufacturingData(BaseModel):
    Temperature: float = Field(..., ge=0, le=150, description="Temperature in Celsius")
    Vibration: float = Field(..., ge=0, le=1, description="Vibration level (0-1)")
    Pressure: float = Field(..., ge=0, le=200, description="Pressure level")
    Run_Time: float = Field(..., ge=0, le=24, description="Run time in hours")
    Oil_Level: float = Field(..., ge=0, le=1, description="Oil level (0-1)")
    Power_Consumption: float = Field(..., ge=0, le=100, description="Power consumption percentage")
    Product_Rate: float = Field(..., ge=0, le=200, description="Product rate per hour")
    Maintenance_Due: int = Field(..., ge=0, le=1, description="Maintenance due (0 or 1)")
    Quality_Score: float = Field(..., ge=0, le=100, description="Quality score (0-100)")

    @validator('*')
    def check_not_none(cls, v):
        if v is None:
            raise ValueError("All fields must have non-null values")
        return v

class PredictionResponse(BaseModel):
    prediction: str
    timestamp: str
    input_data: ManufacturingData

class PredictionHistory(BaseModel):
    predictions: list[PredictionResponse]

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = None
        self.predictions = []
        self.load_model()

    def load_model(self):
        try:
            model_path = 'Models'
            log.info("Loading model and preprocessors...")
            
            # Load model components
            self.model = joblib.load(os.path.join(model_path, 'decision_tree_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
            
            with open(os.path.join(model_path, 'feature_list.txt'), 'r') as f:
                self.features = f.read().splitlines()
            
            log.info("Model loaded successfully!")
            
        except Exception as e:
            log.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, input_data: ManufacturingData) -> str:
        try:
            # Convert input data to DataFrame
            input_dict = input_data.dict()
            input_df = pd.DataFrame([input_dict])[self.features]
            
            # Scale features
            scaled_input = self.scaler.transform(input_df)
            
            # Make prediction
            prediction_encoded = self.model.predict(scaled_input)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            # Create prediction response
            prediction_response = PredictionResponse(
                prediction=prediction,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                input_data=input_data
            )
            
            # Store prediction in history
            self.predictions.append(prediction_response)
            
            log.info(f"Prediction made: {prediction}")
            return prediction_response
            
        except Exception as e:
            log.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

# Initialize predictor
predictor = ModelPredictor()

@app.post("/predict", response_model=PredictionResponse)
async def predict_downtime(data: ManufacturingData):
    """
    Make a downtime prediction based on manufacturing data
    """
    try:
        prediction = predictor.predict(data)
        return prediction
    except Exception as e:
        log.error(f"API Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions", response_model=PredictionHistory)
async def get_predictions():
    """
    Get history of predictions made
    """
    try:
        return PredictionHistory(predictions=predictor.predictions)
    except Exception as e:
        log.error(f"API Error retrieving predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check if the API is running and model is loaded
    """
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)