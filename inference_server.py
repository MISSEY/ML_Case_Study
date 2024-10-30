from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging
import uvicorn
from typing import Dict
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('occupancy_service.log'),
        logging.StreamHandler()
    ]
)

def prepare_args():
    """
    prepare arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',default=None,required=False,help='model path')

    return parser

# Initialize FastAPI app
app = FastAPI(title="Room Occupancy Prediction Service")

class SensorData(BaseModel):
    timestamp: datetime
    temperature: float
    humidity: float
    light: float
    co2: float
    humidity_ratio: float
    
    # Validate sensor readings
    @field_validator('temperature')
    def validate_temperature(cls, v):
        if not 10 <= v <= 50:  # reasonable temperature range in Celsius
            raise ValueError("Temperature out of expected range (10-50°C)")
        return v
    
    @field_validator('humidity')
    def validate_humidity(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Humidity must be between 0-100%")
        return v
    
    @field_validator('co2')
    def validate_co2(cls, v):
        if not 0 <= v <= 5000:  # typical CO2 range in ppm
            raise ValueError("CO2 out of expected range (0-5000 ppm)")
        return v

class OccupancyPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize the predictor with model and scaler"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler=None
        self.last_loaded = None
        self.load_model()

    def load_model(self):
        """Load model and update timestamp"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.last_loaded = datetime.now()
            print(f"Model loaded successfully at {self.last_loaded}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_model(self):
        """Get current model instance"""
        return self.model
    
    def create_features(self, data: SensorData) -> pd.DataFrame:
        """Create features from sensor data"""
        try:
            # Extract time features
            hour = data.timestamp.hour
            day_of_week = data.timestamp.weekday()
            
            
            # Create feature dictionary
            features = {
                'Temperature': data.temperature,
                'Humidity': data.humidity,
                'Light': data.light,
                'CO2': data.co2,
                'HumidityRatio': data.humidity_ratio,
            }
            
            features = pd.DataFrame(features, index=[0])
            
            # Create one-hot encoded columns for hour (0-23)
            for h in range(24):
                features[f'hour_{h}'] = bool(1) if hour == h else bool(0)
                
            # Create one-hot encoded columns for day_of_week (0-6)
            for d in range(7):
                features[f'day_of_week_{d}'] = bool(1) if day_of_week == d else bool(0)

            return features
            
        except Exception as e:
            logging.error(f"Error creating features: {str(e)}")
            raise
    
    def predict(self, data: SensorData) -> Dict:
        """Generate prediction from sensor data"""
        try:
            # Create features
            features = self.create_features(data)
            
            # Scale numerical features
            numerical_features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
            features[numerical_features] = self.scaler.transform(features[numerical_features])
            
            # Generate prediction
            prob = self.model.predict_proba(features)[0, 1]
            prediction = int(prob >= 0.5)
            
            # Log prediction details
            logging.info(f"Prediction generated: {prediction} (probability: {prob:.3f})")
            
            return {
                "timestamp": data.timestamp,
                "occupied": prediction,
                "probability": float(prob),
                "status": "success"
            }
            
        except Exception as e:
            logging.error(f"Error generating prediction: {str(e)}")
            raise

# Initialize predictor
try:
    parser =prepare_args()
    args = parser.parse_args()
    path = args.model_path
    if path is None:
        path = 'models'
    
    predictor = OccupancyPredictor(
        model_path=f'{path}/xgboost_model.joblib',
        scaler_path=f'{path}/scaler.joblib'
    )
except Exception as e:
    logging.error(f"Failed to initialize predictor: {str(e)}")
    raise

@app.post("/predict")
async def predict_occupancy(data: SensorData):
    """Endpoint for generating occupancy predictions"""
    try:
        # Generate prediction
        result = predictor.predict(data)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)