from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging
import uvicorn
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('occupancy_service.log'),
        logging.StreamHandler()
    ]
)

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
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logging.info("Model and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model or scaler: {str(e)}")
            raise
    
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
    predictor = OccupancyPredictor(
        model_path='models/xgboost_model.joblib',
        scaler_path='models/scaler.joblib'
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