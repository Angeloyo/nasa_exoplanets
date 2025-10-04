from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import random
import numpy as np

app = FastAPI(title="NASA Exoplanets API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "models/ML/xgboost/complex_data"
CONSTELLATIONS = ["Draco", "Lyra", "Cygnus", "Andromeda", "Orion", "Perseus", 
                  "Cassiopeia", "Phoenix", "Pegasus", "Vega", "Centaurus", "Aquila"]
LABEL_MAP = {0: "None", 1: "Candidate", 2: "Exoplanet"}

class SinglePredictionRequest(BaseModel):
    PERIOD: float
    RADIUS: float
    DENSITY: float
    NUM_PLANETS: float
    DURATION: float
    TEFF: float
    DEPTH: float

def generate_name(index):
    const = random.choice(CONSTELLATIONS)
    num = random.randint(1, 99)
    letter = random.choice("ABCDEFGH")
    return f"{const}-{num}{letter}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        if 'DISPOSITION_ENCODED' in df.columns:
            df = df.drop(columns=['DISPOSITION_ENCODED'])
        
        # Store radius and density before scaling
        radius_values = df['RADIUS'].values if 'RADIUS' in df.columns else None
        density_values = df['DENSITY'].values if 'DENSITY' in df.columns else None
        
        model = joblib.load(MODEL_PATH / "model.pkl")
        scaler = joblib.load(MODEL_PATH / "scaler.pkl")
        
        X_scaled = scaler.transform(df)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        predictions = [
            {
                "name": generate_name(i),
                "prediction": LABEL_MAP[y_pred[i]],
                "confidence": round(float(y_proba[i][y_pred[i]] * 100), 1),
                "radius": float(radius_values[i]) if radius_values is not None else None,
                "density": float(density_values[i]) if density_values is not None else None
            }
            for i in range(len(y_pred))
        ]

        return {"status": "success", "model": "XGBoost", "total": len(predictions), "predictions": predictions}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/predict/sample")
async def predict_sample():
    try:
        # Load sample data
        # sample_path = Path(__file__).parent.parent / "data/sample/complex_data_sample_A.csv"
        sample_path = "prod_sample_data.csv"
        df = pd.read_csv(sample_path)
        
        if 'DISPOSITION_ENCODED' in df.columns:
            df = df.drop(columns=['DISPOSITION_ENCODED'])
        
        # Store radius and density before scaling
        radius_values = df['RADIUS'].values if 'RADIUS' in df.columns else None
        density_values = df['DENSITY'].values if 'DENSITY' in df.columns else None
        
        model = joblib.load(MODEL_PATH / "model.pkl")
        scaler = joblib.load(MODEL_PATH / "scaler.pkl")
        
        X_scaled = scaler.transform(df)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        predictions = [
            {
                "name": generate_name(i),
                "prediction": LABEL_MAP[y_pred[i]],
                "confidence": round(float(y_proba[i][y_pred[i]] * 100), 1),
                "radius": float(radius_values[i]) if radius_values is not None else None,
                "density": float(density_values[i]) if density_values is not None else None
            }
            for i in range(len(y_pred))
        ]

        return {"status": "success", "model": "XGBoost", "total": len(predictions), "predictions": predictions}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/predict/single")
async def predict_single(data: SinglePredictionRequest):
    try:
        # Convert request data to DataFrame with correct column order
        df = pd.DataFrame([[
            data.PERIOD,
            data.RADIUS,
            data.DENSITY,
            data.NUM_PLANETS,
            data.DURATION,
            data.TEFF,
            data.DEPTH
        ]], columns=['PERIOD', 'RADIUS', 'DENSITY', 'NUM_PLANETS', 'DURATION', 'TEFF', 'DEPTH'])
        
        # Load model and scaler
        model = joblib.load(MODEL_PATH / "model.pkl")
        scaler = joblib.load(MODEL_PATH / "scaler.pkl")
        
        # Scale and predict
        X_scaled = scaler.transform(df)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        prediction = {
            "name": generate_name(0),
            "prediction": LABEL_MAP[y_pred[0]],
            "confidence": round(float(y_proba[0][y_pred[0]] * 100), 1)
        }

        return {"status": "success", "model": "XGBoost", "prediction": prediction}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/")
def root():
    return {"message": "NASA Exoplanets API"}
