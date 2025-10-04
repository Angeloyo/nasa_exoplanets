from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import joblib
import random

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
        
        model = joblib.load(MODEL_PATH / "model.pkl")
        scaler = joblib.load(MODEL_PATH / "scaler.pkl")
        
        X_scaled = scaler.transform(df)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        predictions = [
            {
                "name": generate_name(i),
                "prediction": LABEL_MAP[y_pred[i]],
                "confidence": round(float(y_proba[i][y_pred[i]] * 100), 1)
            }
            for i in range(len(y_pred))
        ]

        return {"status": "success", "model": "XGBoost", "total": len(predictions), "predictions": predictions}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/")
def root():
    return {"message": "NASA Exoplanets API"}
