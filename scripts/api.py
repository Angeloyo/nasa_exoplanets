from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import json

from evaluation_ML import evaluate_ml_model, ML_MODELS_DIR

app = FastAPI(title="NASA Exoplanets API")

# Carpeta donde se guardan los CSV subidos
UPLOAD_DIR = Path("data/processed/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/evaluate")
async def evaluate_csv(file: UploadFile = File(...)):
    # Guardar CSV subido
    file_path = UPLOAD_DIR / "uploaded.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # XGBoost
    model_name = "XGBoost"
    report_path = evaluate_ml_model(
        model_name=model_name,
        dataset_identifier="uploaded",
        base_output_dir=ML_MODELS_DIR
    )

    if report_path:
        report_dir = report_path.parent
        json_path = report_dir / f"predictions_{model_name.lower()}.json"
        if json_path.exists():
            with open(json_path, "r") as jf:
                predictions = json.load(jf)
        else:
            predictions = "JSON no encontrado"
    else:
        predictions = "Error en evaluación ML"

    return {"status": "ok", "model": model_name, "predictions": predictions}


@app.get("/")
def root():
    return {"message": "NASA Exoplanets API"}
