from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import subprocess

from evaluation_ML import evaluate_ml_model, ML_MODELS_DIR
from evaluation_NN import evaluate_nn_model, NN_MODELS_DIR

app = FastAPI(title="NASA Exoplanets API")

UPLOAD_DIR = Path("data/processed/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/evaluate")
async def evaluate_csv(file: UploadFile = File(...)):
    # Guardar CSV en la carpeta de processed
    file_path = UPLOAD_DIR / "uploaded.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ejecutar el preprocesado para generar test_data.pkl
    try:
        subprocess.run(["python", "scripts/preprocess.py", str(file_path)], check=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Fallo en el preprocesado: {e}"}

    reports = {}

    # Evaluar ML
    for model_name in ["RandomForest", "XGBoost"]:
        report_path = evaluate_ml_model(
            model_name=model_name,
            dataset_identifier="uploaded",
            base_output_dir=ML_MODELS_DIR
        )
        reports[model_name] = str(report_path) if report_path else "Error"

    # Evaluar NN
    report_path = evaluate_nn_model(
        dataset_identifier="uploaded",
        base_output_dir=NN_MODELS_DIR
    )
    reports["NN Custom"] = str(report_path) if report_path else "Error"

    return {"status": "ok", "reports": reports}


@app.get("/")
def root():
    return {"message": "NASA Exoplanets API"}
