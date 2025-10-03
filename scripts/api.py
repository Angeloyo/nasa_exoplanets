from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import pandas as pd

# Importa tus funciones de evaluación
from evaluation_ML import evaluate_ml_model, ML_MODELS_DIR
from evaluation_NN import evaluate_nn_model, NN_MODELS_DIR

app = FastAPI(title="NASA Exoplanets API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), model_type: str = "ML", dataset: str = "simple_data"):
    """
    Sube un CSV, lo guarda temporalmente y ejecuta la evaluación.
    - model_type: "ML" o "NN"
    - dataset: "simple_data" o "complex_data"
    """

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Validar CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"El archivo no es un CSV válido: {str(e)}"}

    # Ejecutar la evaluación
    if model_type.upper() == "ML":
        # Evaluar RandomForest y XGBoost en el dataset indicado
        reports = []
        for model_name in ["RandomForest", "XGBoost"]:
            report_path = evaluate_ml_model(model_name, dataset_identifier=dataset, base_output_dir=ML_MODELS_DIR)
            reports.append(str(report_path) if report_path else f"{model_name} no disponible")
        return {"status": "ok", "model_type": "ML", "reports": reports}

    elif model_type.upper() == "NN":
        report_path = evaluate_nn_model(dataset_identifier=dataset, base_output_dir=NN_MODELS_DIR)
        return {"status": "ok", "model_type": "NN", "report": str(report_path)}

    else:
        return {"error": "Tipo de modelo no válido. Usa 'ML' o 'NN'."}


@app.get("/")
def root():
    return {"message": "NASA Exoplanets API lista 🚀"}
