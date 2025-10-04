from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import json

from evaluation_ML import evaluate_ml_model, ML_MODELS_DIR
from evaluation_NN import evaluate_nn_model, NN_MODELS_DIR

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

    reports = {}

    # Evaluar modelos clásicos (ML)
    for model_name in ["RandomForest", "XGBoost"]:
        report_path = evaluate_ml_model(
            model_name=model_name,
            dataset_identifier="uploaded",  # Se usará para buscar el CSV limpio
            base_output_dir=ML_MODELS_DIR
        )

        if report_path:
            # Leer el JSON generado
            report_dir = report_path.parent
            json_path = report_dir / f"predictions_{model_name.lower()}.json"
            if json_path.exists():
                with open(json_path, "r") as jf:
                    reports[model_name] = json.load(jf)
            else:
                reports[model_name] = "JSON no encontrado"
        else:
            reports[model_name] = "Error en evaluación ML"

    # Evaluar red neuronal (NN)
    report_path = evaluate_nn_model(
        dataset_identifier="uploaded",
        base_output_dir=NN_MODELS_DIR
    )

    if report_path:
        report_dir = report_path.parent
        json_path = report_dir / f"predictions_nn_uploaded.json"
        if json_path.exists():
            with open(json_path, "r") as jf:
                reports["NN Custom"] = json.load(jf)
        else:
            reports["NN Custom"] = "JSON no encontrado"
    else:
        reports["NN Custom"] = "Error en evaluación NN"

    return {"status": "ok", "reports": reports}


@app.get("/")
def root():
    return {"message": "NASA Exoplanets API en funcionamiento"}
