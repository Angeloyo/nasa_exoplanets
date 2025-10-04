import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import json

# --- Constantes y Mapeo ---
ROOT_MODELS_DIR = Path('models')
ML_MODELS_DIR = ROOT_MODELS_DIR / 'ML'

LABEL_MAP = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE (PC)',
    2: 'CONFIRMED (CP)'
}

def load_artifacts_for_evaluation(model_name: str, dataset_identifier: str, base_dir: Path):
    """Carga el modelo y los datos de prueba."""
    model_folder = model_name.lower().replace(" ", "")
    data_dir = base_dir / model_folder / dataset_identifier 
    
    model_file = 'model.pkl'
    test_data_file = 'test_data.pkl'

    print(f"1. Cargando artefactos de la ruta: {data_dir}")
    
    try:
        model = joblib.load(data_dir / model_file)
        X_test, y_test = joblib.load(data_dir / test_data_file)
        print(f"   Modelo '{model_name}' y datos cargados. Tamaño: {len(y_test)} muestras.")
        return model, X_test, y_test
    except FileNotFoundError as e:
        print(f"ERROR: No se encontró el archivo: {e}")
        return None, None, None


def generate_evaluation_report(model, X_test, y_test, model_name: str, report_file, report_dir: Path):
    """Genera métricas globales + listado completo de predicciones con su confianza y crea también un JSON."""
    
    print(f"\n2. Evaluando: {model_name}...")

    # Predicciones y probabilidades
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    else:
        y_pred_proba = None

    # === MÉTRICAS GLOBALES ===
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test, y_pred, target_names=class_names)

    # === Escribir TXT ===
    report_file.write(f"RESULTADOS DEL MODELO: {model_name}\n")
    report_file.write(f"{'='*70}\n")
    report_file.write(f"Accuracy General: {accuracy:.4f}\n\n")
    report_file.write(f"Matriz de Confusión:\n{cm}\n\n")
    report_file.write("Informe de Clasificación (Precision, Recall, F1-Score):\n")
    report_file.write(report + "\n")
    report_file.write(f"{'='*70}\n")
    report_file.write("Predicciones individuales:\n\n")

    # === Construcción del JSON ===
    predictions_json = []

    if y_pred_proba is not None:
        for i, probs in enumerate(y_pred_proba):
            clase_predicha = LABEL_MAP[y_pred[i]]
            conf = float(probs[y_pred[i]] * 100)
            report_file.write(f"{i+1} {clase_predicha} {conf:.2f}%\n")

            predictions_json.append({
                "id": i + 1,
                "prediction": clase_predicha,
                "confidence": conf
            })
    else:
        for i, pred in enumerate(y_pred):
            clase_predicha = LABEL_MAP[pred]
            report_file.write(f"{i+1} {clase_predicha} -\n")

            predictions_json.append({
                "id": i + 1,
                "prediction": clase_predicha,
                "confidence": None
            })

    # === Guardar JSON ===
    json_path = report_dir / f"predictions_{model_name.lower().replace(' ', '_')}.json"
    with open(json_path, "w") as jf:
        json.dump(predictions_json, jf, indent=2)

    print(f"   -> Informe completo guardado en {report_dir}")
    print(f"   -> JSON de predicciones generado: {json_path}")


def evaluate_ml_model(model_name: str, dataset_identifier: str, base_output_dir: Path):
    """Carga, evalúa y guarda el informe."""
    model, X_test, y_test = load_artifacts_for_evaluation(model_name, dataset_identifier, base_output_dir)
    if model is None:
        return None

    report_dir = base_output_dir / model_name.lower().replace(" ", "") / dataset_identifier
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f'evaluation_report_{model_name.lower()}_{dataset_identifier}.txt'
    
    print(f"\n3. Guardando resultados detallados en: {report_path}")
    with open(report_path, 'w') as report_file:
        generate_evaluation_report(model, X_test, y_test, model_name, report_file, report_dir)

    print("------------------------------------------------------")
    print(f"¡Evaluación de {model_name} en {dataset_identifier} completada!")
    return report_path


if __name__ == "__main__":
    evaluation_targets = [
        ('RandomForest', 'simple_data'),
        ('RandomForest', 'complex_data'),
        ('XGBoost', 'simple_data'),
        ('XGBoost', 'complex_data'),
    ]

    print("======================================================")
    print("Iniciando Evaluación de las 4 Combinaciones de Modelos ML...")
    for model_name, dataset_id in evaluation_targets:
        evaluate_ml_model(model_name, dataset_id, ML_MODELS_DIR)
    print("======================================================")
