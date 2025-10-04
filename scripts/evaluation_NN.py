import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# --- Constantes y Mapeo ---
ROOT_MODELS_DIR = Path('models')
NN_MODELS_DIR = ROOT_MODELS_DIR / 'NN'

LABEL_MAP = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE (PC)',
    2: 'CONFIRMED (CP)'
}


def load_nn_artifacts_for_evaluation(dataset_identifier: str, base_dir: Path):
    """Carga el modelo de Red Neuronal (Keras) y los datos de prueba desde la ruta modular."""
    
    model_name = 'NN Custom'  # Nombre lógico para el informe
    data_dir = base_dir / 'nn_custom' / dataset_identifier 
    
    model_file = 'model.keras'
    test_data_file = 'test_data.pkl'

    print(f"1. Cargando artefactos de la ruta: {data_dir}")
    
    try:
        # Cargar el modelo y los datos
        nn_model = load_model(data_dir / model_file)
        X_test_scaled, y_test_raw = joblib.load(data_dir / test_data_file)
        print(f"   Modelo '{model_name}' y datos cargados. Tamaño: {len(y_test_raw)} muestras.")
        return nn_model, X_test_scaled, y_test_raw
    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que train_NN.py se ejecutó correctamente.")
        print(f"Ruta faltante: {e}")
        return None, None, None


def generate_nn_evaluation_report(model, X_test, y_test_true, model_name: str, report_file, report_dir: Path):
    """Genera métricas globales + listado completo de predicciones con confianza y crea también un JSON."""
    
    print(f"\n2. Evaluando: {model_name}...")

    # --- Predicciones ---
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # === MÉTRICAS GLOBALES ===
    accuracy = accuracy_score(y_test_true, y_pred)
    cm = confusion_matrix(y_test_true, y_pred)
    class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test_true, y_pred, target_names=class_names)

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

    for i, (pred, probs) in enumerate(zip(y_pred, y_pred_proba)):
        clase_predicha = LABEL_MAP[pred]
        conf = float(probs[pred] * 100)
        report_file.write(f"{i+1} {clase_predicha} {conf:.2f}%\n")

        predictions_json.append({
            "id": i + 1,
            "prediction": clase_predicha,
            "confidence": conf
        })

    # === Guardar JSON ===
    json_path = report_dir / f"predictions_nn_{report_dir.name}.json"
    with open(json_path, "w") as jf:
        json.dump(predictions_json, jf, indent=2)

    print(f"   -> Informe completo guardado en {report_dir}")
    print(f"   -> JSON de predicciones generado: {json_path}")


def evaluate_nn_model(dataset_identifier: str, base_output_dir: Path):
    """Carga, evalúa y guarda el informe de la red neuronal."""
    model_name = f'NN Custom ({dataset_identifier})'
    
    nn_model, X_test, y_test_true = load_nn_artifacts_for_evaluation(dataset_identifier, base_output_dir)
    if nn_model is None:
        return None

    report_dir = base_output_dir / 'nn_custom' / dataset_identifier
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f'evaluation_report_nn_{dataset_identifier}.txt'
    
    print(f"\n3. Guardando resultados detallados en: {report_path}")
    with open(report_path, 'w') as report_file:
        generate_nn_evaluation_report(nn_model, X_test, y_test_true, model_name, report_file, report_dir)

    print("------------------------------------------------------")
    print(f"¡Evaluación de Red Neuronal en {dataset_identifier} completada!")
    return report_path


if __name__ == "__main__":
    
    datasets_to_evaluate = [
        'simple_data',
        'complex_data',
    ]
    
    print("======================================================")
    print("Iniciando Evaluación de Redes Neuronales...")
    
    for dataset_id in datasets_to_evaluate:
        evaluate_nn_model(
            dataset_identifier=dataset_id,
            base_output_dir=NN_MODELS_DIR
        )

    print("======================================================")
