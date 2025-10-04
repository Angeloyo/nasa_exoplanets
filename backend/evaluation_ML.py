import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Necesario para la prueba de sample

# --- Constantes y Mapeo ---
ROOT_MODELS_DIR = Path('backend/models')
ML_MODELS_DIR = ROOT_MODELS_DIR / 'ML'

# --- Rutas de datos (Añadidas para la prueba de sample) ---
DATA_SAMPLE_PATH = Path('data/sample/complex_data_sample.csv')

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
        # Aquí cargamos los datos de prueba específicos del modelo/dataset
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

    # El directorio de reporte debe coincidir con el identificador del dataset, aunque sea un sample.
    report_dir = base_output_dir / model_name.lower().replace(" ", "") / dataset_identifier
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f'evaluation_report_{model_name.lower()}_{dataset_identifier}.txt'
    
    print(f"\n3. Guardando resultados detallados en: {report_path}")
    with open(report_path, 'w') as report_file:
        # Añadimos el identificador completo para que se vea claro en el reporte
        full_model_name = f"{model_name} (TestData: {dataset_identifier})"
        generate_evaluation_report(model, X_test, y_test, full_model_name, report_file, report_dir)

    print("------------------------------------------------------")
    print(f"¡Evaluación de {model_name} en {dataset_identifier} completada!")
    return report_path


# --- Lógica Auxiliar para preparar el SAMPLE para la prueba ---
def prepare_sample_for_evaluation(sample_data_path: Path, model_name: str, base_dir: Path):
    """
    Simula el proceso de train.py: carga un modelo existente y el sample, 
    escala el sample y guarda un nuevo test_data.pkl en una carpeta 'sample' temporal 
    dentro del modelo, permitiendo su evaluación.
    """
    
    model_folder = model_name.lower().replace(" ", "")
    # Usamos 'complex_data' para cargar el escalador y features correctos.
    train_data_dir = base_dir / model_folder / 'complex_data' 
    
    sample_identifier = sample_data_path.stem # 'complex_data_sample'
    target_report_dir = base_dir / model_folder / sample_identifier 
    target_report_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PROCESO AUXILIAR] Preparando SAMPLE para evaluación...")

    try:
        # 1. Cargar el escalador y la lista de features del modelo original
        scaler = joblib.load(train_data_dir / 'scaler.pkl')
        feature_names = joblib.load(train_data_dir / 'features.pkl')
        
        # 2. Cargar el dataframe del sample
        df_sample = pd.read_csv(sample_data_path)
        
        X_sample = df_sample[feature_names]
        y_sample = df_sample['DISPOSITION_ENCODED']
        
        # 3. Escalar el sample (usando el escalador FIT de 'complex_data')
        X_sample_scaled = scaler.transform(X_sample)

        # 4. Guardar los datos escalados como el nuevo 'test_data.pkl' en la ruta de reporte
        joblib.dump((X_sample_scaled, y_sample), target_report_dir / 'test_data.pkl')
        
        # 5. Copiar el modelo original a la nueva carpeta de reporte para que lo encuentre load_artifacts
        import shutil
        shutil.copy(train_data_dir / 'model.pkl', target_report_dir / 'model.pkl')
        
        print(f"   -> Sample preparado y guardado en: {target_report_dir}")
        return sample_identifier

    except FileNotFoundError as e:
        print(f"   ERROR: Asegúrate de que el modelo '{model_name}' fue entrenado con 'complex_data' y que el archivo {sample_data_path} existe.")
        print(f"   Detalle: {e}")
        return None
    except Exception as e:
        print(f"   ERROR al preparar el sample: {e}")
        return None
# ----------------------------------------------------------------------


if __name__ == "__main__":
    
    print("======================================================")
    print("Iniciando Evaluación de las 4 Combinaciones de Modelos ML...")
    
    # Evaluación estándar
    evaluation_targets = [
        ('XGBoost', 'complex_data'),
    ]

    for model_name, dataset_id in evaluation_targets:
        evaluate_ml_model(model_name, dataset_id, ML_MODELS_DIR)
    print("======================================================")