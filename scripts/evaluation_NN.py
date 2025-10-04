import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
# Importaciones auxiliares necesarias para el bloque de prueba del sample
import pandas as pd
import shutil 

# --- Constantes y Mapeo ---
ROOT_MODELS_DIR = Path('models')
NN_MODELS_DIR = ROOT_MODELS_DIR / 'NN'

# --- Rutas de datos (Añadidas para la prueba de sample) ---
DATA_SAMPLE_PATH = Path('data/sample/complex_data_sample.csv')

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
    y_pred = np.argmax(y_pred_proba, axis=1) # Convertir de OHE a etiquetas 0, 1, 2

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
    model_name = f'NN Custom (TestData: {dataset_identifier})'
    
    nn_model, X_test, y_test_true = load_nn_artifacts_for_evaluation(dataset_identifier, base_output_dir)
    if nn_model is None:
        return None

    # El directorio de reporte debe coincidir con el identificador del dataset, aunque sea un sample.
    report_dir = base_output_dir / 'nn_custom' / dataset_identifier
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f'evaluation_report_nn_{dataset_identifier}.txt'
    
    print(f"\n3. Guardando resultados detallados en: {report_path}")
    with open(report_path, 'w') as report_file:
        generate_nn_evaluation_report(nn_model, X_test, y_test_true, model_name, report_file, report_dir)

    print("------------------------------------------------------")
    print(f"¡Evaluación de Red Neuronal en {dataset_identifier} completada!")
    return report_path


# --- Lógica Auxiliar para preparar el SAMPLE para la prueba ---
def prepare_nn_sample_for_evaluation(sample_data_path: Path, base_dir: Path):
    """
    Carga el escalador y features del modelo NN 'complex_data', escala el sample 
    y guarda los artefactos necesarios en la ruta del sample para su evaluación.
    """
    
    model_name_folder = 'nn_custom'
    # Usamos 'complex_data' para cargar el escalador y features correctos.
    train_data_dir = base_dir / model_name_folder / 'complex_data' 
    
    sample_identifier = sample_data_path.stem # 'complex_data_sample'
    target_report_dir = base_dir / model_name_folder / sample_identifier 
    target_report_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PROCESO AUXILIAR] Preparando SAMPLE para evaluación de NN...")

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

        # 4. Guardar los datos escalados (X_test) y las etiquetas crudas (y_test_raw) 
        joblib.dump((X_sample_scaled, y_sample), target_report_dir / 'test_data.pkl')
        
        # 5. Copiar el modelo Keras original a la nueva carpeta de reporte para que lo encuentre load_nn_artifacts
        shutil.copy(train_data_dir / 'model.keras', target_report_dir / 'model.keras')
        
        print(f"   -> Sample preparado y artefactos copiados a: {target_report_dir}")
        return sample_identifier

    except FileNotFoundError as e:
        print(f"   ERROR: Asegúrate de que el modelo 'nn_custom' fue entrenado con 'complex_data' y que el archivo {sample_data_path} existe.")
        print(f"   Detalle: {e}")
        return None
    except Exception as e:
        print(f"   ERROR al preparar el sample: {e}")
        return None
# ----------------------------------------------------------------------


if __name__ == "__main__":
    
    datasets_to_evaluate = [
        'simple_data',
        'complex_data',
    ]
    
    print("======================================================")
    print("Iniciando Evaluación de Redes Neuronales...")
    
    # 1. Evaluación Estándar
    for dataset_id in datasets_to_evaluate:
        evaluate_nn_model(
            dataset_identifier=dataset_id,
            base_output_dir=NN_MODELS_DIR
        )

    print("\n======================================================")
    print("Iniciando Evaluación con el Sample (5% de complex_data) en NN...")
    
    # 2. Preparamos el sample (lo escalamos y guardamos el test_data.pkl)
    sample_id = prepare_nn_sample_for_evaluation(DATA_SAMPLE_PATH, NN_MODELS_DIR)
    
    if sample_id:
        # 3. Ejecutamos la evaluación usando el nuevo identificador de sample.
        # Esto buscará los artefactos en models/NN/nn_custom/complex_data_sample/
        evaluate_nn_model(sample_id, NN_MODELS_DIR)

    print("======================================================")