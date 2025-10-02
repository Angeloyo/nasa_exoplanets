import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# --- Constantes y Mapeo ---
ROOT_MODELS_DIR = Path('models') # Apunta al directorio 'models/'
ML_MODELS_DIR = ROOT_MODELS_DIR / 'ML' # Apunta al directorio 'models/ML/'

LABEL_MAP = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE (PC)',
    2: 'CONFIRMED (CP)'
}

def load_artifacts_for_evaluation(model_name: str, dataset_identifier: str, base_dir: Path):
    """Carga el modelo y los datos de prueba de la ruta específica de Algoritmo/Dataset."""

    # 1. Construir la ruta de guardado (ej: models/ML/randomforest/simple_data/)
    # Nota: El dataset_identifier ahora es 'simple_data' o 'complex_data'
    model_folder = model_name.lower().replace(" ", "")
    data_dir = base_dir / model_folder / dataset_identifier 
    
    # 2. Los nombres de los archivos son genéricos
    model_file = 'model.pkl' 
    test_data_file = 'test_data.pkl' 

    print(f"1. Cargando artefactos de la ruta: {data_dir}")
    
    try:
        # Cargar el modelo
        model = joblib.load(data_dir / model_file)
        
        # Cargar los datos de prueba (X_test_scaled, y_test)
        X_test, y_test = joblib.load(data_dir / test_data_file)
        
        print(f"   Modelo '{model_name}' y datos cargados. Tamaño: {len(y_test)} muestras.")
        
        return model, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que '{model_name}' fue entrenado en '{dataset_identifier}'.")
        print(f"Ruta faltante: {e}")
        return None, None, None

def generate_evaluation_report(model, X_test, y_test, model_name: str, report_file):
    """Genera las métricas de rendimiento y las escribe en el archivo."""
    
    print(f"\n2. Evaluando: {model_name}...")
    
    # Predecir etiquetas
    y_pred = model.predict(X_test)
    
    # Escribir encabezado
    report_file.write(f"\n{'='*70}\n")
    report_file.write(f"RESULTADOS DEL MODELO: {model_name}\n")
    report_file.write(f"{'='*70}\n")
    
    # 2.1. Accuracy General
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy General: {accuracy:.4f}")
    report_file.write(f"Accuracy General: {accuracy:.4f}\n\n")

    # 2.2. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"   -> Matriz de Confusión:\n{cm}")
    report_file.write(f"Matriz de Confusión:\n{cm}\n\n")
    
    # 2.3. Classification Report
    class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=False)
    
    print("   -> Informe de Clasificación:\n" + report)
    report_file.write("Informe de Clasificación (Precision, Recall, F1-Score):\n" + report + "\n")


def evaluate_ml_model(model_name: str, dataset_identifier: str, base_output_dir: Path):
    """
    Función principal modular para cargar, evaluar y guardar el informe de un solo modelo ML.
    
    :param model_name: Nombre del modelo (e.g., 'RandomForest', 'XGBoost').
    :param dataset_identifier: Nombre de la carpeta del dataset (e.g., 'simple_data', 'complex_data').
    :param base_output_dir: Directorio base de modelos ML (Path('models/ML')).
    :return: Ruta al informe generado o None.
    """
    
    # Cargar artefactos
    model, X_test, y_test = load_artifacts_for_evaluation(model_name, dataset_identifier, base_output_dir)
    
    if model is None:
        return None

    # Ruta de guardado: models/ML/{algoritmo}/{dataset}/
    report_dir = base_output_dir / model_name.lower().replace(" ", "") / dataset_identifier
    report_dir.mkdir(parents=True, exist_ok=True) # Asegura que el directorio existe
    
    # El nombre del informe sigue siendo específico para documentar qué se evaluó
    report_path = report_dir / f'evaluation_report_{model_name.lower()}_{dataset_identifier}.txt'
    
    # Abrir el archivo de informe para escribir los resultados
    print(f"\n3. Guardando resultados detallados en: {report_path}")
    with open(report_path, 'w') as report_file:
        generate_evaluation_report(model, X_test, y_test, model_name, report_file)

    print("\n------------------------------------------------------")
    print(f"¡Evaluación de {model_name} en {dataset_identifier} completada!")
    return report_path


if __name__ == "__main__":
    
    # Lista de combinaciones de modelos y datasets a evaluar
    evaluation_targets = [
        # Algoritmo, Dataset_Folder
        ('RandomForest', 'simple_data'),
        ('RandomForest', 'complex_data'),
        ('XGBoost', 'simple_data'),
        ('XGBoost', 'complex_data'),
    ]

    print("======================================================")
    print("Iniciando Evaluación de las 4 Combinaciones de Modelos ML...")
    
    for model_name, dataset_id in evaluation_targets:
        evaluate_ml_model(
            model_name=model_name,
            dataset_identifier=dataset_id,
            base_output_dir=ML_MODELS_DIR
        )
    
    print("======================================================")