import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Constantes y Mapeo ---
ROOT_MODELS_DIR = Path('models') 
NN_MODELS_DIR = ROOT_MODELS_DIR / 'NN' 

# Recordatorio: 0=FALSE POSITIVE, 1=CANDIDATE, 2=CONFIRMED
LABEL_MAP = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE (PC)',
    2: 'CONFIRMED (CP)'
}

def load_nn_artifacts_for_evaluation(dataset_identifier: str, base_dir: Path):
    """Carga el modelo de Red Neuronal (Keras) y los datos de prueba desde la ruta modular."""
    
    model_name = 'NN Custom' # Nombre lógico para el informe
    
    # 1. Construir la ruta de guardado (ej: models/NN/nn_custom/simple_data/)
    # La carpeta del algoritmo de NN es fija: 'nn_custom'
    data_dir = base_dir / 'nn_custom' / dataset_identifier 
    
    # Los nombres de los archivos son genéricos
    model_file = 'model.keras' 
    test_data_file = 'test_data.pkl' 

    print(f"1. Cargando artefactos de la ruta: {data_dir}")
    
    try:
        # Cargar el modelo Keras
        nn_model = load_model(data_dir / model_file)
        
        # Cargar los datos de prueba (X_test_scaled, y_test_raw)
        X_test_scaled, y_test_raw = joblib.load(data_dir / test_data_file)
        
        print(f"   Modelo '{model_name}' y datos cargados. Tamaño: {len(y_test_raw)} muestras.")
        
        return nn_model, X_test_scaled, y_test_raw
        
    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que train_NN.py se ejecutó con éxito en '{dataset_identifier}'.")
        print(f"Ruta faltante: {e}")
        return None, None, None

def generate_nn_evaluation_report(model, X_test, y_test_true, model_name: str, report_file):
    """Realiza predicciones, convierte a etiquetas discretas y genera métricas."""
    
    print(f"\n2. Evaluando: {model_name}...")
    
    # 2.1. Realizar predicciones y convertir a etiquetas discretas
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1) # Convertir OHE a clases 0, 1, 2
    
    # Escribir encabezado en el informe
    report_file.write(f"\n{'='*70}\n")
    report_file.write(f"RESULTADOS DEL MODELO: {model_name}\n")
    report_file.write(f"{'='*70}\n")
    
    # 2.2. Métricas
    accuracy = accuracy_score(y_test_true, y_pred)
    print(f"   -> Accuracy General: {accuracy:.4f}")
    report_file.write(f"Accuracy General: {accuracy:.4f}\n\n")

    cm = confusion_matrix(y_test_true, y_pred)
    print(f"   -> Matriz de Confusión:\n{cm}")
    report_file.write(f"Matriz de Confusión:\n{cm}\n\n")
    
    class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test_true, y_pred, target_names=class_names, output_dict=False)
    
    print("   -> Informe de Clasificación:\n" + report)
    report_file.write("Informe de Clasificación (Precision, Recall, F1-Score):\n" + report + "\n")


def evaluate_nn_model(dataset_identifier: str, base_output_dir: Path):
    """
    Función principal modular para cargar, evaluar y guardar el informe de la NN.
    
    :param dataset_identifier: Nombre de la carpeta del dataset (e.g., 'simple_data', 'complex_data').
    :param base_output_dir: Directorio base de modelos NN (Path('models/NN')).
    :return: Ruta al informe generado o None.
    """
    model_name = f'NN Custom ({dataset_identifier})'
    
    # Cargar artefactos
    nn_model, X_test, y_test_true = load_nn_artifacts_for_evaluation(dataset_identifier, base_output_dir)
    
    if nn_model is None:
        return None

    # Ruta de guardado: models/NN/nn_custom/{dataset}/
    report_dir = base_output_dir / 'nn_custom' / dataset_identifier
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del informe
    report_path = report_dir / f'evaluation_report_nn_{dataset_identifier}.txt'
    
    # Abrir el archivo de informe para escribir los resultados
    print(f"\n3. Guardando resultados detallados en: {report_path}")
    with open(report_path, 'w') as report_file:
        generate_nn_evaluation_report(nn_model, X_test, y_test_true, model_name, report_file)

    print("\n------------------------------------------------------")
    print(f"¡Evaluación de Red Neuronal en {dataset_identifier} completada!")
    return report_path


if __name__ == "__main__":
    
    # Lista de datasets a evaluar (asumiendo que train_NN.py ya los generó)
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