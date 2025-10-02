import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# --- Constantes de Rutas ---
MODELS_DIR = Path('models/NN')
REPORT_PATH = MODELS_DIR / 'evaluation_report_nn.txt'
MODEL_NAME = 'nn_model_best.keras' # Usamos el mejor modelo guardado por ModelCheckpoint

# --- Mapeo Inverso de Etiquetas (para el informe) ---
# Recordatorio: 0=FALSE POSITIVE, 1=CANDIDATE, 2=CONFIRMED
LABEL_MAP = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE (PC)',
    2: 'CONFIRMED (CP)'
}

def load_nn_model_and_data():
    """Carga el modelo de Red Neuronal (Keras) y los datos de prueba."""
    
    print("1. Cargando modelo de NN y datos de prueba...")
    
    try:
        # Cargar el modelo Keras
        nn_model = load_model(MODELS_DIR / MODEL_NAME)
        
        # Cargar los datos de prueba (X_test_scaled, y_test_raw)
        X_test_scaled, y_test_raw = joblib.load(MODELS_DIR / 'test_data_nn.pkl')
        
        print(f"   Modelo {MODEL_NAME} y datos cargados. Tamaño: {len(y_test_raw)} muestras.")
        
        return nn_model, X_test_scaled, y_test_raw
        
    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que train_NN.py se ejecutó con éxito.")
        print(f"Ruta faltante: {e}")
        return None, None, None

def evaluate_nn_model(model, X_test, y_test_true, report_file):
    """Realiza predicciones, convierte a etiquetas discretas y genera métricas."""
    
    print("\n2. Evaluando: Red Neuronal...")
    
    # 2.1. Realizar predicciones
    # El modelo Keras devuelve probabilidades (One-Hot Encoded)
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Convertir probabilidades a etiquetas discretas (el índice con la probabilidad más alta)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Escribir encabezado en el informe
    report_file.write(f"\n{'='*50}\n")
    report_file.write("RESULTADOS DEL MODELO: Red Neuronal (Keras)\n")
    report_file.write(f"{'='*50}\n")
    
    # 2.2. Accuracy General
    accuracy = accuracy_score(y_test_true, y_pred)
    print(f"   -> Accuracy General: {accuracy:.4f}")
    report_file.write(f"Accuracy General: {accuracy:.4f}\n\n")

    # 2.3. Matriz de Confusión
    cm = confusion_matrix(y_test_true, y_pred)
    print(f"   -> Matriz de Confusión:\n{cm}")
    report_file.write(f"Matriz de Confusión:\n{cm}\n\n")
    
    # 2.4. Classification Report
    class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test_true, y_pred, target_names=class_names, output_dict=False)
    
    print("   -> Informe de Clasificación:\n" + report)
    report_file.write("Informe de Clasificación (Precision, Recall, F1-Score):\n" + report + "\n")


def main():
    """Función principal para cargar, evaluar y guardar los resultados de la NN."""
    
    # Cargar datos y modelo
    nn_model, X_test, y_test_true = load_nn_model_and_data()
    
    if nn_model is None:
        return

    # Abrir el archivo de informe para escribir los resultados
    print(f"\n3. Guardando resultados detallados en: {REPORT_PATH}")
    with open(REPORT_PATH, 'w') as report_file:
        evaluate_nn_model(nn_model, X_test, y_test_true, report_file)

    print("\n------------------------------------------------------")
    print("¡Evaluación de Red Neuronal completada!")
    print(f"Revisa {REPORT_PATH} para el análisis detallado.")
    print("------------------------------------------------------")

if __name__ == "__main__":
    main()