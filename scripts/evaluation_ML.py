import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# Rutas
MODELS_DIR = Path('models/ML')
REPORT_PATH = MODELS_DIR / 'evaluation_report.txt'

LABEL_MAP = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE (PC)',
    2: 'CONFIRMED (CP)'
}

def load_models_and_data():
    """Carga los modelos entrenados y los datos de prueba."""
    
    print("1. Cargando modelos y datos de prueba...")
    
    try:
        # Cargar los datos de prueba (X_test_scaled, y_test)
        X_test, y_test = joblib.load(MODELS_DIR / 'test_data.pkl')
        
        # Cargar los modelos
        rf_model = joblib.load(MODELS_DIR / 'randomforest_model.pkl')
        xgb_model = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
        
        print(f"   Datos de prueba cargados. Tamaño: {len(y_test)} muestras.")
        
        return {
            'RandomForest': rf_model,
            'XGBoost': xgb_model
        }, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que train_ML.py se ejecutó con éxito.")
        print(f"Ruta faltante: {e}")
        return None, None, None

def evaluate_model(model, X_test, y_test, model_name: str, report_file):
    """Realiza predicciones y genera métricas de rendimiento."""
    
    print(f"\n2. Evaluando: {model_name}...")
    
    # Predecir etiquetas
    y_pred = model.predict(X_test)
    
    # Escribir encabezado en el informe
    report_file.write(f"\n{'='*50}\n")
    report_file.write(f"RESULTADOS DEL MODELO: {model_name}\n")
    report_file.write(f"{'='*50}\n")
    
    # 2.1. Accuracy General
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Accuracy General: {accuracy:.4f}")
    report_file.write(f"Accuracy General: {accuracy:.4f}\n\n")

    # 2.2. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"   -> Matriz de Confusión:\n{cm}")
    report_file.write(f"Matriz de Confusión:\n{cm}\n\n")
    
    # 2.3. Classification Report (Precision, Recall, F1-Score por clase)
    class_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=False)
    
    print("   -> Informe de Clasificación:\n" + report)
    report_file.write("Informe de Clasificación (Precision, Recall, F1-Score):\n" + report + "\n")


def main():
    """Función principal para cargar, evaluar y guardar los resultados."""
    
    # Cargar datos y modelos
    models, X_test, y_test = load_models_and_data()
    
    if models is None:
        return

    # Abrir el archivo de informe para escribir los resultados
    print(f"\n3. Guardando resultados detallados en: {REPORT_PATH}")
    with open(REPORT_PATH, 'w') as report_file:
        
        # Iterar sobre cada modelo y evaluarlo
        for name, model in models.items():
            evaluate_model(model, X_test, y_test, name, report_file)

    print("\n------------------------------------------------------")
    print("¡Evaluación de modelos de ML completada!")
    print(f"Revisa {REPORT_PATH} para el análisis detallado.")
    print("------------------------------------------------------")

if __name__ == "__main__":
    main()