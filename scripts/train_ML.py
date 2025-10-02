import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Constantes de Rutas ---
PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('models/ML')
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Asegurar que el directorio de modelos existe

# --- Configuración de Entrenamiento ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_TO_USE = 'simple' # O 'complex' si prefieres empezar con más features

def load_and_scale_data(dataset_name: str):
    """Carga el dataset, lo divide en entrenamiento/prueba y lo escala."""
    
    file_path = PROCESSED_DIR / f'{dataset_name}_data.csv'
    print(f"1. Cargando datos desde {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que {file_path} existe.")
        return None, None, None, None, None
    
    # Separar características (X) y etiqueta (y)
    X = df.drop(columns=['DISPOSITION_ENCODED'])
    y = df['DISPOSITION_ENCODED']
    
    # Guardar los nombres de las features para el escalador
    feature_names = X.columns.tolist()
    
    # Dividir el dataset
    print(f"   Filas totales: {len(X)}. División: {100*(1-TEST_SIZE)}% train / {100*TEST_SIZE}% test.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Escalar las características (Crucial para muchos modelos)
    print("2. Escalando características (StandardScaler)...")
    scaler = StandardScaler()
    
    # Ajustar (fit) y transformar solo con los datos de entrenamiento para evitar el data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Devolver los datos escalados y el escalador ajustado
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_and_save_model(model, name: str, X_train, y_train):
    """Entrena un modelo y lo guarda en disco."""
    
    print(f"3. Entrenando {name}...")
    model.fit(X_train, y_train)
    
    # Guardar el modelo entrenado
    model_path = MODELS_DIR / f'{name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, model_path)
    print(f"   Modelo guardado exitosamente en: {model_path}")
    
    return model

def save_scaler_and_features(scaler, feature_names: list):
    """Guarda el escalador y la lista de features para la API."""
    
    scaler_path = MODELS_DIR / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   StandardScaler guardado en: {scaler_path}")
    
    # Guardar los nombres de las features para asegurar la consistencia en la predicción
    features_path = MODELS_DIR / 'features.pkl'
    joblib.dump(feature_names, features_path)
    print(f"   Nombres de features guardados en: {features_path}")


def main():
    """Función principal del script de entrenamiento."""
    
    # 1-2. Cargar y Escalar
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_scale_data(DATASET_TO_USE)
    
    if X_train is None:
        return
        
    # Guardar el escalador y los nombres de las features para el deployment
    save_scaler_and_features(scaler, feature_names)

    # --- Definición y Entrenamiento de Modelos ---

    # Random Forest (Clasificador de ensamblaje basado en árboles de decisión)
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        random_state=RANDOM_STATE, 
        class_weight='balanced' # Importante para datasets desbalanceados
    )
    train_and_save_model(rf_model, "RandomForest", X_train, y_train)

    # XGBoost (Gradient Boosting de alto rendimiento)
    xgb_model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=RANDOM_STATE, 
        use_label_encoder=False, 
        eval_metric='mlogloss' # Métrica para clasificación multi-clase
    )
    train_and_save_model(xgb_model, "XGBoost", X_train, y_train)
    
    # Pasar los datos de prueba para el siguiente script de evaluación
    joblib.dump((X_test, y_test), MODELS_DIR / 'test_data.pkl')
    print("\nDatos de prueba guardados para evaluación posterior.")
    print("------------------------------------------------------")
    print("¡Entrenamiento de modelos de ML completado!")

if __name__ == "__main__":
    main()