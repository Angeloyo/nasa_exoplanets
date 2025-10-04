import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Constantes y Configuración ---
PROCESSED_DIR = Path('data/processed')
ROOT_MODELS_DIR = Path('backend/models') # Directorio raíz de modelos
RANDOM_STATE = 42
TEST_SIZE = 0.2

def get_dataset_folder_name(dataset_identifier: str):
    """Normaliza el identificador del dataset a un nombre de carpeta."""
    if dataset_identifier in ['simple_data', 'complex_data']:
        return dataset_identifier
    # Para cualquier otro nombre de archivo (ruta personalizada)
    return 'user_data' 

def load_data(data_path: Path):
    """Carga un dataset a partir de una ruta de archivo completa, lo divide y devuelve el identificador."""
    
    print(f"1. Cargando datos desde {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado. Asegúrate de que {data_path} existe.")
        return None
    
    # Separar características (X) y etiqueta (y)
    X = df.drop(columns=['DISPOSITION_ENCODED'])
    y = df['DISPOSITION_ENCODED']
    
    # Guardar los nombres de las features
    feature_names = X.columns.tolist()
    
    # Dividir el dataset
    print(f"   Filas totales: {len(X)}. División: {100*(1-TEST_SIZE)}% train / {100*TEST_SIZE}% test.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Usar el nombre del archivo (sin extensión) como identificador del dataset
    dataset_identifier = data_path.stem
    
    return X_train, X_test, y_train, y_test, feature_names, dataset_identifier

def get_ml_model_instance(model_name: str, params: dict):
    """Devuelve una instancia del modelo de ML solicitado con hiperparámetros personalizados."""
    
    if model_name == 'RandomForest':
        default_params = {
            'n_estimators': 200, 'max_depth': 10, 'random_state': RANDOM_STATE, 'class_weight': 'balanced' 
        }
        final_params = {**default_params, **params} 
        print(f"   Hiperparámetros usados: {final_params}")
        return RandomForestClassifier(**final_params)
        
    elif model_name == 'XGBoost':
        default_params = {
            'n_estimators': 100, 'learning_rate': 0.1, 'random_state': RANDOM_STATE, 
            'use_label_encoder': False, 'eval_metric': 'mlogloss'
        }
        final_params = {**default_params, **params}
        print(f"   Hiperparámetros usados: {final_params}")
        return XGBClassifier(**final_params)
        
    else:
        raise ValueError(f"Modelo ML desconocido: {model_name}. Debe ser 'RandomForest' o 'XGBoost'.")

def train_ml_model(
    model_name: str, 
    data_path: Path, 
    output_dir: Path, # Este será 'models/ML'
    model_params: dict = None
):
    """
    Función principal modular para cargar datos, entrenar y guardar un modelo ML.
    """
    
    print(f"\n--- Iniciando Entrenamiento de {model_name} ---")
    
    # 1. Cargar y dividir datos
    data_tuple = load_data(data_path)
    if data_tuple is None:
        return False
        
    X_train, X_test, y_train, y_test, feature_names, dataset_identifier = data_tuple

    # 2. Escalar características
    print("2. Escalando características (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Obtener y entrenar el modelo
    params_to_use = model_params if model_params is not None else {}
    try:
        model = get_ml_model_instance(model_name, params_to_use)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False
        
    print(f"3. Entrenando {model_name} con {dataset_identifier}...")
    model.fit(X_train_scaled, y_train)
    
    # 4. Guardar Artefactos con la nueva estructura
    
    # 4a. Definir la estructura de guardado: models/ML/{Model_Name}/{Dataset_Folder}/
    model_folder = model_name.lower().replace(" ", "") 
    dataset_folder = get_dataset_folder_name(dataset_identifier)
    
    save_dir = output_dir / model_folder / dataset_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 4b. Guardar los artefactos con NOMBRES GENÉRICOS
    
    # Guardar el modelo
    model_path = save_dir / 'model.pkl'
    joblib.dump(model, model_path)
    print(f"   -> Modelo guardado: {model_path}")

    # Guardar el escalador
    scaler_path = save_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   -> StandardScaler guardado: {scaler_path}")

    # Guardar los nombres de las features
    features_path = save_dir / 'features.pkl'
    joblib.dump(feature_names, features_path)
    print(f"   -> Features guardadas: {features_path}")
    
    # Guardar los datos de prueba (escalados)
    test_data_path = save_dir / 'test_data.pkl'
    joblib.dump((X_test_scaled, y_test), test_data_path)
    print(f"   -> Datos de prueba guardados: {test_data_path}")
    
    print("--- ¡Entrenamiento ML Finalizado! ---")
    return True


if __name__ == "__main__":
    
    # Definir la ruta base para los modelos ML
    ML_MODELS_DIR = ROOT_MODELS_DIR / 'ML'
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True) 
    
    # Rutas a los datasets
    simple_data_path = PROCESSED_DIR / 'simple_data.csv'
    complex_data_path = PROCESSED_DIR / 'complex_data.csv'
    
    # --- Hiperparámetros personalizados (puedes ajustarlos aquí) ---
    xgb_custom_params = {
        'n_estimators': 300, 
        'learning_rate': 0.05, 
    }
    rf_custom_params = {
        'n_estimators': 300, # Sobrescribir el default de 200
        'max_depth': 12,     # Sobrescribir el default de 10
    }

    # ======================================================
    # 4. XGBoost en COMPLEX (Usando parámetros personalizados XGB)
    # ======================================================
    print("\n======================================================")
    train_ml_model(
        model_name='XGBoost',
        data_path=complex_data_path,
        output_dir=ML_MODELS_DIR,
        model_params=xgb_custom_params
    )
    # Ruta: models/ML/xgboost/complex_data/model.pkl
    
    print("\n======================================================")
    print("¡Entrenamiento de los 4 modelos (RF/XGB en Simple/Complex) completado!")