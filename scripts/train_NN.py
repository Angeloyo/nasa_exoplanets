import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# --- Constantes de Rutas y Configuración General ---
PROCESSED_DIR = Path('data/processed')
ROOT_MODELS_DIR = Path('models')
ML_MODELS_DIR = ROOT_MODELS_DIR / 'ML' # Usado para guardar el escalador/features por consistencia
NN_MODELS_DIR = ROOT_MODELS_DIR / 'NN' # Directorio base para modelos NN
NN_MODELS_DIR.mkdir(parents=True, exist_ok=True) 

# Número de clases (0:FP, 1:CANDIDATE, 2:CONFIRMED)
NUM_CLASSES = 3 
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Configuración por defecto de la Arquitectura y Entrenamiento ---
# Parámetros que puedes sobrescribir al llamar a train_nn_model
DEFAULT_TRAIN_PARAMS = {
    'epochs': 50,
    'batch_size': 32,
    'patience': 10,  # Para EarlyStopping
}

DEFAULT_ARCH_PARAMS = {
    'layer_1_units': 64,
    'layer_2_units': 32,
    'dropout_rate': 0.2,
}

def get_dataset_folder_name(dataset_identifier: str):
    """Normaliza el identificador del dataset a un nombre de carpeta."""
    if dataset_identifier in ['simple_data', 'complex_data']:
        return dataset_identifier
    return 'user_data' 

def load_and_scale_data(data_path: Path):
    """Carga, escala y codifica las etiquetas para Keras."""
    
    print(f"1. Cargando y preparando datos para NN desde {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en {data_path}.")
        return None
    
    # Separar características (X) y etiqueta (y)
    X = df.drop(columns=['DISPOSITION_ENCODED'])
    y = df['DISPOSITION_ENCODED']
    feature_names = X.columns.tolist()
    
    # Dividir el dataset
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 2. Escalar características
    print("2. Escaland")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. One-Hot Encoding
    print("3. Codificando etiquetas a One-Hot Encoding...")
    y_train_encoded = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_test_encoded = to_categorical(y_test_raw, num_classes=NUM_CLASSES)
    
    # Identificador del dataset para la carpeta
    dataset_identifier = data_path.stem
    
    return (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, 
            scaler, feature_names, y_test_raw, dataset_identifier)

def build_nn_model(input_shape, arch_params: dict):
    """Define la arquitectura de la Red Neuronal (MLP) con parámetros variables."""
    
    print("4. Construyendo la arquitectura de la Red Neuronal...")
    
    model = Sequential([
        Dense(arch_params['layer_1_units'], activation='relu', input_shape=(input_shape,)),
        Dropout(arch_params['dropout_rate']), 
        Dense(arch_params['layer_2_units'], activation='relu'),
        # Capa de salida
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary(print_fn=lambda x: print("   " + x)) # Imprimir con indentación
    return model

def train_nn_model(
    data_path: Path, 
    arch_params: dict = None, 
    train_params: dict = None
):
    """
    Función principal modular para entrenar y guardar una Red Neuronal.
    """
    
    # --- 0. Preparar Parámetros ---
    arch_params_final = {**DEFAULT_ARCH_PARAMS, **(arch_params or {})}
    train_params_final = {**DEFAULT_TRAIN_PARAMS, **(train_params or {})}
    
    print("\n--- Iniciando Entrenamiento de Red Neuronal ---")

    # 1-3. Cargar y Preparar Datos
    results = load_and_scale_data(data_path)
    if results is None:
        return False
    
    (X_train, X_test, y_train_encoded, y_test_encoded, 
     scaler, feature_names, y_test_raw, dataset_identifier) = results
    
    # 4. Construir Modelo
    input_shape = X_train.shape[1]
    nn_model = build_nn_model(input_shape, arch_params_final)
    
    # --- 5. Definir Rutas de Guardado ---
    
    # Estructura: models/NN/nn_custom/{simple_data}/
    dataset_folder = get_dataset_folder_name(dataset_identifier)
    save_dir = NN_MODELS_DIR / 'nn_custom' / dataset_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Definir Callbacks
    checkpoint_path = save_dir / 'model_best.keras'
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=train_params_final['patience'], 
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=0 # Silenciar el checkpoint para un output más limpio
    )
    
    # 6. Entrenar Modelo
    print(f"\n6. Entrenando Red Neuronal en {dataset_identifier}...")
    nn_model.fit(
        X_train, 
        y_train_encoded, 
        epochs=train_params_final['epochs'], 
        batch_size=train_params_final['batch_size'], 
        validation_data=(X_test, y_test_encoded),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # 7. Guardar Artefactos (con nombres genéricos)
    
    # Guardar el modelo final (además del mejor guardado por el checkpoint)
    nn_model.save(save_dir / 'model.keras')
    print(f"   -> Modelo final de NN guardado en: {save_dir / 'model.keras'}")
    
    # Guardar el escalador (necesario para la predicción)
    joblib.dump(scaler, save_dir / 'scaler.pkl')
    
    # Guardar los datos de prueba (X_test escalado y y_test_raw)
    joblib.dump((X_test, y_test_raw), save_dir / 'test_data.pkl')
    
    print(f"--- ¡Entrenamiento NN Finalizado en {dataset_folder}! ---")
    return True


if __name__ == "__main__":
    
    # Crear la carpeta de usuario NN_CUSTOM
    (NN_MODELS_DIR / 'nn_custom').mkdir(parents=True, exist_ok=True)
    
    # Rutas a los datasets
    simple_data_path = PROCESSED_DIR / 'simple_data.csv'
    complex_data_path = PROCESSED_DIR / 'complex_data.csv'

    # ======================================================
    # 1. NN en SIMPLE (Usando parámetros por defecto)
    # ======================================================
    train_nn_model(
        data_path=simple_data_path,
    )

    # ======================================================
    # 2. NN en COMPLEX (Usando parámetros de arquitectura y entrenamiento personalizados)
    # ======================================================
    custom_arch = {
        'layer_1_units': 128, # Más neuronas en la primera capa
        'layer_2_units': 64, 
        'dropout_rate': 0.3, # Más dropout
    }
    custom_train = {
        'epochs': 100,      # Más épocas
        'batch_size': 64,   # Batch size más grande
    }
    
    train_nn_model(
        data_path=complex_data_path,
        arch_params=custom_arch,
        train_params=custom_train
    )
