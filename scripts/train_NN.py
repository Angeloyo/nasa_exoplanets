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

# --- Constantes de Rutas ---
PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('models/NN')
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Crear directorio para modelos NN

# --- Configuración de Entrenamiento ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATASET_TO_USE = 'simple' # Usaremos el conjunto 'simple' por consistencia inicial
EPOCHS = 50
BATCH_SIZE = 32
# Número de clases (0:FP, 1:CANDIDATE, 2:CONFIRMED)
NUM_CLASSES = 3 

def load_and_scale_data(dataset_name: str):
    """Carga el dataset, lo divide en entrenamiento/prueba, lo escala y codifica las etiquetas."""
    
    file_path = PROCESSED_DIR / f'{dataset_name}_data.csv'
    print(f"1. Cargando y preparando datos para NN desde {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en {file_path}. Asegúrate de que preprocess.py se ejecutó.")
        return None, None, None, None, None, None
    
    # Separar características (X) y etiqueta (y)
    X = df.drop(columns=['DISPOSITION_ENCODED'])
    y = df['DISPOSITION_ENCODED']
    
    feature_names = X.columns.tolist()
    
    # Dividir el dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 2. Escalar características (StandardScaler)
    # Reutilizamos la lógica del escalador, guardándolo también para la API
    print("2. Escalando características (StandardScaler)...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. One-Hot Encoding de la variable objetivo (CRUCIAL para Keras)
    # Keras espera las etiquetas en formato one-hot para clasificación multi-clase
    print("3. Codificando etiquetas a One-Hot Encoding...")
    y_train_encoded = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_encoded = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    # Guardar el escalador y los features en el directorio de ML para consistencia
    # (Ya que solo necesitamos un escalador ajustado)
    ml_models_dir = Path('models/ML')
    joblib.dump(scaler, ml_models_dir / 'scaler.pkl')
    joblib.dump(feature_names, ml_models_dir / 'features.pkl')
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, feature_names, X_test, y_test

def build_nn_model(input_shape):
    """Define la arquitectura de la Red Neuronal (MLP simple)."""
    
    print("4. Construyendo la arquitectura de la Red Neuronal...")
    
    model = Sequential([
        # Capa de entrada: input_shape es el número de características (4 en este caso)
        Dense(64, activation='relu', input_shape=(input_shape,)),
        # Dropout para prevenir el sobreajuste
        Dropout(0.2), 
        Dense(32, activation='relu'),
        # Capa de salida: 3 neuronas (una por clase) y activación softmax para probabilidades
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    
    # Compilación: 'categorical_crossentropy' es la función de pérdida estándar para OHE
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def main():
    """Función principal del script de entrenamiento de NN."""
    
    # 1-3. Cargar y Preparar Datos
    results = load_and_scale_data(DATASET_TO_USE)
    if results is None:
        return
    
    X_train, X_test, y_train_encoded, y_test_encoded, scaler, feature_names, X_test_raw, y_test_raw = results
    
    # 4. Construir Modelo
    input_shape = X_train.shape[1]
    nn_model = build_nn_model(input_shape)
    
    # Definir Callbacks
    # EarlyStopping: Detener el entrenamiento si la accuracy no mejora para evitar el sobreajuste
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    # ModelCheckpoint: Guardar solo la mejor versión del modelo (basada en la validation loss)
    checkpoint_path = MODELS_DIR / 'nn_model_best.keras'
    checkpoint = ModelCheckpoint(
        checkpoint_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    # 5. Entrenar Modelo
    print("\n5. Entrenando Red Neuronal...")
    history = nn_model.fit(
        X_train, 
        y_train_encoded, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test_encoded), # Usamos X_test/y_test para validación
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # 6. Guardar Modelo (por si acaso el checkpoint no lo hizo, aunque el checkpoint es mejor)
    final_model_path = MODELS_DIR / 'nn_model_final.keras'
    nn_model.save(final_model_path)
    print(f"   Modelo final de NN guardado en: {final_model_path}")
    
    # 7. Guardar los datos de prueba sin codificar para el script de evaluación
    joblib.dump((X_test, y_test_raw), MODELS_DIR / 'test_data_nn.pkl')
    print("   Datos de prueba de NN guardados para evaluación posterior.")
    print("------------------------------------------------------")
    print("¡Entrenamiento de Red Neuronal completado!")

if __name__ == "__main__":
    main()