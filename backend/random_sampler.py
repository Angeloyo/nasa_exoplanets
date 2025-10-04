import pandas as pd
from pathlib import Path

# --- Configuración ---
DATA_FILE_NAME = 'complex_data.csv'
LABEL_COLUMN = 'DISPOSITION_ENCODED'

# Fracción TOTAL a separar para las dos muestras de prueba (5% + 5% = 10%)
TOTAL_SAMPLE_FRACTION = 0.10  
RANDOM_STATE = 42             # Semilla para asegurar la reproducibilidad

# Nombres de los archivos de salida
SAMPLE_A_FILENAME = 'complex_data_sample_A.csv'
SAMPLE_B_FILENAME = 'complex_data_sample_B.csv'

# --- Rutas ---
PROCESSED_DIR = Path('data/processed')
OUTPUT_DIR = Path('data/sample') 
FILE_PATH = PROCESSED_DIR / DATA_FILE_NAME
CLEAN_FILE_PATH = PROCESSED_DIR / DATA_FILE_NAME # Ruta para sobrescribir el archivo de entrenamiento

def process_and_split_data(file_path: Path, fraction: float, seed: int):
    """
    Carga el CSV, lo divide en 90% (Entrenamiento) y 10% (Muestras).
    Guarda el 90% completo.
    Guarda las dos muestras de 5% sin la columna de etiquetas.
    """
    print(f"--- Iniciando división de datos para {file_path.name} ---")
    try:
        df_full = pd.read_csv(file_path)
        N_total = len(df_full)
        print(f"✅ Datos cargados: {N_total} filas.")
        
        # 1. Separar el 10% total para las muestras (df_samples)
        df_samples = df_full.sample(frac=fraction, random_state=seed)
        df_train_new = df_full.drop(df_samples.index) # El 90% restante para entrenamiento
        
        N_samples = len(df_samples)
        N_train = len(df_train_new)

        print(f"Separado: {N_samples} filas para samples ({fraction*100:.0f}%) y {N_train} filas para entrenamiento.")

        # 2. Dividir el 10% (df_samples) en dos muestras iguales de 5%
        df_sample_A = df_samples.sample(frac=0.5, random_state=seed)
        df_sample_B = df_samples.drop(df_sample_A.index)
        
        print(f"Muestra A generada: {len(df_sample_A)} filas (aprox. 5%).")
        print(f"Muestra B generada: {len(df_sample_B)} filas (aprox. 5%).")
        
        # 3. ELIMINAR LA ETIQUETA de las muestras de prueba
        df_sample_A_clean = df_sample_A.drop(columns=[LABEL_COLUMN], errors='ignore')
        df_sample_B_clean = df_sample_B.drop(columns=[LABEL_COLUMN], errors='ignore')

        print(f"Etiqueta '{LABEL_COLUMN}' eliminada de ambas muestras.")
        
        # 4. Guardar los tres DataFrames
        save_data(df_train_new, PROCESSED_DIR, DATA_FILE_NAME)          # 90% (Con etiqueta)
        save_data(df_sample_A_clean, OUTPUT_DIR, SAMPLE_A_FILENAME)     # 5% (Sin etiqueta)
        save_data(df_sample_B_clean, OUTPUT_DIR, SAMPLE_B_FILENAME)     # 5% (Sin etiqueta)
        
        print("\n¡Proceso de muestreo y limpieza completado!")
        print(f"El nuevo archivo {DATA_FILE_NAME} para entrenamiento tiene {N_train} filas.")

    except FileNotFoundError:
        print(f"❌ ERROR: Archivo fuente no encontrado en {file_path}. Verifica la ruta.")
    except Exception as e:
        print(f"❌ ERROR inesperado durante el procesamiento: {e}")


def save_data(df: pd.DataFrame, output_dir: Path, filename: str):
    """
    Guarda el DataFrame en la ruta y carpeta especificadas.
    """
    # Crear la carpeta de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Definir la ruta completa de guardado
    save_path = output_dir / filename
    
    # Guardar el archivo
    df.to_csv(save_path, index=False)
    
    print(f"💾 Guardado: {save_path.resolve()}")

if __name__ == "__main__":
    process_and_split_data(FILE_PATH, TOTAL_SAMPLE_FRACTION, RANDOM_STATE)