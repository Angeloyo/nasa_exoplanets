import pandas as pd
from pathlib import Path

# --- Configuración ---
DATA_FILE = 'complex_data.csv'
SAMPLE_FRACTION = 0.05      # 5% de las filas
RANDOM_STATE = 42          # Semilla para asegurar la reproducibilidad
OUTPUT_FILENAME = 'complex_data_sample.csv'

# --- Rutas ---
PROCESSED_DIR = Path('data/processed')
OUTPUT_DIR = Path('data/sample') # ¡Nueva carpeta de destino!
FILE_PATH = PROCESSED_DIR / DATA_FILE

def get_random_sample(file_path: Path, fraction: float, seed: int):
    """
    Carga un CSV y devuelve una muestra aleatoria basada en la fracción especificada.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Datos cargados: {len(df)} filas.")
        
        # Usamos sample() para obtener una fracción aleatoria
        df_sample = df.sample(frac=fraction, random_state=seed)
        
        print(f"✨ Muestra aleatoria generada: {len(df_sample)} filas ({fraction * 100:.0f}%).")
        return df_sample
        
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado en {file_path}. Verifica la ruta.")
        return None
    except Exception as e:
        print(f"❌ ERROR inesperado: {e}")
        return None

def save_sample(df_sample: pd.DataFrame, output_dir: Path, filename: str):
    """
    Guarda el DataFrame de la muestra en la ruta y carpeta especificadas.
    """
    # 1. Crear la carpeta de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Definir la ruta completa de guardado
    save_path = output_dir / filename
    
    # 3. Guardar el archivo
    df_sample.to_csv(save_path, index=False)
    
    print(f"💾 Muestra guardada con éxito en: {save_path.resolve()}")

if __name__ == "__main__":
    sample_df = get_random_sample(FILE_PATH, SAMPLE_FRACTION, RANDOM_STATE)
    
    if sample_df is not None:
        save_sample(sample_df, OUTPUT_DIR, OUTPUT_FILENAME)