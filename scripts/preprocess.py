import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib

# --- Rutas ---
RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) 
ML_MODELS_DIR = Path('models/ML') # Necesario para guardar el Imputer
ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Normalización de Nombres de Columnas ---
COLUMN_MAPPING = {
    # Características Comunes
    'DISPOSITION': ['disposition', 'koi_disposition', 'tfopwg_disp'],
    'PERIOD': ['pl_orbper', 'koi_period'],
    'RADIUS': ['pl_rade', 'koi_prad', 'pl_rade'],
    'DENSITY': ['pl_dens', 'koi_srho'],
    'NUM_PLANETS': ['sy_pnum', 'koi_count', 'pl_pnum'], 

    # Características Adicionales
    'DURATION': ['pl_trandurh', 'koi_duration'],
    'TEFF': ['st_teff', 'koi_steff'],
    'DEPTH': ['pl_trandep', 'koi_depth'],
}

# --- Elección de las etiquetas ---
DISP_MAP = {
    'CONFIRMED': 2,
    'CANDIDATE': 1,
    'PC': 1, 
    'FALSE POSITIVE': 0,
    'FP': 0,
}

# Definición de conjuntos de columnas para facilitar el control desde main.py
SIMPLE_COLS = ['PERIOD', 'RADIUS', 'DENSITY', 'NUM_PLANETS']
COMPLEX_COLS = SIMPLE_COLS + ['DURATION', 'TEFF', 'DEPTH']


def load_and_standardize_data(file_path: Path, source_name: str, all_cols: list) -> pd.DataFrame:
    """Carga un CSV, saltándose los comentarios y seleccionando/normalizando las columnas clave."""
    
    try:
        df = pd.read_csv(file_path, comment='#')
    except FileNotFoundError:
        return pd.DataFrame() 
    
    df_standard = pd.DataFrame()
    
    for target_col in all_cols:
        source_cols = COLUMN_MAPPING.get(target_col, [])
        found_col = next((col for col in source_cols if col in df.columns), None)
        
        if target_col == 'DISPOSITION':
            if source_name == 'k2' and 'disposition' in df.columns:
                df_standard[target_col] = df['disposition']
            elif source_name == 'kepler' and 'koi_disposition' in df.columns:
                df_standard[target_col] = df['koi_disposition']
            elif source_name == 'tess' and 'tfopwg_disp' in df.columns:
                df_standard[target_col] = df['tfopwg_disp'].astype(str).str.strip()
            
        elif target_col == 'DENSITY':
            if found_col:
                df_standard[target_col] = df[found_col]
            else:
                df_standard[target_col] = np.nan
        
        elif found_col:
            df_standard[target_col] = df[found_col]

    return df_standard


def run_preprocessing(data_source: str = 'raw_files', custom_df: pd.DataFrame = None):
    """
    Ejecuta el pipeline de preprocesamiento completo.
    
    :param data_source: 'raw_files' para cargar K2/Kepler/TESS, o 'custom_df' si se pasa un DataFrame.
    :param custom_df: DataFrame opcional para procesar directamente.
    :return: Rutas de los archivos guardados (simple_path, complex_path) o None si falla.
    """
    print(f"--- Iniciando Preprocesamiento ({data_source}) ---")

    if data_source == 'raw_files':
        # 1. Cargar y Estandarizar
        ALL_REQUIRED_COLS = list(set(SIMPLE_COLS + COMPLEX_COLS + ['DISPOSITION']))
        dfs = []
        dfs.append(load_and_standardize_data(RAW_DIR / 'k2.csv', 'k2', ALL_REQUIRED_COLS))
        dfs.append(load_and_standardize_data(RAW_DIR / 'kepler.csv', 'kepler', ALL_REQUIRED_COLS))
        dfs.append(load_and_standardize_data(RAW_DIR / 'tess.csv', 'tess', ALL_REQUIRED_COLS))
        df_combined = pd.concat(dfs, ignore_index=True)
    
    elif data_source == 'custom_df' and custom_df is not None:
        # 1. Usar el DataFrame pasado (asume que ya está razonablemente limpio o se usará para inferencia)
        df_combined = custom_df
        # Si la columna DISPOSITION no existe, la creamos vacía para no fallar en la codificación
        if 'DISPOSITION' not in df_combined.columns:
            df_combined['DISPOSITION'] = 'CANDIDATE' # Valor por defecto para que la codificación funcione

    else:
        print("Error: data_source no válido o custom_df es nulo.")
        return None, None

    # 2. Codificación de Etiquetas (Y)
    df_combined['DISPOSITION_ENCODED'] = df_combined['DISPOSITION'].map(DISP_MAP)
    
    # Eliminar filas donde la disposición no pudo ser mapeada o es NaN
    initial_rows = len(df_combined)
    df_combined.dropna(subset=['DISPOSITION_ENCODED'], inplace=True)
    df_combined['DISPOSITION_ENCODED'] = df_combined['DISPOSITION_ENCODED'].astype(int)
    print(f"Filas eliminadas por etiqueta faltante: {initial_rows - len(df_combined)}")
    
    df_base = df_combined.drop(columns=['DISPOSITION'])
    
    # 3. Imputación de Mediana
    all_features = COMPLEX_COLS
    
    # El imputer se ajusta al dataset de entrenamiento y se guarda.
    imputer = SimpleImputer(strategy='median')
    df_base[all_features] = imputer.fit_transform(df_base[all_features])
    
    # Guardar el imputer para usarlo en la predicción de nuevos datos
    joblib.dump(imputer, ML_MODELS_DIR / 'imputer.pkl')
    print("Imputer (SimpleImputer) guardado para uso en predicciones futuras.")

    # 4. Generación y Guardado de los dos DataFrames
    
    # --- A. COMPLEX_DATA.CSV ---
    df_complex = df_base[COMPLEX_COLS + ['DISPOSITION_ENCODED']].copy()
    complex_path = PROCESSED_DIR / 'complex_data.csv'
    df_complex.to_csv(complex_path, index=False)
    
    # --- B. SIMPLE_DATA.CSV ---
    df_simple = df_base[SIMPLE_COLS + ['DISPOSITION_ENCODED']].copy()
    simple_path = PROCESSED_DIR / 'simple_data.csv'
    df_simple.to_csv(simple_path, index=False)
    
    print(f"Total de filas finales: {len(df_simple)}")
    print(f"Archivos guardados en: {PROCESSED_DIR}")
    
    return simple_path, complex_path


if __name__ == "__main__":
    # La ejecución directa todavía es posible para testing
    simple_path, complex_path = run_preprocessing(data_source='raw_files')
    
    if simple_path:
        df_simple = pd.read_csv(simple_path)
        print("\nEstadísticas del dataset SIMPLE generado:")
        print(df_simple['DISPOSITION_ENCODED'].value_counts())

    if complex_path:
        df_complex = pd.read_csv(complex_path)
        print("\nEstadísticas del dataset COMPLEX generado:")
        print(df_complex['DISPOSITION_ENCODED'].value_counts())