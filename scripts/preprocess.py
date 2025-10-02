import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path

# Rutas
RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) 

# Normalización de Nombres de Columnas
COLUMN_MAPPING = {
    # Características Comunes
    'DISPOSITION': ['disposition', 'koi_disposition', 'tfopwg_disp'],
    'PERIOD': ['pl_orbper', 'koi_period'],
    'RADIUS': ['pl_rade', 'koi_prad', 'pl_rade'],
    'DENSITY': ['pl_dens', 'koi_srho'], # Densidad Planetaria (K2) o Estelar (Kepler). TESS se imputa.
    'NUM_PLANETS': ['sy_pnum', 'koi_count', 'pl_pnum'], 

    # Características Adicionales
    'DURATION': ['pl_trandurh', 'koi_duration'],       # Duración del Tránsito (horas)
    'TEFF': ['st_teff', 'koi_steff'],                 # Temperatura Efectiva Estelar (K)
    'DEPTH': ['pl_trandep', 'koi_depth'],             # Profundidad del Tránsito (ppm)
}

# Elección de las etiquetas
DISP_MAP = {
    'CONFIRMED': 2,
    'CANDIDATE': 1,
    'PC': 1, 
    'FALSE POSITIVE': 0,
    'FP': 0,
}

def load_and_standardize_data(file_path: Path, source_name: str, all_cols: list) -> pd.DataFrame:
    """Carga un CSV, saltándose los comentarios iniciales y seleccionando las columnas clave."""
    
    try:
        df = pd.read_csv(file_path, comment='#')
    except FileNotFoundError:
        print(f"Advertencia: Archivo no encontrado en {file_path}. Saltando.")
        return pd.DataFrame() # Devuelve un DataFrame vacío si el archivo no existe
    
    df_standard = pd.DataFrame()
    
    # Mapeo de columnas
    for target_col in all_cols:
        source_cols = COLUMN_MAPPING.get(target_col, [])
        found_col = next((col for col in source_cols if col in df.columns), None)
        
        if target_col == 'DISPOSITION':
            # Manejo especial para la columna objetivo
            if source_name == 'k2' and 'disposition' in df.columns:
                df_standard[target_col] = df['disposition']
            elif source_name == 'kepler' and 'koi_disposition' in df.columns:
                df_standard[target_col] = df['koi_disposition']
            elif source_name == 'tess' and 'tfopwg_disp' in df.columns:
                # Limpiar espacios en blanco adicionales
                df_standard[target_col] = df['tfopwg_disp'].astype(str).str.strip()
            
        elif target_col == 'DENSITY':
            # Manejo especial para densidad (crear NaN si no existe, como en TESS)
            if found_col:
                df_standard[target_col] = df[found_col]
            else:
                df_standard[target_col] = np.nan
        
        elif found_col:
            # Para el resto de columnas, simplemente las añadimos si existen
            df_standard[target_col] = df[found_col]

    return df_standard

def preprocess_exoplanet_data():
    """Ejecuta el pipeline de preprocesamiento completo y genera dos datasets."""
    
    # Definición de los dos conjuntos de características
    SIMPLE_COLS = ['PERIOD', 'RADIUS', 'DENSITY', 'NUM_PLANETS']
    COMPLEX_COLS = SIMPLE_COLS + ['DURATION', 'TEFF', 'DEPTH']
    
    # La lista total de columnas necesarias para cargar (incluida la disposición)
    ALL_REQUIRED_COLS = list(set(SIMPLE_COLS + COMPLEX_COLS + ['DISPOSITION']))
    
    # 1. Cargar y Estandarizar el Conjunto COMPLETO de datos
    print("1. Cargando y combinando datasets de K2, Kepler, y TESS (Conjunto Complejo)...")
    
    dfs = []
    # Usamos ALL_REQUIRED_COLS para asegurarnos de que cargamos todo lo necesario
    dfs.append(load_and_standardize_data(RAW_DIR / 'k2.csv', 'k2', ALL_REQUIRED_COLS))
    dfs.append(load_and_standardize_data(RAW_DIR / 'kepler.csv', 'kepler', ALL_REQUIRED_COLS))
    dfs.append(load_and_standardize_data(RAW_DIR / 'tess.csv', 'tess', ALL_REQUIRED_COLS))
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # 2. Codificación de Etiquetas (Y)
    print("2. Codificando la variable objetivo (DISPOSITION)...")
    df_combined['DISPOSITION_ENCODED'] = df_combined['DISPOSITION'].map(DISP_MAP)
    
    # Eliminar filas donde la disposición no pudo ser mapeada o es NaN
    df_combined.dropna(subset=['DISPOSITION_ENCODED'], inplace=True)
    df_combined['DISPOSITION_ENCODED'] = df_combined['DISPOSITION_ENCODED'].astype(int)
    
    # Crear un DataFrame base sin la columna de texto DISPOSITION
    df_base = df_combined.drop(columns=['DISPOSITION'])
    
    # 3. Generar los dos datasets con imputación de mediana
    
    # Columnas para imputación de mediana (Todas las características numéricas)
    all_features = COMPLEX_COLS
    
    # Imputar la mediana en el DataFrame base
    imputer = SimpleImputer(strategy='median')
    df_base[all_features] = imputer.fit_transform(df_base[all_features])
    
    # --- A. Generar COMPLEX_DATA.CSV ---
    print("\n3.a. Generando COMPLEX_DATA (Imputación por mediana)...")
    # Seleccionar solo las columnas complejas y la etiqueta del DataFrame base imputado
    df_complex = df_base[COMPLEX_COLS + ['DISPOSITION_ENCODED']].copy()
    
    # Guardar el dataset complejo
    complex_path = PROCESSED_DIR / 'complex_data.csv'
    df_complex.to_csv(complex_path, index=False)
    print(f"-> COMPLEX_DATA.CSV guardado: {df_complex.shape}")
    
    # --- B. Generar SIMPLE_DATA.CSV ---
    print("\n3.b. Generando SIMPLE_DATA (Subconjunto del complejo)...")
    # CORRECCIÓN: Creamos el df_simple como subconjunto, ya que la imputación ya se hizo.
    df_simple = df_base[SIMPLE_COLS + ['DISPOSITION_ENCODED']].copy()
    
    # Guardar el dataset simple
    simple_path = PROCESSED_DIR / 'simple_data.csv'
    df_simple.to_csv(simple_path, index=False)
    print(f"-> SIMPLE_DATA.CSV guardado: {df_simple.shape}")
    
    print("\n--- ¡Preprocesamiento Finalizado! ---")
    print(f"Total de filas limpias y codificadas: {len(df_simple)}")
    
    return df_simple, df_complex

if __name__ == "__main__":
    df_simple, df_complex = preprocess_exoplanet_data()
    
    print("\nEstadísticas de DISPOSITION_ENCODED (Ambos datasets tienen el mismo balance):")
    print(df_simple['DISPOSITION_ENCODED'].value_counts())
    
    print("\nDescripción de COMPLEX_DATA (Transpuesta, para verificar rangos de valores):")
    print(df_complex.describe().T)