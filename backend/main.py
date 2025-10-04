import argparse
from pathlib import Path

# Importar funciones modulares
from scripts.train_ML import train_ml_model
from scripts.evaluation_ML import evaluate_ml_model
from scripts.train_NN import train_nn_model
from scripts.evaluation_NN import evaluate_nn_model

# Importar utilidades de predicción (simulando la lógica de la API)
# Se asume que app/model_loader.py y app/schemas.py existen y contienen las funciones necesarias.
try:
    from app.model_loader import load_artifacts, predict_exoplanet, CLASS_NAMES
    from app.schemas import ExoplanetFeatures
except ImportError:
    # Definir un dummy si el usuario no ha creado la carpeta app/ aún
    def load_artifacts(*args): raise NotImplementedError("Crea la carpeta 'app/' con 'model_loader.py' y 'schemas.py' para usar la predicción.")
    def predict_exoplanet(*args): raise NotImplementedError("Crea la carpeta 'app/' con 'model_loader.py' y 'schemas.py' para usar la predicción.")
    CLASS_NAMES = ["FALSE POSITIVE", "CANDIDATE (PC)", "CONFIRMED (CP)"]
    print("ADVERTENCIA: Las funciones de predicción no están disponibles. Ejecuta 'pip install pydantic' y crea los archivos de la API en 'app/'.")

# --- CONSTANTES GLOBALES ---
PROCESSED_DIR = Path('data/processed')
MODELS_ROOT = Path('models')
ML_MODELS_DIR = MODELS_ROOT / 'ML'
NN_MODELS_DIR = MODELS_ROOT / 'NN'

# --- CONFIGURACIÓN DE MODELOS ---
ML_ALGORITHMS = ['RandomForest', 'XGBoost']
NN_ALGORITHMS = ['nn_custom']
DATASET_IDS = ['simple_data', 'complex_data']
MODEL_TYPES = ['ML', 'NN']

def handle_single_prediction(args):
    """
    Realiza una predicción sobre una única fila de datos.
    Requiere que la estructura de la API esté en su lugar.
    """
    print(f"\n--- Iniciando Predicción de Fila Única ({args.model_type}/{args.algorithm}/{args.dataset_id}) ---")

    # 1. Preparar los datos de la fila
    # El valor de args.features es una cadena (ej: "1.2,0.5,3.4,0.1").
    try:
        raw_features = [float(x.strip()) for x in args.features.split(',')]
        # Crear un diccionario que simule el formato de entrada de la API (ExoplanetFeatures)
        # Esto asume 4 features, AJUSTA si necesitas más:
        if len(raw_features) != 4:
             raise ValueError(f"Se esperaban 4 features, se recibieron {len(raw_features)}.")
             
        input_data = {
            'feature_1': raw_features[0],
            'feature_2': raw_features[1],
            'feature_3': raw_features[2],
            'feature_4': raw_features[3],
        }
        
    except ValueError as e:
        print(f"ERROR en el formato de features: {e}. Introduce valores numéricos separados por comas.")
        return

    # 2. Cargar el modelo y escalador
    try:
        model, scaler = load_artifacts(args.model_type, args.algorithm, args.dataset_id)
    except NotImplementedError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo o escalador: {e}")
        return
        
    # 3. Realizar y mostrar la predicción
    prediction, probabilities = predict_exoplanet(
        model, 
        scaler, 
        input_data, 
        args.model_type
    )

    print("\n================ RESULTADOS DE LA PREDICCIÓN ================")
    print(f"Modelo Usado: {args.model_type}/{args.algorithm}/{args.dataset_id}")
    print("-" * 50)
    print(f"CLASE PREDICHA: {prediction}")
    print("PROBABILIDADES:")
    for cls, prob in probabilities.items():
        print(f"  {cls:<15}: {prob * 100:.2f}%")
    print("=============================================================")


def main():
    """Configura los argumentos y dirige la ejecución."""
    
    parser = argparse.ArgumentParser(description="Controlador principal para el flujo de trabajo de Exoplanets (entrenamiento, evaluación y predicción).")
    
    # --- Subparsers (Comandos) ---
    subparsers = parser.add_subparsers(dest='command', required=True, help='Acción a ejecutar.')

    # -----------------------------
    # COMANDO: TRAIN (Entrenar)
    # -----------------------------
    parser_train = subparsers.add_parser('train', help='Entrenar un modelo ML o NN.')
    parser_train.add_argument('model_type', choices=MODEL_TYPES, help='Tipo de modelo (ML o NN).')
    parser_train.add_argument('algorithm', choices=ML_ALGORITHMS + NN_ALGORITHMS, help='Algoritmo a usar (RandomForest, XGBoost, nn_custom).')
    parser_train.add_argument('dataset_id', choices=DATASET_IDS, help='Dataset a usar para el entrenamiento (simple_data o complex_data).')
    parser_train.add_argument('--params', type=str, default='{}', help='Hiperparámetros opcionales en formato JSON. Ejemplo: \'{"n_estimators": 500}\'')

    # -----------------------------
    # COMANDO: EVALUATE (Evaluar)
    # -----------------------------
    parser_eval = subparsers.add_parser('evaluate', help='Evaluar un modelo existente con su conjunto de prueba.')
    parser_eval.add_argument('model_type', choices=MODEL_TYPES, help='Tipo de modelo (ML o NN).')
    parser_eval.add_argument('algorithm', choices=ML_ALGORITHMS + NN_ALGORITHMS, help='Algoritmo a evaluar.')
    parser_eval.add_argument('dataset_id', choices=DATASET_IDS, help='Dataset con el que se entrenó el modelo.')

    # -----------------------------
    # COMANDO: PREDICT (Predicción de fila única)
    # -----------------------------
    parser_pred = subparsers.add_parser('predict', help='Realizar una predicción sobre una única fila de datos.')
    parser_pred.add_argument('model_type', choices=MODEL_TYPES, help='Tipo de modelo a usar (ML o NN).')
    parser_pred.add_argument('algorithm', choices=ML_ALGORITHMS + NN_ALGORITHMS, help='Algoritmo a usar.')
    parser_pred.add_argument('dataset_id', choices=DATASET_IDS, help='Dataset con el que se entrenó el modelo.')
    parser_pred.add_argument('features', type=str, help='Valores de las features separados por comas. EJ: "0.1,2.5,1.0,0.4"')
    
    args = parser.parse_args()
    
    # --- Ejecución de Comandos ---
    
    if args.command == 'train':
        # Convertir JSON de parámetros a diccionario de Python
        try:
            import json
            params = json.loads(args.params)
        except json.JSONDecodeError:
            print("ERROR: El parámetro --params no es un JSON válido. Use comillas simples para la cadena JSON.")
            return

        data_path = PROCESSED_DIR / f'{args.dataset_id}.csv'
        
        if args.model_type == 'ML':
            train_ml_model(args.algorithm, data_path, ML_MODELS_DIR, params)
        elif args.model_type == 'NN':
            # Asumimos que los parámetros se dividen en arch_params y train_params
            train_nn_model(data_path, arch_params=params, train_params=params)
        
    elif args.command == 'evaluate':
        if args.model_type == 'ML':
            evaluate_ml_model(args.algorithm, args.dataset_id, ML_MODELS_DIR)
        elif args.model_type == 'NN':
            evaluate_nn_model(args.dataset_id, NN_MODELS_DIR)

    elif args.command == 'predict':
        handle_single_prediction(args)


if __name__ == "__main__":
    main()