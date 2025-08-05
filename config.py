# Configurações do Modelo ONIA XGBoost

# Configurações de Dados
DATA_DIR = "templates"
OUTPUT_FILE = "resultado.csv"
LOG_FILE = "modelo_onia.log"

# Configurações do Modelo XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "learning_rate": 0.1,
    "random_state": 52,
    "n_jobs": -1,
    "eval_metric": "mlogloss"
}

# Configurações de Validação
VALIDATION_SIZE = 0.1  # 10% para validação
RANDOM_STATE = 52

# Configurações de Normalização
USE_SCALING = True

# Configurações de Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"