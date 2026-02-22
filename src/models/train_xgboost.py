import logging
import yaml
import joblib
import pandas as pd
import time
from pathlib import Path
import sys

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)
    
from features.transformers import DataCleaner, TenureGrouper

# Logging Setup
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load Configurations
def load_config(config_path: str = "config.yaml") -> dict:

    file_path = Path(config_path)

    if not file_path.exists():
        logger.error(f"Configuration file not found at: {file_path}")
        raise FileNotFoundError(f"Expected config.yaml at {file_path.absolute()}")

    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded successfully from {config_path}")
        
        # Basic structure validation
        required_keys = ['paths', 'params']
        if not all(key in config for key in required_keys):
            missing = [k for k in required_keys if k not in config]
            raise KeyError(f"Missing top-level keys in config: {missing}")
            
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise

def run_training_pipeline():
    config = load_config()
    processed_dir = Path(config['paths']['processed_dir'])
    
    # 1. Load Data
    # Note: We load the RAW-ish split data because the Pipeline will handle transformation
    logger.info("Loading data for training...")
    X_train = pd.read_csv(processed_dir / "X_train_raw.csv")
    X_test = pd.read_csv(processed_dir / "X_test_raw.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # 2. Load the Preprocessor created in churn_etl
    preprocessor_path = processed_dir / "preprocessor_pipeline.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run ETL first.")
    
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor from ETL artifacts.")

    # 3. Define the XGBoost Estimator
    xgboost_params = config.get('xgboost_params', {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'random_state': 42
    })
    
    model = XGBClassifier(**xgboost_params)

    # 4. Create the Full Production Pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # 5. Train the Pipeline
    logger.info("Starting Pipeline fit (Preprocessing + Training)...")
    start = time.time()
    full_pipeline.fit(X_train, y_train)
    logger.info(f"Full pipeline trained in {time.time() - start:.2f} seconds.")

    # 6. Evaluate
    preds_proba = full_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    logger.info(f"Evaluation Complete. ROC AUC: {auc:.4f}")
    
    # 7. Save the Complete Production Artifact
    model_dir = Path(config['paths']['model_dir'])
    model_name = config['paths']['xgb_model_name']
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # This file is all we need for API/Inference service
    joblib.dump(full_pipeline, model_dir / model_name)
    logger.info(f"Production pipeline saved to {model_dir}")

if __name__ == "__main__":
    run_training_pipeline()