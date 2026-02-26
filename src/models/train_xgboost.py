import logging
import joblib
import pandas as pd
import time
from pathlib import Path
import sys

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.features.transformers import DataCleaner, TenureGrouper
from src.utils.config import load_config
from src.utils.config import setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)


def run_training_pipeline():
    config = load_config()
    processed_dir = Path(config['paths']['processed_dir'])
    model_dir = Path(config['paths']['model_dir'])
    
    # 1. Load Data
    # Note: We load the RAW-ish split data because the Pipeline will handle transformation
    logger.info("Loading data for training...")
    X_train = pd.read_csv(processed_dir / "X_train_raw.csv")
    X_test = pd.read_csv(processed_dir / "X_test_raw.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # 2. Load the Preprocessor artifact created in churn_etl
    preprocessor_path = model_dir / "preprocessor_pipeline.joblib"
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

    # 6. Evaluation
    preds_proba = full_pipeline.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)

    logger.info(
        f"Evaluation Complete | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-Score: {f1:.4f} | "
        f"ROC-AUC: {auc:.4f}"
    )
    
    # 7. Save the Complete Production Artifact
    model_dir = Path(config['paths']['model_dir'])
    model_name = config['params']['xgboost_model_name']
    model_dir.mkdir(parents=True, exist_ok=True)

    # The final pipeline artifact
    joblib.dump(full_pipeline, model_dir / model_name)
    logger.info(f"Production pipeline saved to {model_dir}")

if __name__ == "__main__":
    run_training_pipeline()