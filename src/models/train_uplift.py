import logging
import joblib
import pandas as pd
import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklift.metrics import uplift_at_k, qini_auc_score, qini_curve
from sklift.metrics import qini_curve


from src.utils.config import load_config
from src.utils.config import setup_logging
from src.models.uplift_models import TwoModelUplift

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

def assign_synthetic_treatment(X: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Randomly assign a synthetic treatment for uplift modeling.

    This function is used for demonstration purposes to create a treatment column.
    In real-world scenarios, the treatment column should come from experimental or observational data.

    """
    np.random.seed(seed)
    treatment = np.random.binomial(1, 0.5, size=len(X))
    #X_with_treatment = X.copy()
    #X_with_treatment["treatment"] = treatment
    return treatment

def evaluate_uplift(y_true, uplift_scores, treatment, config):
    logger.info("Evaluating Uplift metrics...")
    
    metrics = {
        "qini_auc": qini_auc_score(y_true, uplift_scores, treatment),
        "uplift_at_20pct": uplift_at_k(y_true, uplift_scores, treatment, strategy="by_group", k=0.2)
    }
    
    # Data for plotting
    x_basis, y_qini = qini_curve(y_true, uplift_scores, treatment)
    
    plt.figure(figsize=(10, 6))
    # metrics["qini_auc"] is now defined
    plt.plot(x_basis, y_qini, label=f'Model (Qini AUC={metrics["qini_auc"]:.4f})')
    plt.plot([x_basis[0], x_basis[-1]], [y_qini[0], y_qini[-1]], '--', label='Random')
    
    plt.title("Qini Curve: Customer Retention Uplift")
    plt.xlabel("Number of targeted customers")
    plt.ylabel("Cumulative Incremental Outcome")
    plt.legend()
    plt.grid(True)
    
    # Handle plot path from config
    plot_path = Path(config['paths']['plot_path'])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Qini Curve saved to {plot_path}")
    return metrics

def run_uplift_pipeline() -> tuple[pd.Series, pd.Series, pd.Series, dict]:
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

    # 3. Load the Preprocessor artifact created in churn_etl
    preprocessor_path = model_dir / "preprocessor_pipeline.joblib"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run ETL first.")
    
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor from ETL artifacts.")

    # 4. Generate treatments for both train and test sets before the pipeline
    treat_train = assign_synthetic_treatment(X_train)
    treat_test = assign_synthetic_treatment(X_test)

    # 5. Define the XGBoost Estimator
    xgboost_params = config.get('xgboost_params', {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'random_state': 42
    })

    model_t = XGBClassifier(**xgboost_params)
    model_c = XGBClassifier(**xgboost_params)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("uplift", TwoModelUplift(
            model_treat=model_t,
            model_control=model_c
        ))
    ])

    # 6. Train the Pipeline
    logger.info("Starting Pipeline fit (Preprocessing + Training both Treatment and Control)...")
    start = time.time()
    full_pipeline.fit(X_train, y_train, uplift__treatment=treat_train)
    logger.info(f"Full pipeline trained in {time.time() - start:.2f} seconds.")

    # 7. Evaluation
    # Predict only needs X; it uses both internal models to find the delta
    uplift_scores = full_pipeline.predict(X_test)
    metrics = evaluate_uplift(
        y_test.values, 
        uplift_scores, 
        treat_test, 
        config
    )
    
    logger.info("Evaluation Complete: ")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.upper()}: {value:.4f}")  
    
    # 8. Save the Complete Production Artifact
    model_dir = Path(config['paths']['model_dir'])
    model_name = config['params']['uplift_model_name']
    model_dir.mkdir(parents=True, exist_ok=True)

    # The final pipeline artifact
    joblib.dump(full_pipeline, model_dir / model_name)
    logger.info(f"Production pipeline saved to {model_dir}")

if __name__ == "__main__":
    run_uplift_pipeline()
