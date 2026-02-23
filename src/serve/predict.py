import sys
import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from features.transformers import DataCleaner, TenureGrouper
from utils.config import load_config
from utils.config import setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

config = load_config()

def load_pipeline(model_type: str = "xgboost") -> joblib.load:
    """Loads one of the production pipeline artifacts."""

    model_dir = Path(config['paths']['model_dir'])
    
    # Get filename based on model type
    config_key = f"{model_type}_model_name"
    model_name = config['params'].get(config_key)

    if not model_name:
        logger.error(f"Could not find {config_key} in config 'params'.")
        raise ValueError(f"Unsupported model_type: {model_type}")


    logger.info(f"Loading {model_type} pipeline...")
    return joblib.load(model_dir / model_name)

def make_prediction(data: Union[pd.DataFrame, List[Dict]], pipeline) -> List[Dict]:

    # Convert list of dicts to DataFrame if necessary
    if isinstance(data, list):
        df = pd.DataFrame(data)

    # Inference
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "churn_prediction": "Yes" if pred == 1 else "No",
            "churn_probability": round(float(prob), 4)
        })
        
    return results


if __name__ == "__main__":

    # Load the model pipeline
    pipeline = load_pipeline("mlp")
    
    # Two sample raw data points (No preprocessing applied yet!)
    sample_customers = [
        {
            "gender": "Female",
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        },
        {
            "gender": "Male",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": 1889.5
        }
    ]
    
    # Predict
    results = make_prediction(sample_customers, pipeline)
    print("\n=== Prediction Results ===")
    for idx, res in enumerate(results):
        print(f"Customer {idx+1}: {res}")