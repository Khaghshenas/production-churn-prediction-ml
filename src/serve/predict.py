import sys
import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List

from src.features.transformers import DataCleaner, TenureGrouper
from src.utils.config import load_config
from src.utils.config import setup_logging

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

def make_prediction(data: Union[pd.DataFrame, List[Dict]], pipeline, uplift_pipeline=None) -> List[Dict]:

    # Convert list of dicts to DataFrame if necessary
    if isinstance(data, list):
        df = pd.DataFrame(data)

    # Inference - churn prediction
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    # Inference - Uplift predictions (if provided)
    # This calls our TwoModelUplift.predict() which we set as p_control - p_treat
    uplift_scores = uplift_pipeline.predict(df) if uplift_pipeline else [None] * len(df)

    results = []
    for pred, prob, uplift in zip(predictions, probabilities, uplift_scores):
        res = {
            "churn_prediction": "Yes" if pred == 1 else "No",
            "churn_probability": round(float(prob), 4)
        }
        if uplift is not None:
            res["uplift_score"] = round(float(uplift), 4)
            res["recommendation"] = "High Priority" if uplift > 0.05 else "Standard"
        
        results.append(res)
        
    return results


if __name__ == "__main__":

    # Load the model pipeline
    model_pipeline = load_pipeline("mlp")
    
    # Load the uplif pipeline
    uplift_pipeline = load_pipeline("uplift")

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
    
    # Inference
    results = make_prediction(sample_customers, model_pipeline, uplift_pipeline)
    logger.info("Final Inference Results:")
    for idx, res in enumerate(results):
        logger.info(f"Customer {idx+1}:")
        for key, value in res.items():
            logger.info(f"  {key.replace('_', ' ').title()}: {value}")