import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body

from src.serve.predict import load_pipeline, make_prediction
from src.utils.config import setup_logging

# Setup
setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using XGBoost and MLP models.",
    version="1.0.0"
)

# Model Cache
models = {}

@app.on_event("startup")
async def startup_event():
    """Load models once when the server starts."""
    try:
        models["xgboost"] = load_pipeline("xgboost")
        models["mlp"] = load_pipeline("mlp")
        models["uplift"] = load_pipeline("uplift")
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.post("/predict/{model_type}")
async def predict(model_type: str, data: list = Body(...)):
    
    # Check model
    if model_type not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    # Predict
    try:
        results = make_prediction(data, models[model_type], models['uplift'])
        return {"model": model_type, "results": results}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(models.keys())}