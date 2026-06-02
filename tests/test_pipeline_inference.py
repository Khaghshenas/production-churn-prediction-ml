import numpy as np
import pandas as pd
import joblib

def test_inference_pipeline():

    model = joblib.load("models/churn_xgboost_v3.joblib")

    sample = pd.DataFrame([{
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
    }])

    modified_sample = sample.copy()
    modified_sample["tenure"] = 60

    # Run inferences
    base_pred = model.predict_proba(sample)[0][1]
    modified_pred = model.predict_proba(modified_sample)[0][1]

    # 1. Valid probability bounds
    assert 0.0 <= base_pred <= 1.0
    assert 0.0 <= modified_pred <= 1.0

    # 2. Stability (determinism)
    assert np.isclose(base_pred, model.predict_proba(sample)[0][1])

    # 3. Behavioral sanity check
    assert modified_pred < base_pred