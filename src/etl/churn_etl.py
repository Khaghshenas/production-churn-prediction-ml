import logging
import joblib
import pandas as pd
import pandera as pa
from pathlib import Path
from typing import Tuple
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
from src.features.transformers import DataCleaner, TenureGrouper
from src.utils.config import load_config
from src.utils.config import setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

# Schema Validation
# This ensures the ETL fails fast if the source data changes format.
def build_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema({
        "customerID": pa.Column(str, required=False),
        "tenure": pa.Column(int, checks=pa.Check.ge(0)),
        "TotalCharges": pa.Column(float, nullable=True),
        "Churn": pa.Column(str, checks=pa.Check.isin(["Yes", "No"]))
    })

# Pipeline Builder
def build_preprocessing_pipeline(cat_cols, num_cols, target_col):
 
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()) 
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine all steps into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Final Pipeline including custom logic
    full_pipeline = Pipeline(steps=[
        ("cleaner", DataCleaner(target_col=target_col)),
        # TODO: Revisit whether tenure binning should be moved outside the pipeline
        # or made data-dependent. If bins are computed from data statistics,
        # doing this before the split could introduce leakage.
        ("tenure_binning", TenureGrouper()),
        ("preprocessor", preprocessor)
    ])
    
    return full_pipeline

def run_etl():
    config = load_config()
    target = config['params']['target_col']
    
    # 1. Load & Validate
    try:
        df = pd.read_csv(config['paths']['raw_data'])

        # Convert TotalCharges to numeric, turning spaces into NaN
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


        build_schema().validate(df)
        logger.info("Data loaded and schema validated.")

        missing_cols = set(config['params']['numeric_features'] + 
                   config['params']['categorical_features']) - set(['tenure_group']) - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
    except Exception as e:
        logger.error(f"ETL Failed at Ingestion: {e}")
        return

    # 2. Pre-split Cleaning
    df = df.dropna(subset=[target])
    df[target] = df[target].map({"No": 0, "Yes": 1})
    
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['params']['test_size'], 
        random_state=config['params']['random_state'],
        stratify=y
    )

    # 4. Fit & Transform Pipeline
    # Note: categorical_features includes 'tenure_group' which is created in the pipeline
    pipeline = build_preprocessing_pipeline(
        cat_cols=config['params']['categorical_features'],
        num_cols=config['params']['numeric_features'],
        target_col=target
    )

    start = time.time()
    X_train_proc = pipeline.fit_transform(X_train)
    logger.info(f"Pipeline fit in {time.time() - start:.2f} seconds")
    X_test_proc = pipeline.transform(X_test)

    # 5. Save artifacts
    out_dir = Path(config['paths']['processed_dir'])
    model_dir = Path(config['paths']['model_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the whole pipeline.
    joblib.dump(pipeline, model_dir / "preprocessor_pipeline.joblib")
    
    # Save data for modeling
    pd.DataFrame(X_train_proc).to_csv(out_dir / "X_train_proc.csv", index=False)
    pd.DataFrame(X_test_proc).to_csv(out_dir / "X_test_proc.csv", index=False)
    pd.DataFrame(X_train).to_csv(out_dir / "X_train_raw.csv", index=False)
    pd.DataFrame(X_test).to_csv(out_dir / "X_test_raw.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    logger.info(f"ETL Pipeline Complete. Artifacts saved to {out_dir}")

if __name__ == "__main__":
    run_etl()