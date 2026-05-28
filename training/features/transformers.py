import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformers created for preprocessing steps
class DataCleaner(BaseEstimator, TransformerMixin):
    """Basic cleaning that doesn't require 'fitting' (e.g. dropping columns)."""
    def __init__(self, target_col: str):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Strip whitespace from string columns
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype(str).str.strip()
        
        # Drop customerID as it is not a predictive feature
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])
        return X

class TenureGrouper(BaseEstimator, TransformerMixin):
    """Custom logic to bin tenure into groups."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "tenure" in X.columns:
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[0, 12, 24, 48, 72],
                labels=["0-12", "12-24", "24-48", "48-72"],
                include_lowest=True,
            )
            X = X.drop(columns=["tenure"])
        return X