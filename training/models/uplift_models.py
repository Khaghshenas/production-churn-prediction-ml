from sklearn.base import BaseEstimator, ClassifierMixin, clone

class TwoModelUplift(BaseEstimator, ClassifierMixin):
    def __init__(self, model_treat, model_control):
        self.model_treat = model_treat
        self.model_control = model_control

    def fit(self, X, y, treatment=None):
        
        if treatment is None:
            treatment = assign_synthetic_treatment(X)
        
        # Split data
        X_treat = X[treatment == 1]
        y_treat = y[treatment == 1]

        X_control = X[treatment == 0]
        y_control = y[treatment == 0]

        # Clone models for sklearn compatibility
        self.model_treat_ = clone(self.model_treat)
        self.model_control_ = clone(self.model_control)

        self.model_treat_.fit(X_treat, y_treat)
        self.model_control_.fit(X_control, y_control)

        return self  # required for Pipeline compatibility

    def predict(self, X):
        p_treat = self.model_treat_.predict_proba(X)[:, 1]
        p_control = self.model_control_.predict_proba(X)[:, 1]
        # Churn Uplift = P(Churn|Control) - P(Churn|Treatment)
        # Higher positive score = higher "persuadability" (reduction in churn risk)
        return p_control - p_treat