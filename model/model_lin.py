"""
Linear model implementations and utilities.

This module provides:
- Sample weights computation for handling class imbalance
- Linear model creation and training functions
- Model prediction utilities
"""

import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression, SGDClassifier


def compute_sample_weights(y_train):
    """
    Compute sample weights to handle class imbalance.
    
    Args:
        y_train: Training label array.
    
    Returns:
        sample_weight: Array of per-sample weights.
    """
    # Count occurrences of each class
    class_counts = Counter(y_train)
    n_total = len(y_train)
    n_classes = len(class_counts)

    # Compute class-level balanced weights
    class_weights = {k: n_total / (n_classes * v) for k, v in class_counts.items()}

    # Generate per-sample weights
    sample_weight = np.array([class_weights[y] for y in y_train])

    return sample_weight


def create_linear_model(model_type: str, use_balanced: bool = False):
    """
    Create a linear model according to the configuration.
    
    Args:
        model_type: Model type ('ols', 'ridge', 'logistic', 'sgd_hinge', 'sgd_log', 'sgd_modified_huber').
        use_balanced: Whether to use balanced class weights (classification models only).
    
    Returns:
        Model instance.
    """
    if model_type == 'ols':
        return LinearRegression()
    elif model_type == 'ridge':
        if use_balanced:
            return RidgeClassifier(class_weight='balanced')
        else:
            return RidgeClassifier()
    elif model_type == 'logistic':
        if use_balanced:
            return LogisticRegression(class_weight='balanced')
        else:
            return LogisticRegression()
    elif model_type == 'sgd_hinge':
        if use_balanced:
            return SGDClassifier(loss='hinge', class_weight='balanced')
        else:
            return SGDClassifier(loss='hinge')
    elif model_type == 'sgd_log':
        if use_balanced:
            return SGDClassifier(loss='log_loss', class_weight='balanced')
        else:
            return SGDClassifier(loss='log_loss')
    elif model_type == 'sgd_modified_huber':
        if use_balanced:
            return SGDClassifier(loss='modified_huber', class_weight='balanced')
        else:
            return SGDClassifier(loss='modified_huber')
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_linear_model(model, X_train, y_train, model_type: str, use_balanced: bool = False):
    """
    Train a linear model with automatic handling of balanced weights.
    
    Args:
        model: Model instance.
        X_train: Training features.
        y_train: Training labels.
        model_type: Model type.
        use_balanced: Whether to use balanced weights.
    """
    if use_balanced and model_type == 'ols':
        # OLS uses the sample_weight parameter
        sample_weight = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        # Other models use class_weight, already set in create_linear_model
        model.fit(X_train, y_train)


def get_model_predictions(model, X_test, model_type: str):
    """
    Get predictions according to the model type.
    
    Args:
        model: Trained model.
        X_test: Test data.
        model_type: Model type.
    
    Returns:
        y_pred_class: Class predictions.
        y_pred_cont: Continuous predictions used for AUC computation.
    """
    if model_type == 'ols':
        y_pred_cont = model.predict(X_test)
        y_pred_class = (y_pred_cont > 0.5).astype(int)
    elif model_type == 'ridge':
        y_pred_class = model.predict(X_test)
        y_pred_cont = model.decision_function(X_test)
    elif model_type == 'logistic':
        y_pred_class = model.predict(X_test)
        y_pred_cont = model.predict_proba(X_test)[:, 1]
    elif model_type in ['sgd_hinge', 'sgd_log', 'sgd_modified_huber']:
        y_pred_class = model.predict(X_test)
        y_pred_cont = model.decision_function(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return y_pred_class, y_pred_cont
