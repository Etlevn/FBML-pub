"""
Tree model implementations and utilities.

This module provides:
- Sample weights computation for handling class imbalance
- Tree model creation and training functions
- Model prediction utilities
- Hyperparameter optimization support (Optuna and RandomizedSearchCV)
"""

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Optional imports for advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False




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


def create_tree_model(model_type: str, use_balanced: bool = False, **kwargs):
    """
    Create a tree model according to the configuration.
    
    Args:
        model_type: Model type ('decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting',
                              'hist_gradient_boosting', 'xgboost', 'lightgbm', 'catboost')
        use_balanced: Whether to use balanced/weighted variants (available for all model types).
        **kwargs: Additional model parameters for hyperparameter optimization.
    
    Returns:
        Model instance.
    """
    if model_type == 'decision_tree':
        params = {}
        if use_balanced:
            params['class_weight'] = 'balanced'
        return DecisionTreeClassifier(**params)
    
    elif model_type == 'random_forest':
        params = kwargs.copy()
        if use_balanced:
            params['class_weight'] = 'balanced'
        # Set defaults for parameters not provided in kwargs
        if 'random_state' not in params:
            params['random_state'] = 42
        if 'n_jobs' not in params:
            params['n_jobs'] = -1
        return RandomForestClassifier(**params)
    
    elif model_type == 'extra_trees':
        params = {}
        if use_balanced:
            params['class_weight'] = 'balanced'
        return ExtraTreesClassifier(**params)
    
    elif model_type == 'gradient_boosting':
        # GradientBoostingClassifier will use sample_weight in training
        return GradientBoostingClassifier()
    
    elif model_type == 'hist_gradient_boosting':
        # HistGradientBoostingClassifier will use sample_weight in training
        return HistGradientBoostingClassifier()
    
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install it with: pip install xgboost")
        # XGBoost uses scale_pos_weight for class imbalance
        return XGBClassifier(verbosity=0)
    
    elif model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please install it with: pip install lightgbm")
        params = {'verbose': -1}
        if use_balanced:
            params['class_weight'] = 'balanced'
        return LGBMClassifier(**params)
    
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Please install it with: pip install catboost")
        # CatBoost uses class_weights parameter
        return CatBoostClassifier(verbose=False)
    
    else:
        raise ValueError(f"Unknown tree model type: {model_type}")


def train_tree_model(model, X_train, y_train, model_type: str, use_balanced: bool = False):
    """
    Train a tree model with automatic handling of balanced/weighted variants.
    
    Args:
        model: Model instance.
        X_train: Training features.
        y_train: Training labels.
        model_type: Model type.
        use_balanced: Whether to use a balanced/weighted variant.
    """
    if model_type in ['gradient_boosting', 'hist_gradient_boosting', 'xgboost', 'catboost'] and use_balanced:
        # These models use sample_weight parameter
        sample_weight = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    
    else:
        # Other models use class_weight parameter (already set in create_tree_model)
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
    y_pred_class = model.predict(X_test)
    
    # Get continuous predictions (probabilities)
    if hasattr(model, 'predict_proba'):
        y_pred_cont = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_cont = model.decision_function(X_test)
    else:
        # Use predictions if neither probabilities nor a decision function are available
        y_pred_cont = y_pred_class.astype(float)
    
    # CatBoost sometimes returns shape (n, 1) instead of (n,)
    if y_pred_class.ndim > 1 and y_pred_class.shape[1] == 1:
        y_pred_class = y_pred_class.flatten()
    
    return y_pred_class, y_pred_cont


def get_feature_importance(model, model_type: str, feature_names: list):
    """
    Get model feature importance.
    
    Args:
        model: Trained model.
        model_type: Model type.
        feature_names: List of feature names.
    
    Returns:
        dict: Mapping from feature names to importance values.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Use absolute coefficients for linear models
        importances = np.abs(model.coef_)
    else:
        # Return zeros if feature importance is unavailable
        importances = np.zeros(len(feature_names))
    
    # Create the importance dictionary
    importance_dict = {}
    for i, feature_name in enumerate(feature_names):
        if i < len(importances):
            importance_dict[feature_name] = float(importances[i])
        else:
            importance_dict[feature_name] = 0.0
    
    return importance_dict


def get_default_hyperparameter_space(model_type: str):
    """
    Get the default hyperparameter search space.
    
    Args:
        model_type: Model type.
    
    Returns:
        dict: Hyperparameter search space.
    """
    if model_type == 'random_forest':
        return {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [15, 20, 25, 30, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5],
            'max_samples': [0.7, 0.8, 0.9, None],
            'bootstrap': [True, False],
        }
    elif model_type == 'extra_trees':
        return {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [15, 20, 25, 30, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5],
            'bootstrap': [True, False],
        }
    else:
        # For other models, return empty dict (no hyperparameter tuning supported yet)
        return {}


def optimize_hyperparameters_optuna(
    model_type: str,
    X_train,
    y_train,
    n_trials: int = 50,
    cv_folds: int = 3,
    scoring: str = 'roc_auc',
    random_state: int = 42,
    use_balanced: bool = False,
    n_jobs: int = -1,
    enable_pruning: bool = True,
    pruner_type: str = 'median',
    n_startup_trials: int = 10,
    n_warmup_steps: int = 5,
    param_space: dict = None
):
    """
    Optimize hyperparameters with Optuna.
    
    Args:
        model_type: Model type.
        X_train: Training features.
        y_train: Training labels.
        n_trials: Number of optimization trials.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric.
        random_state: Random seed.
        use_balanced: Whether to use balanced class weights.
        n_jobs: Number of parallel jobs.
        enable_pruning: Whether to enable pruning.
        pruner_type: Pruner type ('median', 'percentile', 'successive_halving').
        n_startup_trials: Number of trials before pruning starts.
        n_warmup_steps: Number of warmup steps.
        param_space: Custom search space; use the default space if None.
    
    Returns:
        Optimized model instance, best parameters, and best score.
    """
    try:
        import optuna
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError("Optuna is not available. Please install it with: pip install optuna")
    
    # Get the default search space if none was supplied
    if param_space is None:
        param_space = get_default_hyperparameter_space(model_type)
    
    if not param_space:
        raise ValueError(f"No hyperparameter space defined for {model_type}")
    
    # Create time-series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    def objective(trial):
        # Suggest parameter values from the search space
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in param_values):
                    # Numeric parameters
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            min(param_values), 
                            max(param_values),
                            step=max(1, (max(param_values) - min(param_values)) // (len(param_values) - 1)) if len(param_values) > 1 else 1
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            min(param_values), 
                            max(param_values)
                        )
                else:
                    # Categorical parameters, including None
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                params[param_name] = param_values
        
        # Create the model
        model = create_tree_model(model_type, use_balanced, **params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=tscv, 
            scoring=scoring,
            n_jobs=n_jobs
        )
        
        return scores.mean()
    
    # Create the pruner
    pruner = None
    if enable_pruning:
        if pruner_type == 'median':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps
            )
        elif pruner_type == 'percentile':
            pruner = optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps
            )
        elif pruner_type == 'successive_halving':
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource='auto',
                reduction_factor=4
            )
    
    # Create and optimize the study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=pruner
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Create and train a model with the best parameters
    best_params = study.best_params
    best_model = create_tree_model(model_type, use_balanced, **best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params, study.best_value


def optimize_hyperparameters_random(
    model_type: str,
    X_train,
    y_train,
    n_iter: int = 50,
    cv_folds: int = 3,
    scoring: str = 'roc_auc',
    random_state: int = 42,
    use_balanced: bool = False,
    n_jobs: int = -1,
    param_space: dict = None
):
    """
    Optimize hyperparameters with RandomizedSearchCV.
    
    Args:
        model_type: Model type.
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of random search iterations.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric.
        random_state: Random seed.
        use_balanced: Whether to use balanced class weights.
        n_jobs: Number of parallel jobs.
        param_space: Custom search space; use the default space if None.
    
    Returns:
        Optimized model instance, best parameters, and best score.
    """
    # Get the default search space if none was supplied
    if param_space is None:
        param_space = get_default_hyperparameter_space(model_type)
    
    if not param_space:
        raise ValueError(f"No hyperparameter space defined for {model_type}")
    
    # Create the base model
    base_model = create_tree_model(model_type, use_balanced)
    
    # Create time-series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Run random search
    search = RandomizedSearchCV(
        base_model,
        param_space,
        n_iter=n_iter,
        cv=tscv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=True,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_, search.best_score_


def validate_parameter_stability(param_results, cv_threshold=0.1):
    """
    Validate parameter stability.
    
    Args:
        param_results: List of dictionaries, each containing the best parameters for one window.
        cv_threshold: Coefficient of variation threshold (0.1 = 10%).
    
    Returns:
        is_stable: bool indicating whether parameters are stable.
        stability_scores: dict of per-parameter stability scores (coefficients of variation).
    """
    if not param_results or len(param_results) < 2:
        return True, {}
    
    # Extract numeric parameters
    numeric_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    
    stability_scores = {}
    for param in numeric_params:
        values = []
        for p in param_results:
            if param in p and p[param] is not None:
                values.append(p[param])
        
        if len(values) >= 2:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / (mean_val + 1e-6)  # Coefficient of variation
            stability_scores[param] = cv
    
    # Parameters are stable if every coefficient of variation is below the threshold
    is_stable = all(cv < cv_threshold for cv in stability_scores.values())
    
    return is_stable, stability_scores
