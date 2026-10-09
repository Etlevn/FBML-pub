"""
Importance calculation module for linear models.

This module provides various importance calculation methods for linear models:
- Coefficient-based importance
- Permutation importance
- SHAP importance
"""

import os
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def compute_feature_importance_linear(model, feature_names, target_class=1):
    """
    Compute linear model feature importance from absolute coefficient values.
    
    Args:
        model: Trained linear model (e.g., LinearRegression, Ridge, LogisticRegression).
        feature_names: List of feature names.
        target_class: Target class whose coefficients are used for classification models.
    
    Returns:
        dict: Mapping from feature names to importance values.
    """
    # Get coefficients
    coef = model.coef_
    
    # Handle different coefficient matrix shapes
    # Binary LogisticRegression has coef_ shape (1, n_features); use coef_[0]
    # Multiclass LogisticRegression has coef_ shape (n_classes, n_features); use coef_[target_class]
    # Linear regression/Ridge has coef_ shape (n_features,)
    if coef.ndim == 2:
        # Multiclass case: select the target class coefficients
        if coef.shape[0] > target_class:
            coefficients = np.abs(coef[target_class])
        else:
            # Use the final class if target_class is out of range
            coefficients = np.abs(coef[-1])
    else:
        # Single-output case (linear regression)
        coefficients = np.abs(coef)
    
    # Create the importance dictionary
    importance = {}
    for i, feature_name in enumerate(feature_names):
        if i < len(coefficients):
            importance[feature_name] = float(coefficients[i])
        else:
            importance[feature_name] = 0.0
    
    return importance


def compute_permutation_importance_linear(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    target_class: int = 1,
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute permutation importance.
    """
    from sklearn.inspection import permutation_importance
    
    # Use model predictions as target values
    y_pred = model.predict(X_test)
    
    # Compute permutation importance
    # Select a scoring metric appropriate for the model type
    if hasattr(model, 'predict_proba'):
        # Classification models use accuracy
        scoring = 'accuracy'
    else:
        # Regression models use neg_mean_squared_error
        scoring = 'neg_mean_squared_error'
    
    tqdm.write(f"[INFO] Computing permutation importance with {n_repeats} repeats...")
    perm_importance = permutation_importance(
        model, X_test, y_pred, 
        n_repeats=n_repeats, 
        random_state=random_state,
        scoring=scoring
    )
    
    # Convert to dictionary format
    importance_dict = {}
    for i, feature_name in enumerate(feature_names):
        importance_dict[feature_name] = perm_importance.importances_mean[i]
    
    return importance_dict


def compute_shap_importance_linear(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    target_class: int = 1,
    background_samples: int = 100,
    max_eval_samples: int = 1000
) -> Dict[str, float]:
    """
    Compute SHAP importance.
    """
    try:
        import shap
    except ImportError:
        tqdm.write("[WARN] SHAP not available, skipping SHAP importance")
        return {name: 0.0 for name in feature_names}
    
    # Create background data
    if background_samples < len(X_test):
        background = X_test[:background_samples]
    else:
        background = X_test
    
    tqdm.write(f"[INFO] Computing SHAP importance with {len(background)} background samples...")
    
    # Create the SHAP explainer
    explainer = shap.Explainer(model, background)
    
    # Compute SHAP values
    eval_samples = min(max_eval_samples, len(X_test))
    tqdm.write(f"[INFO] Computing SHAP values for {eval_samples} samples...")
    shap_values = explainer(X_test[:eval_samples])
    
    # Use mean absolute SHAP values as importance
    if hasattr(shap_values, 'values'):
        # SHAP v0.40+
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    else:
        # Older SHAP versions
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Convert to dictionary format
    importance_dict = {}
    for i, feature_name in enumerate(feature_names):
        importance_dict[feature_name] = float(mean_abs_shap[i])
    
    return importance_dict


def compute_multiple_importance_linear(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    methods: List[str] = None,
    target_class: int = 1,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute results for multiple importance methods.
    
    Args:
        model: Trained linear model.
        X_test: Test data.
        feature_names: List of feature names.
        methods: Importance methods, e.g., ['coefficient', 'permutation', 'shap'].
        target_class: Target class.
        **kwargs: Additional parameters.
    
    Returns:
        dict: {method: {feature_name: importance_score}}
    """
    if methods is None:
        methods = ['coefficient']
    
    results = {}
    
    # Add a progress bar for importance computation
    method_iter = tqdm(methods, desc="Computing Importance", leave=False)
    for method in method_iter:
        method_iter.set_description(f"Computing {method} importance")
        
        if method == 'coefficient':
            results[method] = compute_feature_importance_linear(
                model, feature_names, target_class=target_class
            )
        elif method == 'permutation':
            # Only pass permutation-specific parameters
            perm_kwargs = {
                'n_repeats': kwargs.get('n_repeats', 5),
                'random_state': kwargs.get('random_state', 42)
            }
            results[method] = compute_permutation_importance_linear(
                model, X_test, feature_names, target_class=target_class, **perm_kwargs
            )
        elif method == 'shap':
            # Only pass SHAP-specific parameters
            shap_kwargs = {
                'background_samples': kwargs.get('background_samples', 100),
                'max_eval_samples': kwargs.get('max_eval_samples', 1000)
            }
            results[method] = compute_shap_importance_linear(
                model, X_test, feature_names, target_class=target_class, **shap_kwargs
            )
        else:
            tqdm.write(f"[WARN] Unknown importance method: {method}")
        
        method_iter.set_description(f"Completed {method} importance")
    
    return results


def append_feature_importance_wide(
    importance: Dict[str, float],
    feature_names: List[str],
    out_csv: str,
    idx: int,
) -> str:
    """
    Append one fold's importance as a wide row to a single CSV.
    Schema: columns = ['Idx'] + feature_names; values normalized (sum=1)
    If file absent -> write header; else append without header.
    Missing features will be filled with 0 and ordered as feature_names.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    row = {name: float(importance.get(name, 0.0)) for name in feature_names}
    # ensure normalization robustness
    total = float(sum(row.values()))
    if total > 0:
        row = {k: (v / total) for k, v in row.items()}

    data = {"Idx": idx}
    data.update(row)
    df_row = pd.DataFrame([data], columns=["Idx"] + list(feature_names))

    write_header = not os.path.exists(out_csv)
    df_row.to_csv(out_csv, mode='a', header=write_header, index=False)
    return out_csv


def save_multiple_importance_linear(
    importance_results: Dict[str, Dict[str, float]],
    feature_names: List[str],
    fold_idx: int,
    output_dir: str,
    model_label: str,
    fold_suffix_template: str = '_imp_{method}_fold',
    summary_suffix_template: str = '_imp_{method}_sum',
    window_suffix: str = ""
) -> None:
    """
    Save results from multiple importance methods to separate files.
    
    Args:
        importance_results: {method: {feature_name: importance_score}}
        feature_names: List of feature names.
        fold_idx: Fold index.
        output_dir: Output directory.
        model_label: Model label.
        fold_suffix_template: Fold-level filename suffix template (default: '_imp_{method}_fold').
        summary_suffix_template: Summary filename suffix template (default: '_imp_{method}_sum').
        window_suffix: Window suffix for batch mode, e.g., "_0-10".
    """
    for method, importance_dict in importance_results.items():
        # Generate filenames from templates and append the window suffix in batch mode
        suffix = fold_suffix_template.format(method=method)
        filename = f"{model_label}{suffix}{window_suffix}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save importance data
        append_feature_importance_wide(
            importance_dict, feature_names, filepath, idx=fold_idx
        )
        
        tqdm.write(f"[MAIN] {method.upper()} importance saved to: {filename}")


def generate_importance_summary(imp_csv_path: str, summary_csv_path: str) -> str:
    """
    Generate summary statistics from an importance fold file.
    
    Args:
        imp_csv_path: Input importance fold file path containing all folds.
        summary_csv_path: Output summary statistics file path.
    
    Returns:
        str: Summary statistics file path.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    
    # Read importance fold data
    if not os.path.exists(imp_csv_path):
        raise FileNotFoundError(f"Importance fold file not found: {imp_csv_path}")
    
    df_imp = pd.read_csv(imp_csv_path)
    
    # Get all feature columns except Idx
    feature_cols = [col for col in df_imp.columns if col != 'Idx']
    
    # Compute statistics for each feature
    summary_data = []
    for feature in feature_cols:
        values = df_imp[feature].values
        summary_data.append({
            'feature': feature,
            'mean': np.mean(values),
            'std': np.std(values),
            'med': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        })
    
    # Create the summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by mean in descending order
    df_summary = df_summary.sort_values('mean', ascending=False).reset_index(drop=True)
    
    # Save the summary file
    df_summary.to_csv(summary_csv_path, index=False)
    
    return summary_csv_path


def generate_multiple_importance_summary(
    output_dir: str,
    model_label: str,
    methods: List[str] = None,
    fold_suffix_template: str = '_imp_{method}_fold',
    summary_suffix_template: str = '_imp_{method}_sum'
) -> None:
    """
    Generate summary files for multiple importance methods.
    
    Args:
        output_dir: Output directory.
        model_label: Model label.
        methods: List of importance methods.
        fold_suffix_template: Fold-level filename suffix template (default: '_imp_{method}_fold').
        summary_suffix_template: Summary filename suffix template (default: '_imp_{method}_sum').
    """
    if methods is None:
        methods = ['coefficient']
    
    for method in methods:
        # Generate filenames from templates
        fold_suffix = fold_suffix_template.format(method=method)
        summary_suffix = summary_suffix_template.format(method=method)
        
        imp_fold_path = os.path.join(output_dir, f"{model_label}{fold_suffix}.csv")
        imp_sum_path = os.path.join(output_dir, f"{model_label}{summary_suffix}.csv")
        
        if os.path.exists(imp_fold_path):
            try:
                generate_importance_summary(imp_fold_path, imp_sum_path)
                tqdm.write(f"[MAIN] {method.upper()} importance summary saved to: {model_label}{summary_suffix}.csv")
            except Exception as e:
                tqdm.write(f"[WARN] Failed to generate {method} importance summary: {e}")
        else:
            tqdm.write(f"[WARN] {method} importance file not found: {imp_fold_path}")


def aggregate_temporal_importance(
    importance_results: Dict[str, Dict[str, float]],
    aggregation_method: str = 'mean'
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate importance for the same feature across time steps.
    
    Args:
        importance_results: {method: {feature_name: importance_score}}
        aggregation_method: Aggregation method ('mean', 'max', 'sum').
    
    Returns:
        dict: Aggregated importance results.
    """
    aggregated_results = {}
    
    for method, method_results in importance_results.items():
        # Group by feature name after removing the time-step suffix
        feature_groups = {}
        for feature_name, importance_score in method_results.items():
            # Extract the base feature name by removing suffixes such as _t0 and _t1
            if '_t' in feature_name:
                base_feature = '_'.join(feature_name.split('_')[:-1])
            else:
                base_feature = feature_name
            
            if base_feature not in feature_groups:
                feature_groups[base_feature] = []
            feature_groups[base_feature].append(importance_score)
        
        # Aggregate each feature group
        aggregated_method_results = {}
        for base_feature, scores in feature_groups.items():
            if aggregation_method == 'mean':
                aggregated_score = np.mean(scores)
            elif aggregation_method == 'max':
                aggregated_score = np.max(scores)
            elif aggregation_method == 'sum':
                aggregated_score = np.sum(scores)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            aggregated_method_results[base_feature] = aggregated_score
        
        aggregated_results[method] = aggregated_method_results
    
    return aggregated_results


def generate_cross_method_importance_summary(
    output_dir: str,
    model_label: str,
    methods: List[str] = None,
    fold_suffix_template: str = '_imp_{method}_fold',
    cross_method_suffix: str = '_imp_sum'
) -> None:
    """
    Generate an importance summary across methods.
    
    Args:
        output_dir: Output directory.
        model_label: Model label.
        methods: List of importance methods.
        fold_suffix_template: Fold-level filename suffix template.
        cross_method_suffix: Filename suffix for the summary across methods.
    """
    if methods is None:
        methods = ['coefficient']
    
    # Read summary files for all methods
    method_summaries = {}
    for method in methods:
        summary_suffix = fold_suffix_template.replace('_fold', '_sum').format(method=method)
        summary_path = os.path.join(output_dir, f"{model_label}{summary_suffix}.csv")
        
        if os.path.exists(summary_path):
            try:
                df_summary = pd.read_csv(summary_path)
                # Extract the mean column as the importance for this method
                method_summaries[method] = df_summary.set_index('feature')['mean'].to_dict()
            except Exception as e:
                tqdm.write(f"[WARN] Failed to read {method} summary: {e}")
        else:
            tqdm.write(f"[WARN] {method} summary file not found: {summary_path}")
    
    if not method_summaries:
        tqdm.write("[WARN] No method summaries found, skipping cross-method summary")
        return
    
    # Get all features
    all_features = set()
    for method_data in method_summaries.values():
        all_features.update(method_data.keys())
    all_features = sorted(list(all_features))
    
    # Create the DataFrame summarizing all methods
    cross_method_data = {'feature': all_features}
    for method, method_data in method_summaries.items():
        cross_method_data[method] = [method_data.get(feature, 0.0) for feature in all_features]
    
    df_cross_method = pd.DataFrame(cross_method_data)
    
    # Save the summary across methods
    cross_method_path = os.path.join(output_dir, f"{model_label}{cross_method_suffix}.csv")
    df_cross_method.to_csv(cross_method_path, index=False)
    
    tqdm.write(f"[MAIN] Cross-method importance summary saved to: {model_label}{cross_method_suffix}.csv")
