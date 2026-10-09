"""
Importance calculation module for neural network models.

This module provides various importance calculation methods for neural network models:
- Gradient-based importance
- DeepSHAP importance
- Integrated Gradients importance
"""

import os
from typing import List, Literal, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import warnings


def compute_feature_importance(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    target_class: int = 1,
    time_aggregation: Literal["mean", "last"] = "mean",
) -> Dict[str, float]:
    """
    Compute gradient-based feature importance for sequence inputs.

    Importance(x_j) = E_t E_i [ ( d score / d x_{i,t,j} )^2 ]
    - score uses the target class logit by default (avoids softmax saturation)
    - for sequence inputs, aggregate over time by mean (default) or take last step
    - final importance is normalized to sum to 1

    Args:
        model: trained model
        loader: DataLoader yielding (X, y) with X shape [B, T, F]
        feature_names: list of feature column names (len F)
        device: device string
        target_class: which class logit/prob to explain (int)
        time_aggregation: 'mean' over time or 'last' time step only

    Returns:
        dict mapping feature name -> normalized importance
    """
    model.eval()

    num_features = len(feature_names)
    grad_sq_sum = torch.zeros(num_features, dtype=torch.float64, device=device)
    count = 0

    progress_bar = tqdm(loader, desc="Gradient Importance", leave=False)
    for xb, _ in progress_bar:
        xb = xb.to(device)
        xb = xb.detach()
        xb.requires_grad_(True)

        logits = model(xb)  # [B, C]
        # use target class logit for smoother gradients
        score = logits[:, target_class].sum()

        if xb.grad is not None:
            xb.grad.zero_()
        model.zero_grad(set_to_none=True)
        score.backward()

        grad = xb.grad  # [B, T, F]
        if time_aggregation == "last":
            grad = grad[:, -1, :]
        else:
            grad = grad.mean(dim=1)

        grad_sq = grad.pow(2.0)  # [B, F]
        grad_sq_sum += grad_sq.sum(dim=0).to(device=grad_sq_sum.device, dtype=grad_sq_sum.dtype)
        count += grad_sq.size(0)

        # free graph
        xb.requires_grad_(False)

    if count == 0:
        return {name: 0.0 for name in feature_names}

    imp = grad_sq_sum / float(count)
    imp = imp.clamp(min=0)
    total = float(imp.sum().item())
    if total <= 0:
        norm = [0.0 for _ in feature_names]
    else:
        norm = (imp / imp.sum()).tolist()

    return {name: float(val) for name, val in zip(feature_names, norm)}


def compute_feature_importance_deepshap(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    target_class: int = 1,
    background_max_samples: int = 512,
    time_aggregation: Literal["mean", "last"] = "mean",
) -> Dict[str, float]:
    """
    DeepSHAP importance using shap.DeepExplainer for sequence inputs.
    Steps:
      1) Build a background sample (subset of loader inputs)
      2) Explain test batches; get SHAP values per feature
      3) Aggregate over time (mean/last) and over samples (mean absolute)
      4) Normalize across features to sum to 1
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap is required for DeepSHAP. Please install shap.") from e

    model.eval()
    # SHAP requires gradients, so requires_grad must remain enabled

    # collect background from first batches
    bg_list = []
    total_bg = 0
    for xb, _ in loader:
        bg_list.append(xb)
        total_bg += xb.size(0)
        if total_bg >= background_max_samples:
            break
    if not bg_list:
        return {name: 0.0 for name in feature_names}
    background = torch.cat(bg_list, dim=0)[:background_max_samples].to(device)

    # build explainer with warning suppression
    tqdm.write(f"[INFO] Building DeepSHAP explainer with {background.size(0)} background samples...")
    
    # Temporarily suppress SHAP warnings (numerical precision warnings can usually be ignored)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        try:
            explainer = shap.DeepExplainer(model, background)
        except Exception as e:
            tqdm.write(f"[WARN] DeepSHAP explainer creation failed: {e}")
            # Retry with a smaller background set if explainer creation fails
            if background.size(0) > 50:
                tqdm.write(f"[INFO] Retrying with smaller background set (50 samples)...")
                background = background[:50]
                explainer = shap.DeepExplainer(model, background)
            else:
                raise

    num_features = len(feature_names)
    shap_sum = torch.zeros(num_features, dtype=torch.float64)
    count = 0

    progress_bar = tqdm(loader, desc="DeepSHAP Importance", leave=False)
    failed_batches = 0
    total_batches = 0
    
    for xb, _ in progress_bar:
        xb = xb.to(device)
        total_batches += 1
        
        # Suppress SHAP numerical precision warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            try:
                # shap returns a list if model outputs multiple tensors; for single output, returns np.array
                # SHAP may raise an exception when validation fails, although the computed values may still be useful
                # Try check_additivity=False to disable validation when supported
                try:
                    values = explainer.shap_values(xb, check_additivity=False)
                except TypeError:
                    # Use the default call if the argument is unsupported
                    values = explainer.shap_values(xb)
            except Exception as e:
                # SHAP numerical precision validation failed; relative importance rankings are usually unaffected
                failed_batches += 1
                # Print details only for the first failure and count subsequent failures
                if failed_batches == 1:
                    error_msg = str(e)
                    if "sum up to the model's output" in error_msg:
                        tqdm.write(f"[INFO] DeepSHAP: Ignoring numerical precision warnings (this is usually safe)")
                    else:
                        tqdm.write(f"[WARN] DeepSHAP computation failed for batch: {e}")
                continue
        
        # for classification with 2 classes, values is a list [class0, class1]
        if isinstance(values, list):
            if len(values) <= target_class:
                val = values[-1]
            else:
                val = values[target_class]
        else:
            val = values

        # val shape: [B, T, F]
        val = torch.tensor(val, dtype=torch.float32)
        if time_aggregation == "last":
            val = val[:, -1, :]
        else:
            val = val.mean(dim=1)

        # mean absolute contribution per feature
        abs_val = val.abs()  # [B, F]
        shap_sum += abs_val.sum(dim=0).to(dtype=shap_sum.dtype)
        count += abs_val.size(0)

    if count == 0:
        tqdm.write(f"[WARN] DeepSHAP: All batches failed, returning zero importance")
        return {name: 0.0 for name in feature_names}
    
    # Print statistics if any batches failed
    if failed_batches > 0:
        success_rate = (total_batches - failed_batches) / total_batches * 100
        tqdm.write(f"[INFO] DeepSHAP: {failed_batches}/{total_batches} batches had precision warnings, {success_rate:.1f}% succeeded")

    imp = shap_sum / float(count)
    total = float(imp.sum().item())
    if total <= 0:
        norm = [0.0 for _ in feature_names]
    else:
        norm = (imp / imp.sum()).tolist()

    return {name: float(val) for name, val in zip(feature_names, norm)}


def compute_integrated_gradients_importance(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    target_class: int = 1,
    steps: int = 50,
    time_aggregation: Literal["mean", "last"] = "mean",
) -> Dict[str, float]:
    """
    Compute Integrated Gradients importance for sequence inputs.
    
    Args:
        model: trained model
        loader: DataLoader yielding (X, y) with X shape [B, T, F]
        feature_names: list of feature column names
        device: device string
        target_class: which class logit to explain
        steps: number of integration steps
        time_aggregation: 'mean' over time or 'last' time step only
    
    Returns:
        dict mapping feature name -> normalized importance
    """
    model.eval()
    
    num_features = len(feature_names)
    integrated_grads_sum = torch.zeros(num_features, dtype=torch.float64, device=device)
    count = 0
    
    tqdm.write(f"[INFO] Computing Integrated Gradients with {steps} integration steps...")
    progress_bar = tqdm(loader, desc="Integrated Gradients", leave=False)
    for xb, _ in progress_bar:
        xb = xb.to(device)
        batch_size, seq_len, num_features = xb.shape
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(xb)
        
        # Compute integrated gradients
        integrated_grads = torch.zeros_like(xb)
        
        for i in range(steps + 1):
            alpha = i / steps
            interpolated = baseline + alpha * (xb - baseline)
            interpolated.requires_grad_(True)
            
            logits = model(interpolated)
            score = logits[:, target_class].sum()
            
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            model.zero_grad(set_to_none=True)
            score.backward()
            
            grad = interpolated.grad
            integrated_grads += grad / steps
        
        # Multiply by input difference
        integrated_grads *= (xb - baseline)
        
        # Aggregate over time
        if time_aggregation == "last":
            integrated_grads = integrated_grads[:, -1, :]
        else:
            integrated_grads = integrated_grads.mean(dim=1)
        
        # Sum over batch
        integrated_grads_sum += integrated_grads.sum(dim=0).to(device=integrated_grads_sum.device, dtype=integrated_grads_sum.dtype)
        count += integrated_grads.size(0)
    
    if count == 0:
        return {name: 0.0 for name in feature_names}
    
    imp = integrated_grads_sum / float(count)
    imp = imp.clamp(min=0)
    total = float(imp.sum().item())
    if total <= 0:
        norm = [0.0 for _ in feature_names]
    else:
        norm = (imp / imp.sum()).tolist()
    
    return {name: float(val) for name, val in zip(feature_names, norm)}


def compute_multiple_importance_nn(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: List[str],
    methods: List[str] = None,
    device: str = "cpu",
    target_class: int = 1,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute results for multiple importance methods for neural networks.
    
    Args:
        model: Trained neural network model.
        loader: DataLoader
        feature_names: List of feature names.
        methods: Importance methods, e.g., ['gradient', 'deepshap', 'integrated_gradients'].
        device: Computation device.
        target_class: Target class.
        **kwargs: Additional parameters.
    
    Returns:
        dict: {method: {feature_name: importance_score}}
    """
    if methods is None:
        methods = ['gradient']
    
    results = {}
    
    # Add a progress bar for importance computation
    method_iter = tqdm(methods, desc="Computing Importance", leave=False)
    for method in method_iter:
        method_iter.set_description(f"Computing {method} importance")
        
        if method == 'gradient':
            # The gradient method accepts only time_aggregation
            gradient_kwargs = {
                'time_aggregation': kwargs.get('time_aggregation', 'mean')
            }
            results[method] = compute_feature_importance(
                model, loader, feature_names, device=device, target_class=target_class, **gradient_kwargs
            )
        elif method == 'deepshap':
            # The deepshap method accepts background_max_samples and time_aggregation
            deepshap_kwargs = {
                'background_max_samples': kwargs.get('background_max_samples', 512),
                'time_aggregation': kwargs.get('time_aggregation', 'mean')
            }
            results[method] = compute_feature_importance_deepshap(
                model, loader, feature_names, device=device, target_class=target_class, **deepshap_kwargs
            )
        elif method == 'integrated_gradients':
            # The integrated_gradients method accepts steps and time_aggregation
            ig_kwargs = {
                'steps': kwargs.get('steps', 50),
                'time_aggregation': kwargs.get('time_aggregation', 'mean')
            }
            results[method] = compute_integrated_gradients_importance(
                model, loader, feature_names, device=device, target_class=target_class, **ig_kwargs
            )
        else:
            tqdm.write(f"[WARN] Unknown importance method: {method}")
        
        method_iter.set_description(f"Completed {method} importance")
    
    return results


def save_feature_importance(
    importance: Dict[str, float],
    out_dir: str,
    model_label: str,
    fold_idx: int,
    test_date_label: str,
) -> str:
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{model_label}_importance_fold_{fold_idx}_{test_date_label}.csv",
    )
    df = (
        pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(out_path, index=False)
    return out_path


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
    import pandas as pd
    import numpy as np
    import os

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


def save_multiple_importance_nn(
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
    Save neural network importance results from multiple methods to separate files.
    
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
    import pandas as pd
    import numpy as np
    import os
    
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
    Generate summary files for multiple importance methods for LSTM models.
    
    Args:
        output_dir: Output directory.
        model_label: Model label.
        methods: List of importance methods.
        fold_suffix_template: Fold-level filename suffix template (default: '_imp_{method}_fold').
        summary_suffix_template: Summary filename suffix template (default: '_imp_{method}_sum').
    """
    if methods is None:
        methods = ['gradient']
    
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


def generate_cross_method_importance_summary(
    output_dir: str,
    model_label: str,
    methods: List[str] = None,
    fold_suffix_template: str = '_imp_{method}_fold',
    cross_method_suffix: str = '_imp_sum'
) -> None:
    """
    Generate a neural network importance summary across methods.
    
    Args:
        output_dir: Output directory.
        model_label: Model label.
        methods: List of importance methods.
        fold_suffix_template: Fold-level filename suffix template.
        cross_method_suffix: Filename suffix for the summary across methods.
    """
    if methods is None:
        methods = ['gradient']
    
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
