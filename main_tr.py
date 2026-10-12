import os
import gc
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
from modules.data_loader import load_sources
from modules.feature_engineer import layered_missing_fill
from modules.preprocess import build_sequences_last_T, build_sequences_cumulative, generate_segment_id, unify_and_prepare
from modules.prepare import split_train_test_by_window, set_seed
from modules.results import display_fold_results, extract_and_store_metrics, summarize_kfold_results
from importance.importance_tr import (
    compute_multiple_importance_tree,
    save_multiple_importance_tree,
    generate_multiple_importance_summary,
    generate_cross_method_importance_summary,
    aggregate_temporal_importance,
)
from model.model_tr import (
    create_tree_model, 
    get_model_predictions, 
    train_tree_model,
    optimize_hyperparameters_optuna,
    optimize_hyperparameters_random,
    validate_parameter_stability,
    get_default_hyperparameter_space
)
import config.config_tr as C
import gc

T = C.T  


def select_representative_windows(total_windows, selection_mode='auto', n_windows=3, window_indices=None):
    """
    Select representative windows for hyperparameter optimization.
    
    Args:
        total_windows: Total number of windows.
        selection_mode: 'auto', 'manual', 'first_n'
        n_windows: Number of representative windows for 'auto' or 'first_n'.
        window_indices: Manually specified window indices for 'manual' mode.
    
    Returns:
        list: Representative window indices.
    """
    if selection_mode == 'manual':
        if window_indices is None:
            raise ValueError("window_indices must be provided when selection_mode='manual'")
        # Validate window indices
        valid_indices = [idx for idx in window_indices if 0 <= idx < total_windows]
        if len(valid_indices) != len(window_indices):
            tqdm.write(f"[WARN] Some window indices are invalid. Using valid indices: {valid_indices}")
        return valid_indices
    
    elif selection_mode == 'first_n':
        return list(range(min(n_windows, total_windows)))
    
    else:  # 'auto'
        # Select evenly distributed windows
        if n_windows >= total_windows:
            return list(range(total_windows))
        step = total_windows // (n_windows + 1)
        windows = [step * (i + 1) for i in range(n_windows)]
        return windows


def print_experiment_config():
    """Print the configuration parameters for this experiment."""
    print(f"\n=================== CONFIG ===================")
    
    # Model configuration
    print("\n[Model]")
    print(f"  MODEL_LABEL: {C.MODEL_LABEL}")
    print(f"  TREE_MODEL_TYPE: {C.TREE_MODEL_TYPE}")
    print(f"  USE_BALANCED_CLASS_WEIGHTS: {C.USE_BALANCED_CLASS_WEIGHTS}")
    
    # Hyperparameter optimization configuration
    print("\n[Hyperparameter Optimization]")
    enable_hpo = getattr(C, 'ENABLE_HYPERPARAMETER_OPTIMIZATION', False)
    print(f"  ENABLE_HYPERPARAMETER_OPTIMIZATION: {enable_hpo}")
    if enable_hpo:
        print(f"  Strategy: Global Optimization (Strategy 1)")
        print(f"  HPO_METHOD: {getattr(C, 'HPO_METHOD', 'optuna')}")
        print(f"  HPO_N_TRIALS: {getattr(C, 'HPO_N_TRIALS', 50)}")
        print(f"  HPO_CV_FOLDS: {getattr(C, 'HPO_CV_FOLDS', 3)}")
        print(f"  HPO_SCORING: {getattr(C, 'HPO_SCORING', 'roc_auc')}")
        print(f"  HPO_N_JOBS: {getattr(C, 'HPO_N_JOBS', -1)}")
        print(f"  HPO_REPRESENTATIVE_SELECTION: {getattr(C, 'HPO_REPRESENTATIVE_SELECTION', 'auto')}")
        if getattr(C, 'HPO_REPRESENTATIVE_SELECTION', 'auto') == 'auto':
            print(f"    - Number of representative windows: {getattr(C, 'HPO_REPRESENTATIVE_N_WINDOWS', 3)}")
        elif getattr(C, 'HPO_REPRESENTATIVE_SELECTION', 'auto') == 'manual':
            print(f"    - Representative window indices: {getattr(C, 'HPO_REPRESENTATIVE_WINDOW_INDICES', None)}")
        print(f"  HPO_VALIDATE_STABILITY: {getattr(C, 'HPO_VALIDATE_STABILITY', True)}")
        if getattr(C, 'HPO_METHOD', 'optuna') == 'optuna':
            print(f"  Optuna Settings:")
            print(f"    - Enable Pruning: {getattr(C, 'HPO_OPTUNA_ENABLE_PRUNING', True)}")
            if getattr(C, 'HPO_OPTUNA_ENABLE_PRUNING', True):
                print(f"    - Pruner: {getattr(C, 'HPO_OPTUNA_PRUNER', 'median')}")
                print(f"    - Startup Trials: {getattr(C, 'HPO_OPTUNA_N_STARTUP_TRIALS', 10)}")
                print(f"    - Warmup Steps: {getattr(C, 'HPO_OPTUNA_N_WARMUP_STEPS', 5)}")
    else:
        print(f"  Using default or manually specified parameters")
    
    # Data configuration
    print("\n[Data]")
    print(f"  T (Window Length): {C.T}")
    print(f"  ACTION: {C.ACTION}")
    print(f"  TASK: {C.TASK}")
    print(f"  NAN: {C.NAN}")
    print(f"  BATCH_MODE: {C.BATCH_MODE}")
    if C.BATCH_MODE:
        print(f"    - START_WINDOW: {C.START_WINDOW}, END_WINDOW: {C.END_WINDOW}")
    print(f"  USE_CUMULATIVE_TRAINING: {C.USE_CUMULATIVE_TRAINING}")
    
    # Feature engineering configuration
    print("\n[Features]")
    print(f"  ENABLE_SEGMENT_ID: {C.ENABLE_SEGMENT_ID}")
    print(f"  ENABLE_LAYERED_FILL: {C.ENABLE_LAYERED_FILL}")
    
    # Importance configuration
    print("\n[Importance]")
    print(f"  COMPUTE_IMPORTANCE: {C.COMPUTE_IMPORTANCE}")
    if C.COMPUTE_IMPORTANCE:
        print(f"  IMPORTANCE_METHODS: {C.IMPORTANCE_METHODS}")
        print(f"  IMPORTANCE_TARGET_CLASS: {C.IMPORTANCE_TARGET_CLASS}")
        print(f"  AGGREGATE_TEMPORAL_IMPORTANCE: {C.AGGREGATE_TEMPORAL_IMPORTANCE}")
        if C.AGGREGATE_TEMPORAL_IMPORTANCE:
            print(f"    - TEMPORAL_AGGREGATION_METHOD: {C.TEMPORAL_AGGREGATION_METHOD}")
        print(f"  GENERATE_CROSS_METHOD_SUMMARY: {C.GENERATE_CROSS_METHOD_SUMMARY}")
        if 'permutation' in C.IMPORTANCE_METHODS:
            print(f"    - PERMUTATION_N_REPEATS: {C.PERMUTATION_IMPORTANCE_N_REPEATS}")
            print(f"    - PERMUTATION_RANDOM_STATE: {C.PERMUTATION_IMPORTANCE_RANDOM_STATE}")
        if 'shap' in C.IMPORTANCE_METHODS:
            print(f"    - SHAP_BACKGROUND_SAMPLES: {C.SHAP_BACKGROUND_SAMPLES}")
            print(f"    - SHAP_MAX_EVAL_SAMPLES: {C.SHAP_MAX_EVAL_SAMPLES}")
    
    # Output configuration
    print("\n[Output]")
    print(f"  OUTPUT_DIR: {C.OUTPUT_DIR}")
    print(f"  SAVE_FOLD_CSV: {C.SAVE_FOLD_CSV}")
    print(f"  SAVE_SUMMARY_CSV: {C.SAVE_SUMMARY_CSV}")

def train_one_window_tr(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols,
                        T: int, fold_idx: int, test_date_label: str, 
                        best_params=None, is_hpo_mode=False):

    
    tqdm.write("[MAIN] Building training/testing sequences ...")
    
    # ===== CUMULATIVE TRAINING LOGIC (v1.5 style) =====
    if C.USE_CUMULATIVE_TRAINING:
        tqdm.write("[MAIN] Using cumulative training data (v1.5 style) ...")
        X_train, y_train = build_sequences_cumulative(train_df, feature_cols, T)
    else:
        X_train, y_train = build_sequences_last_T(train_df, feature_cols, T)
    
    X_test, y_test = build_sequences_last_T(test_df, feature_cols, T)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    tqdm.write("[MAIN] Preparing data for tree model ...")
    
    # Flatten sequence data into a 2D matrix for tree models
    # X_train shape: (n_samples, T, n_features) -> (n_samples, T*n_features)
    n_train_samples, n_timesteps, n_features = X_train.shape
    n_test_samples = X_test.shape[0]
    
    # Flatten sequence data
    X_train_flat = X_train.reshape(n_train_samples, n_timesteps * n_features)
    X_test_flat = X_test.reshape(n_test_samples, n_timesteps * n_features)
    
    # Create feature names including time-step information
    feature_names_flat = []
    for t in range(n_timesteps):
        for feat in feature_cols:
            feature_names_flat.append(f"{feat}_t{t}")
    
    tqdm.write(f"[MAIN] Training {C.TREE_MODEL_TYPE} model ...")
    
    # Create and train the model
    # Use the supplied best parameters in hyperparameter optimization mode
    if best_params is not None:
        tqdm.write(f"[MAIN] Using optimized hyperparameters: {best_params}")
        model = create_tree_model(
            C.TREE_MODEL_TYPE, 
            use_balanced=C.USE_BALANCED_CLASS_WEIGHTS,
            **best_params
        )
    else:
        model = create_tree_model(
            C.TREE_MODEL_TYPE, 
            use_balanced=C.USE_BALANCED_CLASS_WEIGHTS
        )
    
    train_tree_model(
        model, 
        X_train_flat, 
        y_train, 
        C.TREE_MODEL_TYPE, 
        C.USE_BALANCED_CLASS_WEIGHTS
    )
    
    tqdm.write("[MAIN] Evaluating ...")
    
    # Predict
    y_pred_class, y_pred_cont = get_model_predictions(model, X_test_flat, C.TREE_MODEL_TYPE)
    
    # Ensure y_test is a NumPy array
    y_true = np.array(y_test)

    tqdm.write("[MAIN] Displaying results ...")
    display_fold_results(C.MODEL_LABEL, fold_idx, test_date_label, y_true, y_pred_class, y_pred_cont)
    metrics_batch = []
    extract_and_store_metrics(y_true, y_pred_class, y_pred_cont, metrics_batch, fold_idx)
    
    if C.SAVE_FOLD_CSV:
        try:
            out_dir = C.OUTPUT_DIR
            os.makedirs(out_dir, exist_ok=True)
            if C.FOLD_CSV_NAME is None:
                raise ValueError("FOLD_CSV_NAME must be set in config.py")
            
            # Append the window suffix in batch mode
            if C.BATCH_MODE:
                if C.START_WINDOW is not None and C.END_WINDOW is not None:
                    base_name = C.FOLD_CSV_NAME.replace('.csv', '')
                    fold_csv = os.path.join(out_dir, f"{base_name}_{C.START_WINDOW}-{C.END_WINDOW}.csv")
                else:
                    fold_csv = os.path.join(out_dir, C.FOLD_CSV_NAME)
            else:
                # Use no window suffix outside batch mode
                fold_csv = os.path.join(out_dir, C.FOLD_CSV_NAME)
            
            df_mb = pd.DataFrame(metrics_batch)
            write_header = not os.path.exists(fold_csv)
            df_mb.to_csv(fold_csv, mode='a', header=write_header, index=False)
            tqdm.write(f"\n[MAIN] Fold metrics saved to: {fold_csv}")
        except Exception as e:
            tqdm.write(f"\n[WARN] Failed to write fold metrics: {e}")

    # ========== FEATURE IMPORTANCE ==========
    try:
        if C.COMPUTE_IMPORTANCE:
            tqdm.write(f"[MAIN] Computing feature importance ({', '.join(C.IMPORTANCE_METHODS)}) ...")
            
            # Compute multiple importance methods
            importance_results = compute_multiple_importance_tree(
                model=model,
                X_test=X_test_flat,
                feature_names=feature_names_flat,
                methods=C.IMPORTANCE_METHODS,
                target_class=C.IMPORTANCE_TARGET_CLASS,
                n_repeats=C.PERMUTATION_IMPORTANCE_N_REPEATS,
                random_state=C.PERMUTATION_IMPORTANCE_RANDOM_STATE,
                background_samples=C.SHAP_BACKGROUND_SAMPLES,
                max_eval_samples=C.SHAP_MAX_EVAL_SAMPLES
            )
            
            # Aggregate importance across time steps if enabled
            if C.AGGREGATE_TEMPORAL_IMPORTANCE:
                tqdm.write(f"[INFO] Aggregating temporal importance using {C.TEMPORAL_AGGREGATION_METHOD} method...")
                aggregated_results = aggregate_temporal_importance(
                    importance_results, 
                    aggregation_method=C.TEMPORAL_AGGREGATION_METHOD
                )
                
                # Check whether aggregated results are empty
                if not aggregated_results:
                    tqdm.write(f"[WARN] Aggregated importance results are empty, skipping save")
                elif not any(aggregated_results.values()):
                    tqdm.write(f"[WARN] Aggregated importance results have no features, skipping save")
                else:
                    # Save only aggregated importance without time-step identifiers
                    # Append the window suffix to filenames in batch mode
                    window_suffix = ""
                    if C.BATCH_MODE:
                        if C.START_WINDOW is not None and C.END_WINDOW is not None:
                            window_suffix = f"_{C.START_WINDOW}-{C.END_WINDOW}"
                    
                    # Get feature names from the first method's results
                    first_method = list(aggregated_results.keys())[0]
                    feature_names_aggregated = list(aggregated_results[first_method].keys())
                    
                    save_multiple_importance_tree(
                        importance_results=aggregated_results,
                        feature_names=feature_names_aggregated,
                        fold_idx=fold_idx,
                        output_dir=C.OUTPUT_DIR,
                        model_label=C.MODEL_LABEL,
                        fold_suffix_template=C.IMPORTANCE_FOLD_SUFFIX,
                        summary_suffix_template=C.IMPORTANCE_SUMMARY_SUFFIX,
                        window_suffix=window_suffix  # Pass the window suffix
                    )
                    tqdm.write(f"[INFO] Saved aggregated importance")
            else:
                # Save original importance with time-step identifiers
                # Append the window suffix to filenames in batch mode
                window_suffix = ""
                if C.BATCH_MODE:
                    if C.START_WINDOW is not None and C.END_WINDOW is not None:
                        window_suffix = f"_{C.START_WINDOW}-{C.END_WINDOW}"
                
                save_multiple_importance_tree(
                    importance_results=importance_results,
                    feature_names=feature_names_flat,
                    fold_idx=fold_idx,
                    output_dir=C.OUTPUT_DIR,
                    model_label=C.MODEL_LABEL,
                    fold_suffix_template=C.IMPORTANCE_FOLD_SUFFIX,
                    summary_suffix_template=C.IMPORTANCE_SUMMARY_SUFFIX,
                    window_suffix=window_suffix  # Pass the window suffix
                )
                tqdm.write(f"[INFO] Saved original importance (with temporal suffixes)")
    except Exception as e:
        tqdm.write(f"[WARN] Importance computation failed: {e}")
    
    # [MEM]
    del X_train, y_train, X_test, y_test
    del X_train_flat, X_test_flat
    del model
    del y_pred_cont, y_pred_class, y_true
    gc.collect()

    # return fold metrics for summary
    return metrics_batch


def main() -> None:

    
    # warnings control
    if C.SUPPRESS_WARNINGS:
        warnings.filterwarnings('ignore')

    # ========== CONFIG ==========
    print_experiment_config()

    # ========== DATA ==========
    print(f"\n==================== DATA ====================\n")
    df_raw = load_sources(C.DATA_ROOT, C.FILES)
    
    # ========== PREP ==========
    print(f"\n==================== PREP ====================")
    print(f"\n--------- TARGET ---------\n")
    df_raw, feature_cols = unify_and_prepare(
        df_raw,
        action=C.ACTION,
        include_nan=C.NAN,
        task=C.TASK,
    )
    # ========== SEGMENT ID GENERATION ==========
    if C.ENABLE_SEGMENT_ID:
        print(f"\n--------- SEGMENT ---------\n")
        df_ = generate_segment_id(df_raw)
        print(f"[MAIN] Generated segment_id for layered_missing_fill")
    else:
        print(f"\n--------- SEGMENT ---------\n")
        print("[MAIN] Skipping segment_id generation (ENABLE_SEGMENT_ID = False)")
        df_ = df_raw.copy()
    
    # [MEM]
    del df_raw
    gc.collect()

    print(f"\n==================== FE ====================")
    
    # ========== LAYERED MISSING FILL ==========
    if C.ENABLE_LAYERED_FILL:
        print(f"\n--------- FILL ---------\n")
        df_ = layered_missing_fill(df_, feature_cols, T)
    else:
        print(f"\n--------- FILL ---------\n")
        print("[MAIN] Skipping layered missing fill (ENABLE_LAYERED_FILL = False)")
    
    gc.collect()

    # ========== MAIN ==========
    print(f"\n==================== MAIN ====================")
    all_dates = sorted(df_['mdate'].unique())

    total_windows = len(all_dates) - 1
    
    # Use BATCH_MODE to decide whether START_WINDOW and END_WINDOW apply
    if C.BATCH_MODE:
        # Batch mode: control the window range with START_WINDOW and END_WINDOW
        if C.START_WINDOW is not None and C.END_WINDOW is not None:
            start_window = C.START_WINDOW
            end_window = min(C.END_WINDOW, total_windows)
            # Require end_window < total_windows so all_dates[i+1] stays within bounds
            assert 0 <= start_window <= end_window < total_windows, f"Invalid window range. end_window ({end_window}) must be < total_windows ({total_windows}). Max valid end_window is {total_windows - 1}."
            window_iter = range(start_window, end_window + 1)
            total_iters = end_window - start_window + 1
            print(f"[MAIN] BATCH_MODE: ON | Window Range: {start_window} ~ {end_window} / total={total_windows}")
        else:
            # Warn and use all windows if BATCH_MODE is enabled without a window range
            print(f"[WARN] BATCH_MODE is True but START_WINDOW/END_WINDOW are not set. Running all windows.")
            window_iter = range(total_windows)
            start_window = 0
            end_window = total_windows - 1
            total_iters = total_windows
    else:
        # Full mode: completely ignore START_WINDOW and END_WINDOW and run all windows
        # Equivalent to START_WINDOW = None and END_WINDOW = None
        window_iter = range(total_windows)  # Run all windows: 0 through total_windows-1
        start_window = 0
        end_window = total_windows - 1
        total_iters = total_windows
        print(f"[MAIN] BATCH_MODE: OFF | Full Window Range: 0 ~ {total_windows - 1} (all {total_windows} windows, START_WINDOW/END_WINDOW completely ignored)")

    # ========== HYPERPARAMETER OPTIMIZATION (Strategy 1) ==========
    enable_hpo = getattr(C, 'ENABLE_HYPERPARAMETER_OPTIMIZATION', False)
    
    if enable_hpo:
        print(f"\n==================== HYPERPARAMETER OPTIMIZATION MODE ====================")
        print(f"[HPO] Strategy 1: Global Optimization Mode")
        print(f"[HPO] This mode will ONLY run hyperparameter optimization on representative windows")
        print(f"[HPO] After optimization, optimized parameters will be printed and the program will exit")
        print(f"[HPO] NO full training will be performed in this mode")
        
        # Select representative windows
        selection_mode = getattr(C, 'HPO_REPRESENTATIVE_SELECTION', 'auto')
        n_windows = getattr(C, 'HPO_REPRESENTATIVE_N_WINDOWS', 3)
        window_indices = getattr(C, 'HPO_REPRESENTATIVE_WINDOW_INDICES', None)
        
        rep_windows = select_representative_windows(
            total_windows, 
            selection_mode=selection_mode,
            n_windows=n_windows,
            window_indices=window_indices
        )
        
        print(f"[HPO] Selected {len(rep_windows)} representative windows: {rep_windows}")
        
        # Optimize hyperparameters on each representative window
        hpo_method = getattr(C, 'HPO_METHOD', 'optuna')
        n_trials = getattr(C, 'HPO_N_TRIALS', 50)
        cv_folds = getattr(C, 'HPO_CV_FOLDS', 3)
        scoring = getattr(C, 'HPO_SCORING', 'roc_auc')
        n_jobs = getattr(C, 'HPO_N_JOBS', -1)
        random_state = getattr(C, 'HPO_RANDOM_STATE', 42)
        
        param_results = []
        param_scores = []
        
        for rep_idx, rep_window in enumerate(rep_windows):
            print(f"\n[HPO] Optimizing on representative window {rep_idx + 1}/{len(rep_windows)}: Window {rep_window}")
            set_seed(rep_window)
            test_date_label = str(all_dates[rep_window + 1])
            train_df, test_df = split_train_test_by_window(df_, all_dates, rep_window, T)
            
            # Prepare training data
            if C.USE_CUMULATIVE_TRAINING:
                X_train, y_train = build_sequences_cumulative(train_df, feature_cols, T)
            else:
                X_train, y_train = build_sequences_last_T(train_df, feature_cols, T)
            
            if len(X_train) == 0:
                print(f"[HPO] Warning: Window {rep_window} has no training data, skipping")
                del train_df, test_df
                gc.collect()
                continue
            
            # Flatten sequence data
            n_train_samples, n_timesteps, n_features = X_train.shape
            X_train_flat = X_train.reshape(n_train_samples, n_timesteps * n_features)
            
            # Run hyperparameter optimization
            try:
                if hpo_method == 'optuna':
                    enable_pruning = getattr(C, 'HPO_OPTUNA_ENABLE_PRUNING', True)
                    pruner_type = getattr(C, 'HPO_OPTUNA_PRUNER', 'median')
                    n_startup_trials = getattr(C, 'HPO_OPTUNA_N_STARTUP_TRIALS', 10)
                    n_warmup_steps = getattr(C, 'HPO_OPTUNA_N_WARMUP_STEPS', 5)
                    
                    best_model, best_params_window, best_score = optimize_hyperparameters_optuna(
                        model_type=C.TREE_MODEL_TYPE,
                        X_train=X_train_flat,
                        y_train=y_train,
                        n_trials=n_trials,
                        cv_folds=cv_folds,
                        scoring=scoring,
                        random_state=random_state,
                        use_balanced=C.USE_BALANCED_CLASS_WEIGHTS,
                        n_jobs=n_jobs,
                        enable_pruning=enable_pruning,
                        pruner_type=pruner_type,
                        n_startup_trials=n_startup_trials,
                        n_warmup_steps=n_warmup_steps
                    )
                else:  # 'random'
                    best_model, best_params_window, best_score = optimize_hyperparameters_random(
                        model_type=C.TREE_MODEL_TYPE,
                        X_train=X_train_flat,
                        y_train=y_train,
                        n_iter=n_trials,
                        cv_folds=cv_folds,
                        scoring=scoring,
                        random_state=random_state,
                        use_balanced=C.USE_BALANCED_CLASS_WEIGHTS,
                        n_jobs=n_jobs
                    )
                
                print(f"[HPO] Window {rep_window} - Best {scoring}: {best_score:.4f}")
                print(f"[HPO] Window {rep_window} - Best params: {best_params_window}")
                
                param_results.append(best_params_window)
                param_scores.append(best_score)
                
            except Exception as e:
                print(f"[HPO] Error optimizing window {rep_window}: {e}")
                import traceback
                traceback.print_exc()
            
            # [MEM]
            del X_train, y_train, X_train_flat, train_df, test_df
            gc.collect()
        
        # Validate parameter stability and determine the final parameters
        if param_results:
            validate_stability = getattr(C, 'HPO_VALIDATE_STABILITY', True)
            is_stable = True
            stability_scores = {}
            
            if validate_stability and len(param_results) > 1:
                cv_threshold = getattr(C, 'HPO_STABILITY_CV_THRESHOLD', 0.1)
                is_stable, stability_scores = validate_parameter_stability(param_results, cv_threshold)
                
                print(f"\n[HPO] Parameter Stability Validation:")
                print(f"  Stable: {is_stable}")
                if stability_scores:
                    print(f"  Stability scores (CV):")
                    for param, cv in stability_scores.items():
                        print(f"    {param}: {cv:.4f}")
                
                if is_stable:
                    # Average numeric parameters and use the most frequent categorical values
                    print(f"[HPO] Parameters are stable, using averaged parameters")
                    best_params = {}
                    for param_name in param_results[0].keys():
                        values = [p[param_name] for p in param_results if param_name in p]
                        if all(isinstance(v, (int, float)) and v is not None for v in values):
                            # Numeric parameters: use the mean
                            best_params[param_name] = int(np.mean(values))
                        elif all(v is None for v in values):
                            best_params[param_name] = None
                        else:
                            # Categorical parameters: use the most frequent value
                            from collections import Counter
                            best_params[param_name] = Counter(values).most_common(1)[0][0]
                else:
                    # Use parameters from the best-performing window
                    best_idx = np.argmax(param_scores)
                    best_params = param_results[best_idx]
                    print(f"[HPO] Parameters are not stable, using best-performing window ({rep_windows[best_idx]}) parameters")
            else:
                # Use the result directly when there is only one result or stability validation is disabled
                best_params = param_results[0]
                print(f"[HPO] Using parameters from single representative window")
            
            print(f"\n[HPO] Final optimized parameters: {best_params}")
            
            # Print the optimized configuration for copying into the configuration file
            print(f"\n{'='*70}")
            print(f"[HPO] OPTIMIZED HYPERPARAMETER CONFIGURATION")
            print(f"{'='*70}")
            print(f"\n# Copy the following parameters to your config file:")
            print(f"# (Uncomment and set ENABLE_HYPERPARAMETER_OPTIMIZATION=False)")
            print(f"\n# Optimized hyperparameters for {C.TREE_MODEL_TYPE.upper()}:")
            
            if C.TREE_MODEL_TYPE == 'random_forest':
                print(f"RF_N_ESTIMATORS = {best_params.get('n_estimators', 100)}")
                max_depth_val = best_params.get('max_depth', None)
                print(f"RF_MAX_DEPTH = {max_depth_val if max_depth_val is not None else 'None'}")
                print(f"RF_MIN_SAMPLES_SPLIT = {best_params.get('min_samples_split', 2)}")
                print(f"RF_MIN_SAMPLES_LEAF = {best_params.get('min_samples_leaf', 1)}")
                max_features_val = best_params.get('max_features', 'sqrt')
                if isinstance(max_features_val, str):
                    print(f"RF_MAX_FEATURES = '{max_features_val}'")
                else:
                    print(f"RF_MAX_FEATURES = {max_features_val}")
                print(f"RF_BOOTSTRAP = {best_params.get('bootstrap', True)}")
                max_samples_val = best_params.get('max_samples', None)
                print(f"RF_MAX_SAMPLES = {max_samples_val if max_samples_val is not None else 'None'}")
                print(f"RF_CRITERION = '{best_params.get('criterion', 'gini')}'")
                print(f"RF_RANDOM_STATE = {best_params.get('random_state', 42)}")
                print(f"RF_N_JOBS = {best_params.get('n_jobs', -1)}")
            
            print(f"\n# Optimization Summary:")
            print(f"#   - Method: {hpo_method}")
            print(f"#   - Representative windows: {rep_windows}")
            print(f"#   - Best scores: {[f'{s:.4f}' for s in param_scores]}")
            if validate_stability and len(param_results) > 1:
                print(f"#   - Parameter stability: {is_stable}")
                if stability_scores:
                    print(f"#   - Stability scores (CV): {stability_scores}")
            
            print(f"\n{'='*70}")
            print(f"[HPO] Hyperparameter optimization completed!")
            print(f"[HPO] Please copy the optimized parameters above to your config file.")
            print(f"[HPO] Then set ENABLE_HYPERPARAMETER_OPTIMIZATION=False to run training.")
            print(f"{'='*70}\n")
            
            # Optimization mode: return after optimization without running full training
            return None
        else:
            print(f"[HPO] Warning: No successful optimization results")
            print(f"[HPO] Please check the error messages above and try again.")
            return None
    
    # ========== MAIN TRAINING LOOP ==========
    # Run full training only outside hyperparameter optimization mode
    print(f"\n==================== MAIN TRAINING ====================")
    print(f"[MAIN] {C.TREE_MODEL_TYPE.upper()} | Starting sliding-window training ...")
    all_results = []  # collect all fold results for summary
    p_windows = tqdm(total=total_iters, desc="Windows", leave=True)
    
    for i in window_iter:
        set_seed(i)
        test_date_label = str(all_dates[i+1])
        tqdm.write(f"\n---------- [ FOLD {i} | {test_date_label} ] ----------\n")
        train_df, test_df = split_train_test_by_window(df_, all_dates, i, T)
        fold_metrics = train_one_window_tr(
            train_df, test_df, feature_cols, T, i, test_date_label,
            best_params=None,  # Normal training uses manually configured parameters rather than optimization results
            is_hpo_mode=False
        )
        if fold_metrics:
            all_results.extend(fold_metrics)
        # [MEM]
        del train_df, test_df
        gc.collect()
        p_windows.update(1)
    p_windows.close()
    
    # Generate and save the summary only outside batch mode
    if all_results and C.SAVE_SUMMARY_CSV and not C.BATCH_MODE:
        summary_df = summarize_kfold_results(all_results, C.MODEL_LABEL)
        summary_path = os.path.join(C.OUTPUT_DIR, C.SUM_CSV_NAME)
        summary_df.to_csv(summary_path, index=False)
        tqdm.write(f"\n[MAIN] Summary saved to: {summary_path}")
    elif C.BATCH_MODE:
        tqdm.write(f"\n[INFO] BATCH_MODE: Summary files will be generated later using summary.py")
    
    # Generate and save the importance summary only outside batch mode
    if C.COMPUTE_IMPORTANCE and not C.BATCH_MODE:
        try:
            tqdm.write(f"\n[MAIN] Generating importance summaries...")
            generate_multiple_importance_summary(
                output_dir=C.OUTPUT_DIR,
                model_label=C.MODEL_LABEL,
                methods=C.IMPORTANCE_METHODS,
                fold_suffix_template=C.IMPORTANCE_FOLD_SUFFIX,
                summary_suffix_template=C.IMPORTANCE_SUMMARY_SUFFIX
            )
            tqdm.write(f"[MAIN] Importance summaries generated successfully")
        except Exception as e:
            tqdm.write(f"\n[WARN] Failed to generate importance summary: {e}")
    elif C.COMPUTE_IMPORTANCE and C.BATCH_MODE:
        tqdm.write(f"\n[INFO] BATCH_MODE: Importance summary files will be generated later using summary.py")
    
    # Generate the importance summary across methods only outside batch mode
    if C.COMPUTE_IMPORTANCE and C.GENERATE_CROSS_METHOD_SUMMARY and not C.BATCH_MODE:
        try:
            tqdm.write(f"[MAIN] Generating cross-method importance summary...")
            generate_cross_method_importance_summary(
                output_dir=C.OUTPUT_DIR,
                model_label=C.MODEL_LABEL,
                methods=C.IMPORTANCE_METHODS,
                fold_suffix_template=C.IMPORTANCE_FOLD_SUFFIX,
                cross_method_suffix=C.CROSS_METHOD_SUMMARY_SUFFIX
            )
            tqdm.write(f"[MAIN] Cross-method importance summary generated successfully")
        except Exception as e:
            tqdm.write(f"\n[WARN] Failed to generate cross-method importance summary: {e}")
    
    return None


if __name__ == '__main__':
    main()    
    print(f"\n[MAIN] {C.MODEL_LABEL} - {C.TREE_MODEL_TYPE.upper()} model DONE\n")
