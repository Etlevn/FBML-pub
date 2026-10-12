import os
import gc
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings

# Suppress multiprocessing semaphore leak warnings on macOS
if 'PYTHONWARNINGS' not in os.environ:
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    
from modules.data_loader import load_sources
from modules.feature_engineer import layered_missing_fill
from modules.preprocess import build_sequences_last_T, build_sequences_cumulative, generate_segment_id, unify_and_prepare
from modules.prepare import split_train_test_by_window, set_seed
from modules.results import display_fold_results, extract_and_store_metrics, summarize_kfold_results
from importance.importance_lin import (
    compute_multiple_importance_linear,
    save_multiple_importance_linear,
    generate_multiple_importance_summary,
    generate_cross_method_importance_summary,
    aggregate_temporal_importance,
)
from model.model_lin import create_linear_model, get_model_predictions, train_linear_model
import config.config_lin as C
import gc

T = C.T  


def print_experiment_config():
    """Print the configuration parameters for this experiment."""
    print(f"\n=================== CONFIG ===================")
    
    # Model configuration
    print("\n[Model]")
    print(f"  MODEL_LABEL: {C.MODEL_LABEL}")
    print(f"  LINEAR_MODEL_TYPE: {C.LINEAR_MODEL_TYPE}")
    print(f"  USE_BALANCED_CLASS_WEIGHTS: {C.USE_BALANCED_CLASS_WEIGHTS}")
    
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


def train_one_window_lin(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols,
                        T: int, fold_idx: int, test_date_label: str):

    
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

    tqdm.write("[MAIN] Preparing data for linear model ...")
    
    # Flatten sequence data into a 2D matrix for linear models
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
    
    tqdm.write(f"[MAIN] Training {C.LINEAR_MODEL_TYPE} model ...")
    
    # Create and train the model
    model = create_linear_model(C.LINEAR_MODEL_TYPE, use_balanced=C.USE_BALANCED_CLASS_WEIGHTS)
    train_linear_model(model, X_train_flat, y_train, C.LINEAR_MODEL_TYPE, C.USE_BALANCED_CLASS_WEIGHTS)
    
    tqdm.write("[MAIN] Evaluating ...")
    
    # Predict
    y_pred_class, y_pred_cont = get_model_predictions(model, X_test_flat, C.LINEAR_MODEL_TYPE)
    
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
            importance_results = compute_multiple_importance_linear(
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
                    
                    save_multiple_importance_linear(
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
                
                save_multiple_importance_linear(
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
        # Full mode: ignore START_WINDOW and END_WINDOW and run all windows
        window_iter = range(total_windows)
        start_window = 0
        end_window = total_windows - 1
        total_iters = total_windows
        print(f"[MAIN] BATCH_MODE: OFF | Full Window Range: 0 ~ {total_windows - 1} (START_WINDOW/END_WINDOW ignored)")

    print(f"[MAIN] {C.LINEAR_MODEL_TYPE.upper()} | Starting sliding-window training ...")
    all_results = []  # collect all fold results for summary
    p_windows = tqdm(total=total_iters, desc="Windows", leave=True)
    
    for i in window_iter:
        set_seed(i)
        test_date_label = str(all_dates[i+1])
        tqdm.write(f"\n---------- [ FOLD {i} | {test_date_label} ] ----------\n")
        train_df, test_df = split_train_test_by_window(df_, all_dates, i, T)
        fold_metrics = train_one_window_lin(train_df, test_df, feature_cols, T, i, test_date_label)
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
    print(f"\n[MAIN] {C.MODEL_LABEL} - {C.LINEAR_MODEL_TYPE.upper()} model DONE\n")

