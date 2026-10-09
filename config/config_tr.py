# Tree Model Configuration
import os

# Project root (infer FB_ML/)
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# Output root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')

# Private datasets are supplied locally and are never committed.
DATA_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FBML_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
))
FILES = [
    ("master_file.dta", "stata", {}),
    ("associative_memory.sas7bdat", "sas", {"public_date": "mdate"}),
    ("AP_factors.dta", "stata", {"time_avail_m": "mdate"}),
    ("stock_ret_controls.dta", "stata", {}),
]

# Window length
T = 36

# Task flags:
# ACTION: label source 'sell' or 'buy' (bin: y = (ACTION==1))
# TASK: 'bin' or 'tri' (y in {-1,0,1})
# NAN: True keep rows (fill NaN with 0); False drop rows with both NaN

ACTION = 'sell'
TASK = 'bin'
NAN = True

# Window control
# BATCH_MODE: If True, use START_WINDOW and END_WINDOW to control window range
#             If False, ignore START_WINDOW and END_WINDOW, run all windows
BATCH_MODE = False  # Set to True to enable batch processing mode

# If BATCH_MODE is True, these control the window range (inclusive [START_WINDOW, END_WINDOW])
# If BATCH_MODE is False, these are ignored and all windows will be processed
START_WINDOW = None  # e.g., 50
END_WINDOW = None    # e.g., 59

# Model label (for logs and outputs) - manually specified for tree models
MODEL_LABEL = 'LightGBM-b'

# Explicit output filenames (no fallback) - dynamically updated based on MODEL_LABEL
FOLD_CSV_NAME = f"{MODEL_LABEL}_fold.csv"
SUM_CSV_NAME = f"{MODEL_LABEL}_sum.csv"

# Warnings control
SUPPRESS_WARNINGS = True

# Output control
SAVE_FOLD_CSV = True    # save per-fold detailed metrics
SAVE_SUMMARY_CSV = True # save aggregated summary metrics

# ========= TREE MODEL TYPE =========

# Options: 'decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting', 
#          'hist_gradient_boosting', 'xgboost', 'lightgbm', 'catboost'
TREE_MODEL_TYPE = 'lightgbm'

# Balanced/weighted versions
USE_BALANCED_CLASS_WEIGHTS = True  # Use balanced/weighted versions for all tree models

# ========= HYPERPARAMETER OPTIMIZATION (Strategy 1: Global Optimization) =========
# Strategy 1: Optimize hyperparameters on representative windows, then print optimized config
# This significantly reduces time cost compared to optimizing on every window
# 
# When ENABLE_HYPERPARAMETER_OPTIMIZATION=True:
#   - The program will ONLY run hyperparameter optimization on representative windows
#   - After optimization, it will print the optimized parameters and exit
#   - NO full training will be performed
#   - You can then manually copy the optimized parameters to the config file
# 
# When ENABLE_HYPERPARAMETER_OPTIMIZATION=False:
#   - Normal training mode: use default or manually specified parameters
ENABLE_HYPERPARAMETER_OPTIMIZATION = False  # Set to True to enable Strategy 1 (optimization only mode)

# Hyperparameter optimization settings (only used if ENABLE_HYPERPARAMETER_OPTIMIZATION=True)
HPO_METHOD = 'optuna'  # 'optuna' or 'random' (RandomizedSearchCV)
HPO_N_TRIALS = 50  # Number of optimization trials (for Optuna) or iterations (for Random)
HPO_CV_FOLDS = 3  # Number of cross-validation folds (TimeSeriesSplit)
HPO_SCORING = 'f1'  # Scoring metric: 'roc_auc', 'pr_auc', 'f1', etc.
HPO_N_JOBS = -1  # Number of parallel jobs (-1 for all cores)
HPO_RANDOM_STATE = 42  # Random state for reproducibility

# Optuna-specific settings (only used if HPO_METHOD='optuna')
HPO_OPTUNA_ENABLE_PRUNING = True  # Enable early stopping/pruning
HPO_OPTUNA_PRUNER = 'median'  # 'median', 'percentile', 'successive_halving'
HPO_OPTUNA_N_STARTUP_TRIALS = 10  # Number of trials before pruning starts
HPO_OPTUNA_N_WARMUP_STEPS = 5  # Minimum steps before pruning

# Representative windows selection (only used if ENABLE_HYPERPARAMETER_OPTIMIZATION=True)
# Options:
#   - 'auto': Automatically select evenly distributed windows (default)
#   - 'manual': Use HPO_REPRESENTATIVE_WINDOW_INDICES to specify windows
#   - 'first_n': Use first N windows (specify HPO_REPRESENTATIVE_N_WINDOWS)
HPO_REPRESENTATIVE_SELECTION = 'manual'  # 'auto', 'manual', 'first_n'
HPO_REPRESENTATIVE_N_WINDOWS = 3  # Number of representative windows (for 'auto' or 'first_n')
HPO_REPRESENTATIVE_WINDOW_INDICES = [36,48,60]  # List of window indices for 'manual' mode, e.g., [10, 30, 50]

# Parameter stability validation (only used if ENABLE_HYPERPARAMETER_OPTIMIZATION=True)
HPO_VALIDATE_STABILITY = True  # Validate parameter stability across representative windows
HPO_STABILITY_CV_THRESHOLD = 0.1  # Coefficient of variation threshold for stability (0.1 = 10%)

# ======== BASIC FEATURE PROCESSING ========

# Feature processing configuration switches
# Enable/disable different feature processing steps
ENABLE_SEGMENT_ID = True     # Generate segment_id for layered_missing_fill
ENABLE_LAYERED_FILL = True    # Apply layered missing value filling

# Training mode
USE_CUMULATIVE_TRAINING = False  # Use cumulative training data or sliding window

# Feature importance computation (Tree Model Specific)
COMPUTE_IMPORTANCE = True
IMPORTANCE_METHODS = ['feature_importances']  # ['feature_importances', 'tree_specific', 'permutation', 'shap']
IMPORTANCE_TARGET_CLASS = 1  # Target class for importance calculation (0 or 1)

# Temporal importance aggregation
AGGREGATE_TEMPORAL_IMPORTANCE = True  # Aggregate importance across time steps
TEMPORAL_AGGREGATION_METHOD = 'mean'  # 'mean', 'max', or 'sum'

# Generate cross-method importance summary
GENERATE_CROSS_METHOD_SUMMARY = False  # Generate summary with all methods' mean importance
CROSS_METHOD_SUMMARY_SUFFIX = '_imp_sum'  # Suffix for cross-method summary file

# Importance file naming rules
IMPORTANCE_FOLD_SUFFIX = '_imp_{method}_fold'  # Template for fold-level importance files
IMPORTANCE_SUMMARY_SUFFIX = '_imp_{method}_sum'  # Template for summary importance files
# Examples: 
# - feature_importances: MODEL_LABEL_imp_feature_importances_fold.csv, MODEL_LABEL_imp_feature_importances_sum.csv
# - tree_specific: MODEL_LABEL_imp_tree_specific_fold.csv, MODEL_LABEL_imp_tree_specific_sum.csv
# - permutation: MODEL_LABEL_imp_permutation_fold.csv, MODEL_LABEL_imp_permutation_sum.csv
# - shap: MODEL_LABEL_imp_shap_fold.csv, MODEL_LABEL_imp_shap_sum.csv

# For permutation importance
PERMUTATION_IMPORTANCE_N_REPEATS = 5
PERMUTATION_IMPORTANCE_RANDOM_STATE = 42

# For SHAP importance (if using SHAP)
SHAP_BACKGROUND_SAMPLES = 100
SHAP_MAX_EVAL_SAMPLES = 1000

