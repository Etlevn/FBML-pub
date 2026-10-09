# Linear Model Configuration
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
BATCH_MODE = True # Set to True to enable batch processing mode

# If BATCH_MODE is True, these control the window range (inclusive [START_WINDOW, END_WINDOW])
# If BATCH_MODE is False, these are ignored and all windows will be processed
START_WINDOW = 0  # e.g., 0
END_WINDOW = 34    # e.g., 67

# Model label (for logs and outputs) - manually specified for linear models
MODEL_LABEL = 'SGD-Modified_Huber-b'

# Explicit output filenames (no fallback) - dynamically updated based on MODEL_LABEL
FOLD_CSV_NAME = f"{MODEL_LABEL}_fold.csv"
SUM_CSV_NAME = f"{MODEL_LABEL}_sum.csv"

# Warnings control
SUPPRESS_WARNINGS = True

# Output control
SAVE_FOLD_CSV = True    # save per-fold detailed metrics
SAVE_SUMMARY_CSV = True # save aggregated summary metrics

# ========= LINEAR MODEL TYPE =========

# Options: 'ols', 'ridge', 'logistic', 'sgd_hinge', 'sgd_log', 'sgd_modified_huber'
LINEAR_MODEL_TYPE = 'sgd_modified_huber'

# Balanced class weights for classification models
USE_BALANCED_CLASS_WEIGHTS = True

# ======== BASIC FEATURE PROCESSING ========

# Feature processing configuration switches
# Enable/disable different feature processing steps
ENABLE_SEGMENT_ID = True     # Generate segment_id for layered_missing_fill
ENABLE_LAYERED_FILL = True    # Apply layered missing value filling

# Training mode
USE_CUMULATIVE_TRAINING = False  # Use cumulative training data or sliding window

# Feature importance computation (Linear Model Specific)
COMPUTE_IMPORTANCE = True
IMPORTANCE_METHODS = ['coefficient']  # ['coefficient', 'permutation', 'shap']
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
# - coefficient: MODEL_LABEL_imp_coefficient_fold.csv, MODEL_LABEL_imp_coefficient_sum.csv
# - permutation: MODEL_LABEL_imp_permutation_fold.csv, MODEL_LABEL_imp_permutation_sum.csv
# - shap: MODEL_LABEL_imp_shap_fold.csv, MODEL_LABEL_imp_shap_sum.csv

# For permutation importance
PERMUTATION_IMPORTANCE_N_REPEATS = 5
PERMUTATION_IMPORTANCE_RANDOM_STATE = 42

# For SHAP importance (if using SHAP)
SHAP_BACKGROUND_SAMPLES = 100
SHAP_MAX_EVAL_SAMPLES = 1000
