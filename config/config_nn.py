# Neural Network Model Configuration
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

# Model label (for logs and outputs) - manually specified for neural network models
MODEL_LABEL = 'BiLSTM-V0.91'

# Explicit output filenames (no fallback) - dynamically updated based on MODEL_LABEL
FOLD_CSV_NAME = f"{MODEL_LABEL}_fold.csv"
SUM_CSV_NAME = f"{MODEL_LABEL}_sum.csv"
IMPORTANCE_CSV_NAME = f"{MODEL_LABEL}_imp_fold.csv"
IMPORTANCE_SUMMARY_CSV_NAME = f"{MODEL_LABEL}_imp_sum.csv"

# Warnings control
SUPPRESS_WARNINGS = True

# Output control
SAVE_FOLD_CSV = True    # save per-fold detailed metrics
SAVE_SUMMARY_CSV = True # save aggregated summary metrics

# ========= LOSS / OPTIMIZATION CONFIG =========

# LOSS_TYPE options:
# - 'ce': CrossEntropyLoss (default; for binary classification with two logits)
# - 'bce': BCEWithLogitsLoss (uses the class-1 logit); pos_weight can balance positive and negative samples
# - 'focal': FocalLoss (CE variant weighted by softmax pt); supports two-logit output
LOSS_TYPE = 'focal'

# BCEWithLogitsLoss: how to compute pos_weight
# - 'balance': compute automatically from the training set class ratio
# - None: do not use pos_weight
BCE_POS_WEIGHT_MODE = 'balance'

# FocalLoss parameters (used only when LOSS_TYPE='focal')
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0

# ========= SEQ MODEL TYPE CONFIG =========

# Options: 'fnn' | 'rnn' | 'lstm' | 'bilstm' | 'gru' | 'transformer' | 'cnn'
SEQ_MODEL_TYPE = 'bilstm'

# Attention configuration (for all sequence models)
ENABLE_ATTENTION = False
ATTENTION_TYPE = 'self'  # 'self' | 'cross' (reserved)

# ========= MODEL STRUCTURE CONFIG =========

# Classification head structure (architecture optimization)
# Control classification head complexity and model capacity
ENABLE_DEEP_HEAD = False         # True: multilayer MLP head; False: single Linear head (original default)
HEAD_HIDDEN_SIZE = 128           # Hidden dimension of the multilayer head (only when ENABLE_DEEP_HEAD=True)
HEAD_NUM_LAYERS = 2              # Number of head layers (only when ENABLE_DEEP_HEAD=True)
ENABLE_HEAD_BATCH_NORM = False   # Use Batch Normalization in the head (disabled by default, as originally)

# Sequence aggregation mode
# Control how features are extracted from LSTM output sequences for classification
# Options: 'last' | 'mean' | 'max' | 'concat_last_mean'
# - 'last': use only the final time step (original default)
# - 'mean': average all time steps
# - 'max': take the maximum over all time steps
# - 'concat_last_mean': concatenate the final time step and the mean for more information
SEQUENCE_AGGREGATION = 'concat_last_mean'    # First optimization: concatenate the final step and mean for richer information

# ========= TRAINING OPTIMIZATION CONFIG =========

# Early Stopping
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MODE = 'min'  # 'min' | 'max'

# Cosine Annealing Learning Rate Scheduler
ENABLE_COSINE_ANNEALING = False
COSINE_ANNEALING_T_MAX = 30  # Maximum number of epochs
COSINE_ANNEALING_ETA_MIN = 1e-5  # Minimum learning rate

# Gradient Clipping
ENABLE_GRADIENT_CLIPPING = False
GRADIENT_CLIPPING_MAX_NORM = 1.0

# Weight Decay
ENABLE_WEIGHT_DECAY = False
WEIGHT_DECAY_VALUE = 1e-5

# Learning Rate
LR = 1e-3  # Initial learning rate for Adam optimizer

# Training epochs (when using early stopping, this is the maximum)
MAX_EPOCHS = 30

# ======== FEATURE ENGINEERING ========

# Feature processing configuration switches
# Enable/disable different feature processing steps
ENABLE_SEGMENT_ID = True      # Generate segment_id for sequence modeling

ENABLE_LAYERED_FILL = True    # Apply layered missing value filling

ENABLE_LOG1P_TRANSFORM = True # Apply log1p transformation for long-tailed features
ENABLE_STANDARDIZATION = True # Apply feature standardization

# Feature standardization mode (only used when ENABLE_STANDARDIZATION = True)
# Options:
# - 'rolling': per-month rolling StandardScaler fit on window [i-T+1, i] then transform month i
# - 'global': fit one StandardScaler on all rows, transform all rows (may leak future info)
# - 'train_set': fit StandardScaler on training set only, then transform both train and test sets (v2 style)
STANDARDIZE_MODE = 'train_set'

# Training data accumulation mode (v1.5 style)
# Options:
# - False: Use fixed window size T for training (standard sliding window)
# - True: Use cumulative data from start to current month for training (v1.5 style)
# WARNING: Cumulative mode can cause memory issues and data leakage
USE_CUMULATIVE_TRAINING = False

# ======== FEATURE IMPORTANCE (Neural Network Model Specific) ========

# Feature importance computation
COMPUTE_IMPORTANCE = True
IMPORTANCE_METHODS = ['gradient']  # ['gradient', 'deepshap', 'integrated_gradients']
IMPORTANCE_TARGET_CLASS = 1  # Target class for importance calculation (0 or 1)

# Generate cross-method importance summary
GENERATE_CROSS_METHOD_SUMMARY = True  # Generate summary with all methods' mean importance
CROSS_METHOD_SUMMARY_SUFFIX = '_imp_sum'  # Suffix for cross-method summary file

# Importance file naming rules
IMPORTANCE_FOLD_SUFFIX = '_imp_{method}_fold'  # Template for fold-level importance files
IMPORTANCE_SUMMARY_SUFFIX = '_imp_{method}_sum'  # Template for summary importance files
# Examples: 
# - gradient: MODEL_LABEL_imp_gradient_fold.csv, MODEL_LABEL_imp_gradient_sum.csv
# - deepshap: MODEL_LABEL_imp_deepshap_fold.csv, MODEL_LABEL_imp_deepshap_sum.csv
# - integrated_gradients: MODEL_LABEL_imp_integrated_gradients_fold.csv, MODEL_LABEL_imp_integrated_gradients_sum.csv

# For DeepSHAP importance
DEEPSHAP_BACKGROUND_MAX_SAMPLES = 512
DEEPSHAP_TIME_AGGREGATION = 'mean'  # 'mean' or 'last'

# For Integrated Gradients importance
INTEGRATED_GRADIENTS_STEPS = 50
INTEGRATED_GRADIENTS_TIME_AGGREGATION = 'mean'  # 'mean' or 'last'

# For gradient-based importance
GRADIENT_TIME_AGGREGATION = 'mean'  # 'mean' or 'last'

# ======== ADVANCED FEATURE ENGINEERING ========

# Statistical features
ENABLE_STATISTICAL_FEATURES = False
STATISTICAL_FEATURES_CONFIG = {
    'rolling_stats': {
        'enabled': True,
        'windows': [3, 6, 12, 24],
        'functions': ['mean', 'std', 'min', 'max', 'median', 'skew']
    },
    'quantile_features': {
        'enabled': True,
        'windows': [6, 12, 24],
        'quantiles': [0.1, 0.25, 0.75, 0.9]
    },
    'rank_features': {
        'enabled': True,
        'windows': [6, 12, 24],
        'methods': ['percentile', 'zscore']
    },
    'change_features': {
        'enabled': True,
        'periods': [1, 3, 6, 12]
    },
    'volatility_features': {
        'enabled': True,
        'windows': [6, 12, 24]
    }
}

# Interaction features
ENABLE_INTERACTION_FEATURES = False
INTERACTION_FEATURES_CONFIG = {
    'ratio_features': {
        'enabled': True,
        'ratio_pairs': [
            ('roa', 'roe'),
            ('curr_ratio', 'quick_ratio'),
            ('debt_assets', 'debt_capital'),
            ('VOL', 'DOLLARVOL'),
            ('RET', 'VOL'),
            ('GDP_g', 'con_g'),
            ('INFLATION', 'GDP_g')
        ]
    },
    'polynomial_features': {
        'enabled': True,
        'degree': 2,
        'selected_features': ['price', 'VOL', 'RET', 'roa', 'roe', 'curr_ratio', 'debt_assets']
    },
    'cross_features': {
        'enabled': True,
        'feature_groups': [
            ['price', 'VOL', 'RET'],
            ['roa', 'roe', 'npm'],
            ['curr_ratio', 'quick_ratio', 'cash_ratio'],
            ['debt_assets', 'debt_capital', 'intcov'],
            ['GDP_g', 'con_g', 'ipt_g'],
            ['INFLATION', 'Unemployment']
        ]
    },
    'financial_ratio_features': {
        'enabled': True
    },
    'market_ratio_features': {
        'enabled': True
    },
    'macro_ratio_features': {
        'enabled': True
    }
}

# Technical indicators features
ENABLE_TECHNICAL_INDICATORS = False
TECHNICAL_INDICATORS_CONFIG = {
    'technical_indicators': {
        'enabled': True,
        'RSI': {'enabled': True, 'period': 14, 'features': ['price']},
        'MACD': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9, 'features': ['price']},
        'BOLLINGER': {'enabled': True, 'period': 20, 'std': 2, 'features': ['price']},
        'STOCHASTIC': {'enabled': True, 'k_period': 14, 'd_period': 3, 'features': ['price']},
        'WILLIAMS_R': {'enabled': True, 'period': 14, 'features': ['price']},
        'CCI': {'enabled': True, 'period': 20, 'features': ['price']}
    },
    'volume_indicators': {
        'enabled': True,
        'volume_features': ['VOL', 'DOLLARVOL']
    }
}

# Temporal features
ENABLE_TEMPORAL_FEATURES = False
TEMPORAL_FEATURES_CONFIG = {
    'cyclical_encoding': {
        'enabled': True
    },
    'lag_features': {
        'enabled': True,
        'lags': [1, 3, 6, 12]
    },
    'lead_features': {
        'enabled': False,  # Disabled by default to avoid data leakage
        'leads': [1, 3, 6]
    },
    'trend_features': {
        'enabled': True,
        'windows': [3, 6, 12]
    },
    'seasonal_features': {
        'enabled': True
    },
    'time_distance_features': {
        'enabled': True
    }
}


