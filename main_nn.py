import os
import gc
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
from modules.data_loader import load_sources
from modules.feature_engineer import layered_missing_fill, rolling_standardize_by_month, long_tail_log1p_transform, global_standardize
from modules.preprocess import build_sequences_last_T, build_sequences_cumulative, generate_segment_id, unify_and_prepare
from modules.prepare import split_train_test_by_window, set_seed
from model.model_nn import (
    FNNClassifier, RNNClassifier, LSTMClassifier, BiLSTMClassifier, GRUClassifier, TransformerClassifier, CNNClassifier,
    TRAIN_EPOCHS, TRAIN_BATCH, EVAL_BATCH, LR, SequenceDataset,
    TRANSFORMER_NUM_HEADS, TRANSFORMER_MAX_LEN,
    CNN_NUM_FILTERS, CNN_KERNEL_SIZE, CNN_NUM_CONV_LAYERS, CNN_POOL_SIZE, CNN_POOL_MODE,
    FNN_FLATTEN_MODE
)
from modules.results import display_fold_results, extract_and_store_metrics, summarize_kfold_results
from modules.early_stopping import EarlyStopping
from importance.importance_nn import (
    compute_feature_importance,
    compute_feature_importance_deepshap,
    compute_multiple_importance_nn,
    save_multiple_importance_nn,
    generate_multiple_importance_summary,
    generate_cross_method_importance_summary,
)
from modules.temporal_features import apply_temporal_features
from modules.technical_indicators import apply_technical_indicators
from modules.statistical_features import apply_statistical_features
from modules.interaction_features import apply_interaction_features
import config.config_nn as C
import gc

T = C.T  


def train_one_window(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols,
                     T: int, device: str, fold_idx: int, test_date_label: str):
    
    # ===== APPLY TRAINING SET STANDARDIZATION (if needed) =====
    if C.STANDARDIZE_MODE.lower() == 'train_set':
        tqdm.write("[MAIN] Applying training set standardization ...")
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()
        
        # Fit on training set, transform both train and test
        train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        
        # Use scaled data for sequence building
        train_df = train_df_scaled
        test_df = test_df_scaled
        
        # Clean up
        del scaler, train_df_scaled, test_df_scaled
    
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

    tqdm.write("[MAIN] Creating DataLoaders ...")
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=TRAIN_BATCH, shuffle=True)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=EVAL_BATCH, shuffle=False)

    tqdm.write("[MAIN] Initializing model ...")
    if C.SEQ_MODEL_TYPE == 'fnn':
        # FNN shares HIDDEN_SIZE, NUM_LAYERS, DROPOUT with other models
        model = FNNClassifier(
            input_size=len(feature_cols),
            seq_len=T,
            flatten_mode=FNN_FLATTEN_MODE
        ).to(device)
    elif C.SEQ_MODEL_TYPE == 'rnn':
        model = RNNClassifier(input_size=len(feature_cols), use_attention=C.ENABLE_ATTENTION).to(device)
    elif C.SEQ_MODEL_TYPE == 'lstm':
        model = LSTMClassifier(input_size=len(feature_cols), use_attention=C.ENABLE_ATTENTION).to(device)
    elif C.SEQ_MODEL_TYPE == 'bilstm':
        model = BiLSTMClassifier(
            input_size=len(feature_cols), 
            use_attention=C.ENABLE_ATTENTION,
            enable_deep_head=C.ENABLE_DEEP_HEAD,
            head_hidden_size=C.HEAD_HIDDEN_SIZE,
            head_num_layers=C.HEAD_NUM_LAYERS,
            enable_head_batch_norm=C.ENABLE_HEAD_BATCH_NORM,
            sequence_aggregation=C.SEQUENCE_AGGREGATION
        ).to(device)
    elif C.SEQ_MODEL_TYPE == 'gru':
        model = GRUClassifier(input_size=len(feature_cols), use_attention=C.ENABLE_ATTENTION).to(device)
    elif C.SEQ_MODEL_TYPE == 'transformer':
        # Transformer shares HIDDEN_SIZE, NUM_LAYERS, DROPOUT with other models
        model = TransformerClassifier(
            input_size=len(feature_cols),
            num_heads=TRANSFORMER_NUM_HEADS,
            max_len=TRANSFORMER_MAX_LEN,
            use_attention=False  # Transformer has built-in attention
        ).to(device)
    elif C.SEQ_MODEL_TYPE == 'cnn':
        # CNN shares HIDDEN_SIZE, DROPOUT with other models
        model = CNNClassifier(
            input_size=len(feature_cols),
            num_filters=CNN_NUM_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            num_conv_layers=CNN_NUM_CONV_LAYERS,
            pool_size=CNN_POOL_SIZE,
            pool_mode=CNN_POOL_MODE,
            seq_len=T
        ).to(device)
    else:
        raise ValueError(f"Unknown SEQ_MODEL_TYPE: {C.SEQ_MODEL_TYPE}")
    
    # ===== Initialize optimizer with weight decay =====
    if C.ENABLE_WEIGHT_DECAY:
        opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY_VALUE)
        tqdm.write(f"[MAIN] Using Adam optimizer with weight_decay={C.WEIGHT_DECAY_VALUE}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=C.LR)
        tqdm.write("[MAIN] Using Adam optimizer without weight_decay")
    
    # ===== Initialize learning rate scheduler =====
    scheduler = None
    if C.ENABLE_COSINE_ANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=C.COSINE_ANNEALING_T_MAX, eta_min=C.COSINE_ANNEALING_ETA_MIN
        )
        tqdm.write(f"[MAIN] Using CosineAnnealingLR (T_max={C.COSINE_ANNEALING_T_MAX}, eta_min={C.COSINE_ANNEALING_ETA_MIN})")
    
    # ===== Initialize early stopping =====
    early_stopping = None
    if C.ENABLE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=C.EARLY_STOPPING_PATIENCE,
            min_delta=C.EARLY_STOPPING_MIN_DELTA,
            mode=C.EARLY_STOPPING_MODE
        )
        tqdm.write(f"[MAIN] Using EarlyStopping (patience={C.EARLY_STOPPING_PATIENCE}, mode={C.EARLY_STOPPING_MODE})")
    
    # ===== Select loss by config =====
    loss_type = C.LOSS_TYPE.lower()
    if loss_type == 'ce':
        crit = torch.nn.CrossEntropyLoss()
    elif loss_type == 'bce':
        # BCEWithLogitsLoss uses the positive-class logit; extract it from the two-logit output
        # Convert y to float during training and use the positive-class logit
        # Forward still returns two logits; extract the positive-class column here
        crit = None  # Construct per batch later in the training loop to support pos_weight
    elif loss_type == 'focal':
        from modules.losses import FocalLoss
        crit = FocalLoss(alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA)
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {C.LOSS_TYPE}")

    tqdm.write("[MAIN] Training ...")
    model.train()
    # ===== Training loop with optimization techniques =====
    ema_loss = None
    best_loss = float('inf')
    best_model_state = None
    
    # Determine number of epochs
    max_epochs = C.MAX_EPOCHS if C.ENABLE_EARLY_STOPPING else TRAIN_EPOCHS
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        num_steps = 0
        num_samples = 0
        
        # Training phase
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            
            # Forward pass
            logits = model(xb)
            
            # Loss calculation
            if loss_type == 'ce':
                loss = crit(logits, yb)
            elif loss_type == 'bce':
                # Compute pos_weight once from the current training set distribution
                if 'bce_pos_weight_tensor' not in locals():
                    if C.BCE_POS_WEIGHT_MODE == 'balance':
                        with torch.no_grad():
                            y_all = torch.tensor(y_train, device=device)
                            n_pos = (y_all == 1).sum().item()
                            n_neg = (y_all == 0).sum().item()
                            pos_weight_val = n_neg / max(1, n_pos)
                            bce_pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
                    else:
                        bce_pos_weight_tensor = None
                    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight_tensor)
                # Extract the positive-class logit and convert yb to float with shape [batch, 1]
                logit_pos = logits[:, 1:2]
                loss = bce_loss_fn(logit_pos, yb.float().unsqueeze(1))
            elif loss_type == 'focal':
                loss = crit(logits, yb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if C.ENABLE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=C.GRADIENT_CLIPPING_MAX_NORM)
            
            opt.step()
            
            # Statistics
            batch_loss = float(loss.item())
            total_loss += batch_loss
            num_steps += 1
            num_samples += xb.size(0)
            if ema_loss is None:
                ema_loss = batch_loss
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * batch_loss
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Validation phase (for early stopping)
        val_loss = None
        if early_stopping is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    if loss_type == 'ce':
                        loss = crit(logits, yb)
                    elif loss_type == 'bce':
                        logit_pos = logits[:, 1:2]
                        loss = bce_loss_fn(logit_pos, yb.float().unsqueeze(1))
                    elif loss_type == 'focal':
                        loss = crit(logits, yb)
                    val_loss += loss.item()
                    val_steps += 1
            val_loss = val_loss / max(1, val_steps)
        
        # Statistics and logging
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, num_steps)
        best_loss = min(best_loss, avg_loss)
        throughput = num_samples / max(1e-6, epoch_time)
        
        # Logging
        log_msg = f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | EMA: {ema_loss:.4f} | Best: {best_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        log_msg += f" | Time: {epoch_time:.1f}s | {throughput/1000.0:.1f}k/s"
        tqdm.write(log_msg)
        
        # Early stopping check
        if early_stopping is not None and val_loss is not None:
            if val_loss < best_loss:
                best_model_state = model.state_dict().copy()
            
            if early_stopping(val_loss):
                tqdm.write(f"[MAIN] Early stopping triggered at epoch {epoch}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    tqdm.write("[MAIN] Evaluating ...")
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())

    y_proba = torch.softmax(torch.tensor(np.vstack(all_logits)), dim=1).numpy()
    y_true = np.concatenate(all_y)
    y_pred_class = y_proba.argmax(axis=1)
    y_pred_cont = y_proba[:, 1]

    tqdm.write("[MAIN] Displaying results ...")
    display_fold_results(C.MODEL_LABEL, fold_idx, test_date_label, np.array(y_true), np.array(y_pred_class), np.array(y_pred_cont))
    metrics_batch = []
    extract_and_store_metrics(np.array(y_true), np.array(y_pred_class), np.array(y_pred_cont), metrics_batch, fold_idx)
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
            # build a smaller loader for efficiency using test set (out-of-sample)
            imp_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=EVAL_BATCH, shuffle=False)
            
            # Compute multiple importance methods
            importance_results = compute_multiple_importance_nn(
                model=model,
                loader=imp_loader,
                feature_names=list(feature_cols),
                methods=C.IMPORTANCE_METHODS,
                device=device,
                target_class=C.IMPORTANCE_TARGET_CLASS,
                background_max_samples=C.DEEPSHAP_BACKGROUND_MAX_SAMPLES,
                time_aggregation=C.GRADIENT_TIME_AGGREGATION,
                steps=C.INTEGRATED_GRADIENTS_STEPS
            )
            
            # Save results from multiple importance methods
            # Append the window suffix to filenames in batch mode
            window_suffix = ""
            if C.BATCH_MODE:
                if C.START_WINDOW is not None and C.END_WINDOW is not None:
                    window_suffix = f"_{C.START_WINDOW}-{C.END_WINDOW}"
            
            save_multiple_importance_nn(
                importance_results=importance_results,
                feature_names=list(feature_cols),
                fold_idx=fold_idx,
                output_dir=C.OUTPUT_DIR,
                model_label=C.MODEL_LABEL,
                fold_suffix_template=C.IMPORTANCE_FOLD_SUFFIX,
                summary_suffix_template=C.IMPORTANCE_SUMMARY_SUFFIX,
                window_suffix=window_suffix  # Pass the window suffix
            )
    except Exception as e:
        tqdm.write(f"[WARN] Importance computation failed: {e}")
    # [MEM]
    del X_train, y_train, X_test, y_test
    del train_loader, test_loader
    del model, opt, crit
    del all_logits, all_y, y_proba, y_true
    gc.collect()

    # return fold metrics for summary
    return metrics_batch

def print_experiment_config():
    """Print the configuration parameters for this experiment."""
    print(f"\n=================== CONFIG ===================")
    
    # Model configuration
    print("\n[Model]")
    print(f"  MODEL_LABEL: {C.MODEL_LABEL}")
    print(f"  SEQ_MODEL_TYPE: {C.SEQ_MODEL_TYPE}")
    if C.SEQ_MODEL_TYPE == 'fnn':
        from model.model_nn import HIDDEN_SIZE, NUM_LAYERS, DROPOUT
        print(f"    - NUM_LAYERS: {NUM_LAYERS} (shared)")
        print(f"    - HIDDEN_SIZE: {HIDDEN_SIZE} (shared)")
        print(f"    - DROPOUT: {DROPOUT} (shared)")
        print(f"    - FLATTEN_MODE: {FNN_FLATTEN_MODE} (FNN-specific)")
    elif C.SEQ_MODEL_TYPE == 'cnn':
        from model.model_nn import HIDDEN_SIZE, DROPOUT
        print(f"    - NUM_FILTERS: {CNN_NUM_FILTERS} (CNN-specific)")
        print(f"    - KERNEL_SIZE: {CNN_KERNEL_SIZE} (CNN-specific)")
        print(f"    - NUM_CONV_LAYERS: {CNN_NUM_CONV_LAYERS} (CNN-specific)")
        print(f"    - POOL_SIZE: {CNN_POOL_SIZE} (CNN-specific)")
        print(f"    - POOL_MODE: {CNN_POOL_MODE} (CNN-specific)")
        print(f"    - HIDDEN_SIZE: {HIDDEN_SIZE} (shared, for FC layers)")
        print(f"    - DROPOUT: {DROPOUT} (shared)")
    elif C.SEQ_MODEL_TYPE == 'transformer':
        from model.model_nn import HIDDEN_SIZE, NUM_LAYERS, DROPOUT
        print(f"    - NUM_HEADS: {TRANSFORMER_NUM_HEADS} (Transformer-specific)")
        print(f"    - NUM_LAYERS: {NUM_LAYERS} (shared)")
        print(f"    - HIDDEN_DIM: {HIDDEN_SIZE} (shared)")
        print(f"    - DROPOUT: {DROPOUT} (shared)")
        print(f"    - MAX_LEN: {TRANSFORMER_MAX_LEN} (Transformer-specific)")
    print(f"  ENABLE_ATTENTION: {C.ENABLE_ATTENTION}")
    if C.ENABLE_ATTENTION and C.SEQ_MODEL_TYPE != 'transformer':
        print(f"  ATTENTION_TYPE: {C.ATTENTION_TYPE}")
    
    # Training configuration
    print("\n[Training]")
    print(f"  LOSS_TYPE: {C.LOSS_TYPE}")
    if C.LOSS_TYPE == 'bce':
        print(f"  BCE_POS_WEIGHT_MODE: {C.BCE_POS_WEIGHT_MODE}")
    elif C.LOSS_TYPE == 'focal':
        print(f"  FOCAL_ALPHA: {C.FOCAL_ALPHA}, FOCAL_GAMMA: {C.FOCAL_GAMMA}")
    print(f"  LR: {C.LR}")
    print(f"  MAX_EPOCHS: {C.MAX_EPOCHS}")
    print(f"  ENABLE_EARLY_STOPPING: {C.ENABLE_EARLY_STOPPING}")
    if C.ENABLE_EARLY_STOPPING:
        print(f"    - Patience: {C.EARLY_STOPPING_PATIENCE}, Mode: {C.EARLY_STOPPING_MODE}")
    print(f"  ENABLE_COSINE_ANNEALING: {C.ENABLE_COSINE_ANNEALING}")
    print(f"  ENABLE_GRADIENT_CLIPPING: {C.ENABLE_GRADIENT_CLIPPING}")
    if C.ENABLE_GRADIENT_CLIPPING:
        print(f"    - Max Norm: {C.GRADIENT_CLIPPING_MAX_NORM}")
    print(f"  ENABLE_WEIGHT_DECAY: {C.ENABLE_WEIGHT_DECAY}")
    if C.ENABLE_WEIGHT_DECAY:
        print(f"    - Weight Decay: {C.WEIGHT_DECAY_VALUE}")
    
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
    print(f"  ENABLE_LOG1P_TRANSFORM: {C.ENABLE_LOG1P_TRANSFORM}")
    print(f"  ENABLE_STANDARDIZATION: {C.ENABLE_STANDARDIZATION}")
    if C.ENABLE_STANDARDIZATION:
        print(f"    - STANDARDIZE_MODE: {C.STANDARDIZE_MODE}")
    print(f"  ENABLE_STATISTICAL_FEATURES: {C.ENABLE_STATISTICAL_FEATURES}")
    print(f"  ENABLE_INTERACTION_FEATURES: {C.ENABLE_INTERACTION_FEATURES}")
    print(f"  ENABLE_TECHNICAL_INDICATORS: {C.ENABLE_TECHNICAL_INDICATORS}")
    print(f"  ENABLE_TEMPORAL_FEATURES: {C.ENABLE_TEMPORAL_FEATURES}")
    
    # Importance configuration
    print("\n[Importance]")
    print(f"  COMPUTE_IMPORTANCE: {C.COMPUTE_IMPORTANCE}")
    if C.COMPUTE_IMPORTANCE:
        print(f"  IMPORTANCE_METHODS: {C.IMPORTANCE_METHODS}")
        print(f"  IMPORTANCE_TARGET_CLASS: {C.IMPORTANCE_TARGET_CLASS}")
        print(f"  GENERATE_CROSS_METHOD_SUMMARY: {C.GENERATE_CROSS_METHOD_SUMMARY}")
        if 'gradient' in C.IMPORTANCE_METHODS:
            print(f"    - GRADIENT_TIME_AGGREGATION: {C.GRADIENT_TIME_AGGREGATION}")
        if 'deepshap' in C.IMPORTANCE_METHODS:
            print(f"    - DEEPSHAP_BACKGROUND_MAX_SAMPLES: {C.DEEPSHAP_BACKGROUND_MAX_SAMPLES}")
            print(f"    - DEEPSHAP_TIME_AGGREGATION: {C.DEEPSHAP_TIME_AGGREGATION}")
        if 'integrated_gradients' in C.IMPORTANCE_METHODS:
            print(f"    - INTEGRATED_GRADIENTS_STEPS: {C.INTEGRATED_GRADIENTS_STEPS}")
            print(f"    - INTEGRATED_GRADIENTS_TIME_AGGREGATION: {C.INTEGRATED_GRADIENTS_TIME_AGGREGATION}")
    
    # Output configuration
    print("\n[Output]")
    print(f"  OUTPUT_DIR: {C.OUTPUT_DIR}")
    print(f"  SAVE_FOLD_CSV: {C.SAVE_FOLD_CSV}")
    print(f"  SAVE_SUMMARY_CSV: {C.SAVE_SUMMARY_CSV}")


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    else:
        print(f"\n--------- SEGMENT ---------\n")
        print("[MAIN] Skipping segment_id generation (ENABLE_SEGMENT_ID = False)")
        df_ = df_raw.copy()
    
    # [MEM]
    del df_raw
    gc.collect()

    print(f"\n==================== FE ====================")
    
    # ========== TEMPORAL FEATURES ==========
    if C.ENABLE_TEMPORAL_FEATURES:
        print(f"\n--------- TEMPORAL FEATURES ---------\n")
        df_ = apply_temporal_features(df_, feature_cols, C.TEMPORAL_FEATURES_CONFIG)
        print(f"[MAIN] Updated feature count: {len(feature_cols)}")
    else:
        print(f"\n--------- TEMPORAL FEATURES ---------\n")
        print("[MAIN] Skipping temporal features (ENABLE_TEMPORAL_FEATURES = False)")
    
    # ========== TECHNICAL INDICATORS ==========
    if C.ENABLE_TECHNICAL_INDICATORS:
        print(f"\n--------- TECHNICAL INDICATORS ---------\n")
        df_ = apply_technical_indicators(df_, feature_cols, C.TECHNICAL_INDICATORS_CONFIG)

        print(f"[MAIN] Updated feature count: {len(feature_cols)}")
    else:
        print(f"\n--------- TECHNICAL INDICATORS ---------\n")
        print("[MAIN] Skipping technical indicators (ENABLE_TECHNICAL_INDICATORS = False)")
    
    # ========== STATISTICAL FEATURES ==========
    if C.ENABLE_STATISTICAL_FEATURES:
        print(f"\n--------- STATISTICAL FEATURES ---------\n")
        df_ = apply_statistical_features(df_, feature_cols, C.STATISTICAL_FEATURES_CONFIG)
        print(f"[MAIN] Updated feature count: {len(feature_cols)}")
    else:
        print(f"\n--------- STATISTICAL FEATURES ---------\n")
        print("[MAIN] Skipping statistical features (ENABLE_STATISTICAL_FEATURES = False)")
    
    # ========== INTERACTION FEATURES ==========
    if C.ENABLE_INTERACTION_FEATURES:
        print(f"\n--------- INTERACTION FEATURES ---------\n")
        df_ = apply_interaction_features(df_, feature_cols, C.INTERACTION_FEATURES_CONFIG)
        print(f"[MAIN] Updated feature count: {len(feature_cols)}")
    else:
        print(f"\n--------- INTERACTION FEATURES ---------\n")
        print("[MAIN] Skipping interaction features (ENABLE_INTERACTION_FEATURES = False)")
    
    # ========== LAYERED MISSING FILL ==========
    if C.ENABLE_LAYERED_FILL:
        print(f"\n--------- FILL ---------\n")
        df_ = layered_missing_fill(df_, feature_cols, T)
    else:
        print(f"\n--------- FILL ---------\n")
        print("[MAIN] Skipping layered missing fill (ENABLE_LAYERED_FILL = False)")
    
    # ========== LOG1P TRANSFORMATION ==========
    if C.ENABLE_LOG1P_TRANSFORM:
        print(f"\n--------- LOG1P ---------\n")
        df_ = long_tail_log1p_transform(df_, feature_cols)
    else:
        print(f"\n--------- LOG1P ---------\n")
        print("[MAIN] Skipping log1p transformation (ENABLE_LOG1P_TRANSFORM = False)")
    
    # ========== STANDARDIZATION ==========
    if C.ENABLE_STANDARDIZATION:
        print(f"\n--------- STANDARD ---------\n")
        std_mode = C.STANDARDIZE_MODE.lower()
        if std_mode == 'rolling':
            df_ = rolling_standardize_by_month(df_, feature_cols, T)
        elif std_mode == 'global':
            df_ = global_standardize(df_, feature_cols)
        elif std_mode == 'train_set':
            tqdm.write("[MAIN] Skipping global standardization - will be done per-window on training set")
        else:
            raise ValueError(f"Unknown STANDARDIZE_MODE: {std_mode}. Use 'rolling', 'global', or 'train_set'.")
    else:
        print(f"\n--------- STANDARD ---------\n")
        print("[MAIN] Skipping standardization (ENABLE_STANDARDIZATION = False)")
    
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

    print("[MAIN] LSTM | Starting sliding-window training ...")
    all_results = []  # collect all fold results for summary
    p_windows = tqdm(total=total_iters, desc="Windows", leave=True)
    for i in window_iter:
        set_seed(i)
        test_date_label = str(all_dates[i+1])
        tqdm.write(f"\n---------- [ FOLD {i} | {test_date_label} ] ----------\n")
        train_df, test_df = split_train_test_by_window(df_, all_dates, i, T)
        fold_metrics = train_one_window(train_df, test_df, feature_cols, T, device, i, test_date_label)
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
    print(f"\n[MAIN] {C.MODEL_LABEL} DONE\n")