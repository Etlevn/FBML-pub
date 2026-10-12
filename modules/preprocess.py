import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm


def build_sequences_last_T(df: pd.DataFrame, feature_cols: List[str], T: int = 36,
                           group_keys=None):
    """Take last T steps per group (front-pad with zeros if shorter). Return X, y."""
    tqdm.write("[PREP] Building last-T sequences ...")
    
    # Auto-detect group keys based on available columns
    if group_keys is None:
        if 'segment_id' in df.columns:
            group_keys = ('accid', 'permno', 'segment_id')
        else:
            group_keys = ('accid', 'permno')
    
    X_list, y_list = [] , []
    for _, g in tqdm(df.groupby(list(group_keys)), desc="Groups", leave=False):
        g = g.sort_values('mdate')
        vals = g[feature_cols].values.astype(np.float32)
        targets = g['y'].values
        if len(g) < T:
            pad = np.zeros((T - len(g), len(feature_cols)), dtype=np.float32)
            X_list.append(np.vstack([pad, vals]))
            y_list.append(targets[-1])
        else:
            X_list.append(vals[-T:])
            y_list.append(targets[-1])
    return np.stack(X_list), np.asarray(y_list)


def build_sequences_cumulative(df: pd.DataFrame, feature_cols: List[str], T: int = 36,
                              group_keys=None):
    """Build sequences using cumulative data (v1.5 style). Return X, y."""
    tqdm.write("[PREP] Building cumulative sequences (v1.5 style) ...")
    
    # Auto-detect group keys based on available columns
    if group_keys is None:
        if 'segment_id' in df.columns:
            group_keys = ('accid', 'permno', 'segment_id')
        else:
            group_keys = ('accid', 'permno')
    
    X_list, y_list = [], []
    
    # Get unique dates in training data
    train_dates = sorted(df['mdate'].unique())
    tqdm.write(f"[PREP] Processing {len(train_dates)} months cumulatively ...")
    
    for month_idx, current_month in tqdm(enumerate(train_dates), desc="Months", leave=False):
        # Cumulative data up to current month
        current_mask = df['mdate'] <= current_month
        current_df = df[current_mask]
        
        # Build sequences for current month
        for _, g in tqdm(current_df.groupby(list(group_keys)), desc=f"Groups-M{month_idx}", leave=False):
            g = g.sort_values('mdate')
            vals = g[feature_cols].values.astype(np.float32)
            targets = g['y'].values
            
            if len(g) < T:
                pad = np.zeros((T - len(g), len(feature_cols)), dtype=np.float32)
                X_list.append(np.vstack([pad, vals]))
                y_list.append(targets[-1])
            else:
                X_list.append(vals[-T:])
                y_list.append(targets[-1])
    
    tqdm.write(f"[PREP] Built {len(X_list)} cumulative sequences")
    return np.stack(X_list), np.asarray(y_list)


def generate_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create segment_id per (accid, permno) based on month continuity."""
    out = df.copy()
    print(f"[PREP] Generating segment_id ...")
    out['mdate'] = pd.to_datetime(out['mdate'])
    out = out.sort_values(by=['accid', 'permno', 'mdate']).reset_index(drop=True)
    out['month_num'] = out['mdate'].dt.to_period('M').astype(int)
    out['segment_id'] = 0

    for (accid, permno), group_idx in tqdm(out.groupby(['accid', 'permno']).indices.items(), desc="Segments"):
        month_vals = out.loc[group_idx, 'month_num'].values
        month_diff = np.diff(month_vals, prepend=month_vals[0])
        break_flags = (month_diff > 1).astype(int)
        segment_ids = np.cumsum(break_flags) + 1
        out.loc[group_idx, 'segment_id'] = segment_ids

    out.drop(columns=['month_num'], inplace=True)
    print(f"[PREP] Restoring mdate ...")
    out['mdate'] = out['mdate'].dt.to_period('M').astype(str)
    return out


def unify_and_prepare(df: pd.DataFrame,
                      action: str,        
                      include_nan: bool,  
                      task: str           
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize schema and generate targets:
    - NaN handling (FILL0 or DROP)
    - Target construction (bin/tri)
    - Return df and feature_cols
    """
    print("[PREP] Unifying schema & preparing targets ...")
    out = df.copy()

    # required columns
    for col in ['accid','permno','mdate']:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    # NaN handling
    print(f"[PREP] Mode: action={action}, task={task}, nan={'FILL0' if include_nan else 'DROP'}")
    if ('buy' in out.columns) and ('sell' in out.columns):
        try:
            before_counts = out[['buy','sell']].value_counts(dropna=True)
            print(f"\n[PREP] buy/sell distribution BEFORE:\n", before_counts, "\n")
        except Exception:
            pass
        if include_nan:
            out['buy'] = out['buy'].fillna(0)
            out['sell'] = out['sell'].fillna(0)
        else:
            out = out[~(out['buy'].isna() & out['sell'].isna())].copy()

    # target construction
    action_col = action.lower().strip()
    if action_col not in out.columns:
        raise ValueError(f"Missing column: {action_col}")

    target_name = 'y'
    if task == 'bin':
        out[target_name] = (out[action_col] == 1).astype(int)
        try:
            print("[PREP] y distribution AFTER (bin):\n", out[target_name].value_counts().sort_index().rename('count'))
        except Exception:
            pass
    elif task == 'tri':
        def classify_trade_regression(row):
            if row.get('buy', 0) == 1 and row.get('sell', 0) == 0:
                return 1
            elif row.get('sell', 0) == 1 and row.get('buy', 0) == 0:
                return -1
            else:
                return 0
        out[target_name] = out.apply(classify_trade_regression, axis=1).astype(int)
        try:
            print("[PREP] y distribution AFTER (tri):\n", out[target_name].value_counts().sort_index().rename('count'))
        except Exception:
            pass
    else:
        raise ValueError("task must be 'bin' or 'tri'")

    # sorting & feature columns
    print(f"\n[PREP] Sorting & preparing feature columns ...")
    group_keys = ['accid','permno','segment_id'] if 'segment_id' in out.columns else ['accid','permno']
    sort_keys = [c for c in group_keys + ['mdate'] if c in out.columns]
    out = out.sort_values(sort_keys)

    exclude = set(['accid','permno','segment_id','mdate','buy','sell', target_name])
    feature_cols = [c for c in out.columns if c not in exclude]
    return out, feature_cols
