import pandas as pd
import numpy as np
from typing import List
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler


def layered_missing_fill(df: pd.DataFrame, feature_cols: List[str], T: int = 36) -> pd.DataFrame:
    """
    Layered missing fill:
    1) ffill within segment (limit=T) - if segment_id exists
    2) fill with segment mean - if segment_id exists
    3) fill with (accid, permno) group mean
    4) fallback to 0
    """
    out = df.copy()
    
    # Check if segment_id exists
    has_segment_id = 'segment_id' in out.columns
    
    # ensure sorting
    print("[FE] Sorting ...")
    if has_segment_id:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        print("[FE] Using segment_id for layered fill ...")
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        print("[FE] No segment_id found, using simplified layered fill ...")
    
    out = out.sort_values(by=sort_keys).reset_index(drop=True)
    print("[FE] Layered missing fill ...")

    # Define grouping keys based on segment_id availability
    if has_segment_id:
        seg_keys = ['accid', 'permno', 'segment_id']
    else:
        seg_keys = ['accid', 'permno']  # Use accid, permno as segment-level grouping
    
    grp_keys = ['accid', 'permno']

    for col in tqdm(feature_cols, desc="[FE] Layered fill"):
        # Step 1: segment-wise ffill(limit=T)
        out[col] = out.groupby(seg_keys, sort=False)[col].transform(lambda x: x.ffill(limit=T))
        
        # Step 2: segment mean (or accid, permno mean if no segment_id)
        seg_mean = out.groupby(seg_keys, sort=False)[col].transform('mean')
        out[col] = out[col].fillna(seg_mean)
        
        # Step 3: (accid, permno) group mean (only if segment_id exists, otherwise redundant)
        if has_segment_id:
            grp_mean = out.groupby(grp_keys, sort=False)[col].transform('mean')
            out[col] = out[col].fillna(grp_mean)

    # Step 4: fallback to 0
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out

def long_tail_log1p_transform(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Detect long-tailed features and apply log1p as in v1 notebook."""
    out = df.copy()
    print("[FE] Detecting long-tailed features ...")
    # skew on df[feature_cols] (no numeric_only), threshold > 1
    skew_vals = out[feature_cols].skew()
    long_tail_vars = skew_vals[skew_vals > 1].index.tolist()
    print(f"[FE] Detected {len(long_tail_vars)} long-tailed features for log1p transform:")
    if len(long_tail_vars) > 0:
        print(long_tail_vars)
    # for each feature, shift to positive if needed, then log1p; no dtype casts
    for col in tqdm(long_tail_vars, desc="[FE] Features"):
        min_val = out[col].min()
        if pd.isna(min_val):
            # keep behavior simple: if entire col is NaN, skip
            continue
        if min_val <= 0:
            out[col] = out[col] + abs(min_val) + 1e-6
        out[col] = np.log1p(out[col])
    return out

def rolling_standardize_by_month(df: pd.DataFrame, feature_cols: List[str], T: int = 36) -> pd.DataFrame:
    """Rolling StandardScaler per feature: fit on window [i-T+1, i] and transform month i (v1-consistent)."""
    out = df.copy()
    print("[FE] Rolling standardization (StandardScaler per feature) ...")
    out['mdate_dt'] = pd.to_datetime(out['mdate'])
    months = sorted(out['mdate_dt'].unique())
    for i, month in enumerate(tqdm(months, desc="[FE] Months")):
        start = months[max(0, i - (T - 1))]
        window_mask = (out['mdate_dt'] >= start) & (out['mdate_dt'] <= month)
        month_mask = (out['mdate_dt'] == month)
        # Fit-transform per feature to match notebook behavior
        for feature in feature_cols:
            scaler = StandardScaler()
            # fit on window data (including current month per v1)
            try:
                window_vals = out.loc[window_mask, feature].values.reshape(-1, 1)
                scaler.fit(window_vals)
                current_vals = out.loc[month_mask, feature].values.reshape(-1, 1)
                transformed = scaler.transform(current_vals).astype('float32')
                out.loc[month_mask, feature] = transformed.flatten()
            except Exception:
                # fallback: no-op if scaler fails
                pass
    out = out.drop(columns=['mdate_dt'])
    return out

def global_standardize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Fit a single StandardScaler on all rows (per-feature standardization) and
    transform all rows. Mirrors notebook's global fit_transform behavior.
    """
    out = df.copy()
    print("[FE] Global standardization (fit on all rows) ...")
    scaler = StandardScaler()
    try:
        out[feature_cols] = scaler.fit_transform(out[feature_cols])
    except Exception:
        # Fallback: attempt per-feature transformation to avoid entire failure
        for feature in tqdm(feature_cols, desc="[FE] Global per-feature"):
            try:
                vals = out[[feature]].values
                out[feature] = StandardScaler().fit_transform(vals).astype('float32')
            except Exception:
                # leave the column as-is if transformation fails
                pass
    return out

