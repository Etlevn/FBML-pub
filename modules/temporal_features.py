import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import warnings


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encoding features.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with cyclical encoding features added.
    """
    out = df.copy()
    print(f"[TEMP_FEAT] Adding cyclical encoding...")
    
    # Ensure mdate has datetime type
    out['mdate_dt'] = pd.to_datetime(out['mdate'])
    
    # Cyclical month encoding
    out['month_sin'] = np.sin(2 * np.pi * out['mdate_dt'].dt.month / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['mdate_dt'].dt.month / 12)
    
    # Cyclical quarter encoding
    out['quarter_sin'] = np.sin(2 * np.pi * out['mdate_dt'].dt.quarter / 4)
    out['quarter_cos'] = np.cos(2 * np.pi * out['mdate_dt'].dt.quarter / 4)
    
    # Cyclical year encoding relative to the first year in the data
    start_year = out['mdate_dt'].dt.year.min()
    out['year_sin'] = np.sin(2 * np.pi * (out['mdate_dt'].dt.year - start_year) / 10)  # Ten-year cycle
    out['year_cos'] = np.cos(2 * np.pi * (out['mdate_dt'].dt.year - start_year) / 10)
    
    # Remove temporary columns
    out = out.drop(columns=['mdate_dt'])
    
    print(f"[TEMP_FEAT] Added 6 cyclical encoding features")
    return out


def add_lag_features(df: pd.DataFrame, feature_cols: List[str],
                    lags: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Add lag features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        lags: List of lag periods.
    
    Returns:
        DataFrame with lag features added.
    """
    out = df.copy()
    print(f"[TEMP_FEAT] Adding lag features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    
    # Select important features for lagging
    important_features = [f for f in feature_cols if f in ['price', 'VOL', 'RET', 'roa', 'roe', 'curr_ratio', 'debt_assets', 'GDP_g', 'INFLATION']]
    
    for feature in tqdm(important_features, desc="[TEMP_FEAT] Lag features"):
        if feature not in out.columns:
            continue
            
        for lag in lags:
            feature_name = f"{feature}_lag_{lag}"
            out[feature_name] = out.groupby(group_keys)[feature].shift(lag)
            new_features.append(feature_name)
    
    print(f"[TEMP_FEAT] Added {len(new_features)} lag features")
    return out


def add_lead_features(df: pd.DataFrame, feature_cols: List[str],
                     leads: List[int] = [1, 3, 6]) -> pd.DataFrame:
    """
    Add lead features; use cautiously because they can cause data leakage.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        leads: List of lead periods.
    
    Returns:
        DataFrame with lead features added.
    """
    out = df.copy()
    print(f"[TEMP_FEAT] Adding lead features (use with caution)...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    
    # Use lead features only for macroeconomic indicators assumed to be known
    macro_features = [f for f in feature_cols if f in ['GDP_g', 'con_g', 'ipt_g', 'INFLATION', 'Unemployment']]
    
    for feature in tqdm(macro_features, desc="[TEMP_FEAT] Lead features"):
        if feature not in out.columns:
            continue
            
        for lead in leads:
            feature_name = f"{feature}_lead_{lead}"
            out[feature_name] = out.groupby(group_keys)[feature].shift(-lead)
            new_features.append(feature_name)
    
    print(f"[TEMP_FEAT] Added {len(new_features)} lead features")
    return out


def add_trend_features(df: pd.DataFrame, feature_cols: List[str],
                      windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Add trend features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        windows: List of trend calculation windows.
    
    Returns:
        DataFrame with trend features added.
    """
    out = df.copy()
    print(f"[TEMP_FEAT] Adding trend features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    
    # Select important features for trend calculation
    important_features = [f for f in feature_cols if f in ['price', 'VOL', 'RET', 'roa', 'roe', 'GDP_g', 'INFLATION']]
    
    for feature in tqdm(important_features, desc="[TEMP_FEAT] Trend features"):
        if feature not in out.columns:
            continue
            
        for window in windows:
            # Linear trend slope
            feature_name_slope = f"{feature}_trend_slope_{window}"
            out[feature_name_slope] = out.groupby(group_keys)[feature].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else np.nan
                )
            )
            new_features.append(feature_name_slope)
            
            # Trend strength (R squared)
            feature_name_r2 = f"{feature}_trend_r2_{window}"
            out[feature_name_r2] = out.groupby(group_keys)[feature].transform(
                lambda x: x.rolling(window=window, min_periods=3).apply(
                    lambda y: np.corrcoef(range(len(y)), y)[0, 1]**2 if len(y) >= 3 else np.nan
                )
            )
            new_features.append(feature_name_r2)
    
    print(f"[TEMP_FEAT] Added {len(new_features)} trend features")
    return out


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal features.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with seasonal features added.
    """
    out = df.copy()
    print(f"[TEMP_FEAT] Adding seasonal features...")
    
    # Ensure mdate has datetime type
    out['mdate_dt'] = pd.to_datetime(out['mdate'])
    
    # Month features
    out['month'] = out['mdate_dt'].dt.month
    out['quarter'] = out['mdate_dt'].dt.quarter
    out['year'] = out['mdate_dt'].dt.year
    
    # Seasonal indicators
    out['is_q1'] = (out['quarter'] == 1).astype(int)
    out['is_q2'] = (out['quarter'] == 2).astype(int)
    out['is_q3'] = (out['quarter'] == 3).astype(int)
    out['is_q4'] = (out['quarter'] == 4).astype(int)
    
    # Month indicators
    out['is_jan'] = (out['month'] == 1).astype(int)
    out['is_feb'] = (out['month'] == 2).astype(int)
    out['is_mar'] = (out['month'] == 3).astype(int)
    out['is_apr'] = (out['month'] == 4).astype(int)
    out['is_may'] = (out['month'] == 5).astype(int)
    out['is_jun'] = (out['month'] == 6).astype(int)
    out['is_jul'] = (out['month'] == 7).astype(int)
    out['is_aug'] = (out['month'] == 8).astype(int)
    out['is_sep'] = (out['month'] == 9).astype(int)
    out['is_oct'] = (out['month'] == 10).astype(int)
    out['is_nov'] = (out['month'] == 11).astype(int)
    out['is_dec'] = (out['month'] == 12).astype(int)
    
    # Remove temporary columns
    out = out.drop(columns=['mdate_dt'])
    
    print(f"[TEMP_FEAT] Added 20 seasonal features")
    return out


def add_time_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-distance features.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with time-distance features added.
    """
    out = df.copy()
    print(f"[TEMP_FEAT] Adding time distance features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    # Ensure mdate has datetime type
    out['mdate_dt'] = pd.to_datetime(out['mdate'])
    
    # Compute time since the start of the data
    start_date = out['mdate_dt'].min()
    out['months_since_start'] = ((out['mdate_dt'] - start_date).dt.days / 30.44).astype(int)
    
    # Compute time since the start of each group
    out['months_since_group_start'] = out.groupby(group_keys)['mdate_dt'].transform(
        lambda x: ((x - x.min()).dt.days / 30.44).astype(int)
    )
    
    # Compute time until the end of each group
    out['months_to_group_end'] = out.groupby(group_keys)['mdate_dt'].transform(
        lambda x: ((x.max() - x).dt.days / 30.44).astype(int)
    )
    
    # Remove temporary columns
    out = out.drop(columns=['mdate_dt'])
    
    print(f"[TEMP_FEAT] Added 3 time distance features")
    return out


def apply_temporal_features(df: pd.DataFrame, feature_cols: List[str],
                          config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Apply all temporal feature engineering steps.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with temporal features added.
    """
    if config is None:
        config = {
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
    
    print("=" * 60)
    print("APPLYING TEMPORAL FEATURE ENGINEERING")
    print("=" * 60)
    
    # Apply the temporal feature generators
    if config.get('cyclical_encoding', {}).get('enabled', True):
        df = add_cyclical_encoding(df)
    
    if config.get('lag_features', {}).get('enabled', True):
        df = add_lag_features(
            df, feature_cols,
            lags=config['lag_features']['lags']
        )
    
    if config.get('lead_features', {}).get('enabled', False):
        df = add_lead_features(
            df, feature_cols,
            leads=config['lead_features']['leads']
        )
    
    if config.get('trend_features', {}).get('enabled', True):
        df = add_trend_features(
            df, feature_cols,
            windows=config['trend_features']['windows']
        )
    
    if config.get('seasonal_features', {}).get('enabled', True):
        df = add_seasonal_features(df)
    
    if config.get('time_distance_features', {}).get('enabled', True):
        df = add_time_distance_features(df)
    
    print("=" * 60)
    print("TEMPORAL FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    
    return df
