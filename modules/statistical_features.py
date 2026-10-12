import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import warnings


def add_rolling_statistical_features(df: pd.DataFrame, feature_cols: List[str], 
                                   windows: List[int] = [3, 6, 12, 24],
                                   functions: List[str] = ['mean', 'std', 'min', 'max', 'median', 'skew']) -> pd.DataFrame:
    """
    Add rolling statistical features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        windows: List of rolling window sizes.
        functions: List of statistical functions.
    
    Returns:
        DataFrame with rolling statistical features added.
    """
    out = df.copy()
    print(f"[STAT_FEAT] Adding rolling statistical features for {len(feature_cols)} features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    new_cols_frames = []
    
    for window in tqdm(windows, desc="[STAT_FEAT] Windows"):
        cols_dict = {}
        for feature in feature_cols:
            for func in functions:
                feature_name = f"{feature}_roll_{window}_{func}"
                # Compute rolling statistics within each group
                if func == 'mean':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                elif func == 'std':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                elif func == 'min':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                elif func == 'max':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                elif func == 'median':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).median()
                    )
                elif func == 'skew':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=3).skew()
                    )
                new_features.append(feature_name)
        # Concatenate columns once per window to reduce fragmentation
        if cols_dict:
            new_cols_frames.append(pd.DataFrame(cols_dict, index=out.index))
    if new_cols_frames:
        out = pd.concat([out] + new_cols_frames, axis=1)
    
    print(f"[STAT_FEAT] Added {len(new_features)} rolling statistical features")
    return out


def add_quantile_features(df: pd.DataFrame, feature_cols: List[str],
                         windows: List[int] = [6, 12, 24],
                         quantiles: List[float] = [0.1, 0.25, 0.75, 0.9]) -> pd.DataFrame:
    """
    Add quantile features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        windows: List of rolling window sizes.
        quantiles: List of quantiles.
    
    Returns:
        DataFrame with quantile features added.
    """
    out = df.copy()
    print(f"[QUANTILE_FEAT] Adding quantile features for {len(feature_cols)} features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    new_cols_frames = []
    
    for window in tqdm(windows, desc="[QUANTILE_FEAT] Windows"):
        cols_dict = {}
        for feature in feature_cols:
            for q in quantiles:
                feature_name = f"{feature}_roll_{window}_q{int(q*100)}"
                cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).quantile(q)
                )
                new_features.append(feature_name)
        if cols_dict:
            new_cols_frames.append(pd.DataFrame(cols_dict, index=out.index))
    if new_cols_frames:
        out = pd.concat([out] + new_cols_frames, axis=1)
    
    print(f"[QUANTILE_FEAT] Added {len(new_features)} quantile features")
    return out


def add_rank_features(df: pd.DataFrame, feature_cols: List[str],
                     windows: List[int] = [6, 12, 24],
                     methods: List[str] = ['percentile', 'zscore']) -> pd.DataFrame:
    """
    Add rank features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        windows: List of rolling window sizes.
        methods: List of ranking methods.
    
    Returns:
        DataFrame with rank features added.
    """
    out = df.copy()
    print(f"[RANK_FEAT] Adding rank features for {len(feature_cols)} features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    new_cols_frames = []
    
    for window in tqdm(windows, desc="[RANK_FEAT] Windows"):
        cols_dict = {}
        for feature in feature_cols:
            for method in methods:
                feature_name = f"{feature}_roll_{window}_{method}"
                if method == 'percentile':
                    cols_dict[feature_name] = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).rank(pct=True)
                    )
                elif method == 'zscore':
                    rolling_mean = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    rolling_std = out.groupby(group_keys)[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    cols_dict[feature_name] = (out[feature] - rolling_mean) / (rolling_std + 1e-8)
                new_features.append(feature_name)
        if cols_dict:
            new_cols_frames.append(pd.DataFrame(cols_dict, index=out.index))
    if new_cols_frames:
        out = pd.concat([out] + new_cols_frames, axis=1)
    
    print(f"[RANK_FEAT] Added {len(new_features)} rank features")
    return out


def add_change_features(df: pd.DataFrame, feature_cols: List[str],
                       periods: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Add change features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        periods: List of change periods.
    
    Returns:
        DataFrame with change features added.
    """
    out = df.copy()
    print(f"[CHANGE_FEAT] Adding change features for {len(feature_cols)} features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    new_cols_frames = []
    
    for period in tqdm(periods, desc="[CHANGE_FEAT] Periods"):
        cols_dict = {}
        for feature in feature_cols:
            feature_name_abs = f"{feature}_change_{period}"
            cols_dict[feature_name_abs] = out.groupby(group_keys)[feature].transform(
                lambda x: x.diff(period)
            )
            new_features.append(feature_name_abs)
            feature_name_pct = f"{feature}_pct_change_{period}"
            cols_dict[feature_name_pct] = out.groupby(group_keys)[feature].transform(
                lambda x: x.pct_change(period)
            )
            new_features.append(feature_name_pct)
        if cols_dict:
            new_cols_frames.append(pd.DataFrame(cols_dict, index=out.index))
    if new_cols_frames:
        out = pd.concat([out] + new_cols_frames, axis=1)
    
    print(f"[CHANGE_FEAT] Added {len(new_features)} change features")
    return out


def add_volatility_features(df: pd.DataFrame, feature_cols: List[str],
                           windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
    """
    Add volatility features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        windows: List of rolling window sizes.
    
    Returns:
        DataFrame with volatility features added.
    """
    out = df.copy()
    print(f"[VOLATILITY_FEAT] Adding volatility features for {len(feature_cols)} features...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    new_cols_frames = []
    
    for window in tqdm(windows, desc="[VOLATILITY_FEAT] Windows"):
        cols_dict = {}
        for feature in feature_cols:
            feature_name_std = f"{feature}_vol_{window}"
            cols_dict[feature_name_std] = out.groupby(group_keys)[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            new_features.append(feature_name_std)
            rolling_mean = out.groupby(group_keys)[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            rolling_std = out.groupby(group_keys)[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            feature_name_cv = f"{feature}_cv_{window}"
            cols_dict[feature_name_cv] = rolling_std / (rolling_mean + 1e-8)
            new_features.append(feature_name_cv)
        if cols_dict:
            new_cols_frames.append(pd.DataFrame(cols_dict, index=out.index))
    if new_cols_frames:
        out = pd.concat([out] + new_cols_frames, axis=1)
    
    print(f"[VOLATILITY_FEAT] Added {len(new_features)} volatility features")
    return out


def apply_statistical_features(df: pd.DataFrame, feature_cols: List[str],
                             config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Apply all statistical feature engineering steps.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with statistical features added.
    """
    if config is None:
        config = {
            'rolling_stats': {
                'windows': [3, 6, 12, 24],
                'functions': ['mean', 'std', 'min', 'max', 'median', 'skew']
            },
            'quantile_features': {
                'windows': [6, 12, 24],
                'quantiles': [0.1, 0.25, 0.75, 0.9]
            },
            'rank_features': {
                'windows': [6, 12, 24],
                'methods': ['percentile', 'zscore']
            },
            'change_features': {
                'periods': [1, 3, 6, 12]
            },
            'volatility_features': {
                'windows': [6, 12, 24]
            }
        }
    
    print("=" * 60)
    print("APPLYING STATISTICAL FEATURE ENGINEERING")
    print("=" * 60)
    
    # Apply the statistical feature generators
    if config.get('rolling_stats', {}).get('enabled', True):
        df = add_rolling_statistical_features(
            df, feature_cols,
            windows=config['rolling_stats']['windows'],
            functions=config['rolling_stats']['functions']
        )
    
    if config.get('quantile_features', {}).get('enabled', True):
        df = add_quantile_features(
            df, feature_cols,
            windows=config['quantile_features']['windows'],
            quantiles=config['quantile_features']['quantiles']
        )
    
    if config.get('rank_features', {}).get('enabled', True):
        df = add_rank_features(
            df, feature_cols,
            windows=config['rank_features']['windows'],
            methods=config['rank_features']['methods']
        )
    
    if config.get('change_features', {}).get('enabled', True):
        df = add_change_features(
            df, feature_cols,
            periods=config['change_features']['periods']
        )
    
    if config.get('volatility_features', {}).get('enabled', True):
        df = add_volatility_features(
            df, feature_cols,
            windows=config['volatility_features']['windows']
        )
    
    print("=" * 60)
    print("STATISTICAL FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    
    return df
