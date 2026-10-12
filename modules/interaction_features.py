import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
import warnings
from itertools import combinations


def add_ratio_features(df: pd.DataFrame, feature_cols: List[str],
                      ratio_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Add ratio features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        ratio_pairs: List of ratio pairs in the form [(numerator, denominator), ...].
    
    Returns:
        DataFrame with ratio features added.
    """
    out = df.copy()
    print(f"[RATIO_FEAT] Adding ratio features...")
    
    new_features = []
    
    for numerator, denominator in tqdm(ratio_pairs, desc="[RATIO_FEAT] Ratio pairs"):
        if numerator in out.columns and denominator in out.columns:
            # Avoid division by zero
            feature_name = f"{numerator}_div_{denominator}"
            out[feature_name] = np.where(
                out[denominator] != 0,
                out[numerator] / (out[denominator] + 1e-8),
                0
            )
            new_features.append(feature_name)
        else:
            print(f"[RATIO_FEAT] Warning: {numerator} or {denominator} not found in columns")
    
    print(f"[RATIO_FEAT] Added {len(new_features)} ratio features")
    return out


def add_polynomial_features(df: pd.DataFrame, feature_cols: List[str],
                           degree: int = 2, selected_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add polynomial features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        degree: Polynomial degree.
        selected_features: Features selected for polynomial transformation.
    
    Returns:
        DataFrame with polynomial features added.
    """
    out = df.copy()
    print(f"[POLY_FEAT] Adding polynomial features (degree={degree})...")
    
    if selected_features is None:
        # Select numeric features, excluding already transformed features
        numeric_features = []
        for col in feature_cols:
            if col in out.columns and out[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Exclude features containing special markers such as ratio or roll
                if not any(x in col for x in ['_div_', '_roll_', '_change_', '_vol_', '_cv_']):
                    numeric_features.append(col)
        selected_features = numeric_features[:10]  # Limit feature count to avoid dimensionality explosion
    
    new_features = []
    
    for feature in tqdm(selected_features, desc="[POLY_FEAT] Features"):
        if feature in out.columns:
            for d in range(2, degree + 1):
                feature_name = f"{feature}_pow_{d}"
                out[feature_name] = np.power(out[feature], d)
                new_features.append(feature_name)
    
    print(f"[POLY_FEAT] Added {len(new_features)} polynomial features")
    return out


def add_cross_features(df: pd.DataFrame, feature_cols: List[str],
                      feature_groups: List[List[str]]) -> pd.DataFrame:
    """
    Add cross features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        feature_groups: List of feature groups; features within each group are crossed.
    
    Returns:
        DataFrame with cross features added.
    """
    out = df.copy()
    print(f"[CROSS_FEAT] Adding cross features...")
    
    new_features = []
    
    for group in tqdm(feature_groups, desc="[CROSS_FEAT] Feature groups"):
        # Check whether every feature in the group exists
        available_features = [f for f in group if f in out.columns]
        if len(available_features) < 2:
            print(f"[CROSS_FEAT] Warning: Not enough features in group {group}")
            continue
        
        # Generate all pairwise combinations
        for feat1, feat2 in combinations(available_features, 2):
            # Product features
            feature_name_mult = f"{feat1}_x_{feat2}"
            out[feature_name_mult] = out[feat1] * out[feat2]
            new_features.append(feature_name_mult)
            
            # Difference features
            feature_name_diff = f"{feat1}_minus_{feat2}"
            out[feature_name_diff] = out[feat1] - out[feat2]
            new_features.append(feature_name_diff)
    
    print(f"[CROSS_FEAT] Added {len(new_features)} cross features")
    return out


def add_financial_ratio_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Add financial ratio features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    
    Returns:
        DataFrame with financial ratio features added.
    """
    out = df.copy()
    print(f"[FIN_RATIO_FEAT] Adding financial ratio features...")
    
    new_features = []
    
    # Define financial ratio pairs
    financial_ratios = [
        # Profitability ratios
        ('roa', 'roa'),  # ROA itself
        ('roe', 'roe'),  # ROE itself
        ('npm', 'npm'),  # Net profit margin itself
        
        # Liquidity ratios
        ('curr_ratio', 'curr_ratio'),  # Current ratio itself
        ('quick_ratio', 'quick_ratio'),  # Quick ratio itself
        ('cash_ratio', 'cash_ratio'),  # Cash ratio itself
        
        # Leverage ratios
        ('debt_assets', 'debt_assets'),  # Debt-to-assets ratio itself
        ('debt_capital', 'debt_capital'),  # Debt-to-capital ratio itself
        ('intcov', 'intcov'),  # Interest coverage ratio itself
        
        # Efficiency ratios
        ('at_turn', 'at_turn'),  # Asset turnover itself
        ('inv_turn', 'inv_turn'),  # Inventory turnover itself
        ('rect_turn', 'rect_turn'),  # Receivables turnover itself
        
        # Valuation ratios
        ('pe_exi', 'pe_exi'),  # Price-to-earnings ratio itself
        ('pe_inc', 'pe_inc'),  # Price-to-earnings ratio including extraordinary items itself
        ('ps', 'ps'),  # Price-to-sales ratio itself
        ('pb', 'pb'),  # Price-to-book ratio itself
        ('pcf', 'pcf'),  # Price-to-cash-flow ratio itself
        
        # Custom ratios
        ('roa', 'roa'),  # ROA/ROE
        ('roe', 'roa'),
        ('curr_ratio', 'quick_ratio'),  # Current ratio / quick ratio
        ('debt_assets', 'debt_capital'),  # Debt-to-assets ratio / debt-to-capital ratio
    ]
    
    # Add financial ratios
    for numerator, denominator in tqdm(financial_ratios, desc="[FIN_RATIO_FEAT] Financial ratios"):
        if numerator in out.columns and denominator in out.columns:
            if numerator == denominator:
                # Single-feature ratios preserve the original value
                continue
            else:
                feature_name = f"{numerator}_div_{denominator}"
                out[feature_name] = np.where(
                    out[denominator] != 0,
                    out[numerator] / (out[denominator] + 1e-8),
                    0
                )
                new_features.append(feature_name)
    
    print(f"[FIN_RATIO_FEAT] Added {len(new_features)} financial ratio features")
    return out


def add_market_ratio_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Add market ratio features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    
    Returns:
        DataFrame with market ratio features added.
    """
    out = df.copy()
    print(f"[MARKET_RATIO_FEAT] Adding market ratio features...")
    
    new_features = []
    
    # Define market ratio pairs
    market_ratios = [
        # Price-related ratios
        ('price', 'price'),  # Price itself
        ('VOL', 'DOLLARVOL'),  # Trading volume / dollar volume
        ('RET', 'VOL'),  # Return / trading volume
        ('RET', 'DOLLARVOL'),  # Return / dollar volume
        
        # Risk-related ratios
        ('Beta', 'IdioVolCAPM'),  # Beta / idiosyncratic volatility
        ('VOL', 'ret_sd_12'),  # Trading volume / annual volatility
        ('VOL', 'ret_sd_6'),  # Trading volume / six-month volatility
        ('VOL', 'ret_sd_3'),  # Trading volume / quarterly volatility
        
        # Momentum-related ratios
        ('Mom12m', 'VOL'),  # Twelve-month momentum / trading volume
        ('Mom12m', 'ret_sd_12'),  # Twelve-month momentum / annual volatility
        ('STreversal', 'VOL'),  # Short-term reversal / trading volume
        
        # Liquidity-related ratios
        ('Illiquidity', 'VOL'),  # Illiquidity / trading volume
        ('Illiquidity', 'DOLLARVOL'),  # Illiquidity / dollar volume
        ('Size', 'VOL'),  # Market capitalization / trading volume
        ('Size', 'DOLLARVOL'),  # Market capitalization / dollar volume
    ]
    
    # Add market ratios
    for numerator, denominator in tqdm(market_ratios, desc="[MARKET_RATIO_FEAT] Market ratios"):
        if numerator in out.columns and denominator in out.columns:
            if numerator == denominator:
                # Single-feature ratios preserve the original value
                continue
            else:
                feature_name = f"{numerator}_div_{denominator}"
                out[feature_name] = np.where(
                    out[denominator] != 0,
                    out[numerator] / (out[denominator] + 1e-8),
                    0
                )
                new_features.append(feature_name)
    
    print(f"[MARKET_RATIO_FEAT] Added {len(new_features)} market ratio features")
    return out


def add_macro_ratio_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Add macroeconomic ratio features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
    
    Returns:
        DataFrame with macroeconomic ratio features added.
    """
    out = df.copy()
    print(f"[MACRO_RATIO_FEAT] Adding macro ratio features...")
    
    new_features = []
    
    # Define macroeconomic ratio pairs
    macro_ratios = [
        # Economic growth ratios
        ('GDP_g', 'con_g'),  # GDP growth / consumption growth
        ('GDP_g', 'ipt_g'),  # GDP growth / industrial production growth
        ('con_g', 'ipt_g'),  # Consumption growth / industrial production growth
        
        # Inflation ratios
        ('INFLATION', 'GDP_g'),  # Inflation / GDP growth
        ('INFLATION', 'con_g'),  # Inflation / consumption growth
        ('INFLATION', 'ipt_g'),  # Inflation / industrial production growth
        
        # Employment ratios
        ('Unemployment', 'GDP_g'),  # Unemployment / GDP growth
        ('Unemployment', 'INFLATION'),  # Unemployment / inflation
    ]
    
    # Add macroeconomic ratios
    for numerator, denominator in tqdm(macro_ratios, desc="[MACRO_RATIO_FEAT] Macro ratios"):
        if numerator in out.columns and denominator in out.columns:
            feature_name = f"{numerator}_div_{denominator}"
            out[feature_name] = np.where(
                out[denominator] != 0,
                out[numerator] / (out[denominator] + 1e-8),
                0
            )
            new_features.append(feature_name)
    
    print(f"[MACRO_RATIO_FEAT] Added {len(new_features)} macro ratio features")
    return out


def apply_interaction_features(df: pd.DataFrame, feature_cols: List[str],
                             config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Apply all interaction feature engineering steps.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with interaction features added.
    """
    if config is None:
        config = {
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
    
    print("=" * 60)
    print("APPLYING INTERACTION FEATURE ENGINEERING")
    print("=" * 60)
    
    # Apply the interaction feature generators
    if config.get('ratio_features', {}).get('enabled', True):
        df = add_ratio_features(
            df, feature_cols,
            ratio_pairs=config['ratio_features']['ratio_pairs']
        )
    
    if config.get('polynomial_features', {}).get('enabled', True):
        df = add_polynomial_features(
            df, feature_cols,
            degree=config['polynomial_features']['degree'],
            selected_features=config['polynomial_features'].get('selected_features')
        )
    
    if config.get('cross_features', {}).get('enabled', True):
        df = add_cross_features(
            df, feature_cols,
            feature_groups=config['cross_features']['feature_groups']
        )
    
    if config.get('financial_ratio_features', {}).get('enabled', True):
        df = add_financial_ratio_features(df, feature_cols)
    
    if config.get('market_ratio_features', {}).get('enabled', True):
        df = add_market_ratio_features(df, feature_cols)
    
    if config.get('macro_ratio_features', {}).get('enabled', True):
        df = add_macro_ratio_features(df, feature_cols)
    
    print("=" * 60)
    print("INTERACTION FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    
    return df
