import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import warnings


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).
    
    Args:
        prices: Price series.
        period: Calculation period.
    
    Returns:
        RSI series.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Compute MACD indicators.
    
    Args:
        prices: Price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period.
    
    Returns:
        Dictionary containing MACD, signal line, and histogram.
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Args:
        prices: Price series.
        period: Calculation period.
        std: Standard deviation multiplier.
    
    Returns:
        Dictionary containing upper, middle, and lower bands.
    """
    middle = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Compute the Stochastic oscillator.
    
    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        k_period: K period.
        d_period: D period.
    
    Returns:
        Dictionary containing K and D values.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'k': k_percent,
        'd': d_percent
    }


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Williams %R.
    
    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        period: Calculation period.
    
    Returns:
        Williams %R series.
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-8))
    return williams_r


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute the Commodity Channel Index (CCI).
    
    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        period: Calculation period.
    
    Returns:
        CCI series.
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma) / (0.015 * mad + 1e-8)
    return cci


def add_technical_indicators(df: pd.DataFrame, feature_cols: List[str],
                           price_features: List[str] = ['price'],
                           volume_features: List[str] = ['VOL'],
                           config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Add technical indicator features.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        price_features: List of price-related features.
        volume_features: List of volume-related features.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with technical indicator features added.
    """
    out = df.copy()
    print(f"[TECH_IND] Adding technical indicators...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    
    # Default configuration
    if config is None:
        config = {
            'RSI': {'period': 14, 'features': price_features},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9, 'features': price_features},
            'BOLLINGER': {'period': 20, 'std': 2, 'features': price_features},
            'STOCHASTIC': {'k_period': 14, 'd_period': 3, 'features': price_features},
            'WILLIAMS_R': {'period': 14, 'features': price_features},
            'CCI': {'period': 20, 'features': price_features}
        }
    
    for indicator, params in tqdm(config.items(), desc="[TECH_IND] Indicators"):
        if not params.get('enabled', True):
            continue
            
        target_features = params.get('features', price_features)
        
        for feature in target_features:
            if feature not in out.columns:
                print(f"[TECH_IND] Warning: {feature} not found in columns")
                continue
            
            if indicator == 'RSI':
                period = params.get('period', 14)
                feature_name = f"{feature}_rsi_{period}"
                out[feature_name] = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_rsi(x, period)
                )
                new_features.append(feature_name)
                
            elif indicator == 'MACD':
                fast = params.get('fast', 12)
                slow = params.get('slow', 26)
                signal = params.get('signal', 9)
                
                # Compute MACD
                macd_result = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_macd(x, fast, slow, signal)['macd']
                )
                signal_result = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_macd(x, fast, slow, signal)['signal']
                )
                histogram_result = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_macd(x, fast, slow, signal)['histogram']
                )
                
                out[f"{feature}_macd_{fast}_{slow}"] = macd_result
                out[f"{feature}_macd_signal_{signal}"] = signal_result
                out[f"{feature}_macd_histogram"] = histogram_result
                
                new_features.extend([
                    f"{feature}_macd_{fast}_{slow}",
                    f"{feature}_macd_signal_{signal}",
                    f"{feature}_macd_histogram"
                ])
                
            elif indicator == 'BOLLINGER':
                period = params.get('period', 20)
                std = params.get('std', 2)
                
                # Compute Bollinger Bands
                upper = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_bollinger_bands(x, period, std)['upper']
                )
                middle = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_bollinger_bands(x, period, std)['middle']
                )
                lower = out.groupby(group_keys)[feature].transform(
                    lambda x: calculate_bollinger_bands(x, period, std)['lower']
                )
                
                out[f"{feature}_bb_upper_{period}"] = upper
                out[f"{feature}_bb_middle_{period}"] = middle
                out[f"{feature}_bb_lower_{period}"] = lower
                out[f"{feature}_bb_width_{period}"] = (upper - lower) / (middle + 1e-8)
                out[f"{feature}_bb_position_{period}"] = (out[feature] - lower) / (upper - lower + 1e-8)
                
                new_features.extend([
                    f"{feature}_bb_upper_{period}",
                    f"{feature}_bb_middle_{period}",
                    f"{feature}_bb_lower_{period}",
                    f"{feature}_bb_width_{period}",
                    f"{feature}_bb_position_{period}"
                ])
                
            elif indicator == 'STOCHASTIC':
                k_period = params.get('k_period', 14)
                d_period = params.get('d_period', 3)
                
                # The Stochastic oscillator requires high, low, and close
                # Simplify by using the price feature as close
                if feature in ['price']:
                    # Use price as close and create simplified high and low values
                    high_feature = feature  # Simplification: use price as high
                    low_feature = feature   # Simplification: use price as low
                    
                    k_result = out.groupby(group_keys)[feature].transform(
                        lambda x: calculate_stochastic(x, x, x, k_period, d_period)['k']
                    )
                    d_result = out.groupby(group_keys)[feature].transform(
                        lambda x: calculate_stochastic(x, x, x, k_period, d_period)['d']
                    )
                    
                    out[f"{feature}_stoch_k_{k_period}"] = k_result
                    out[f"{feature}_stoch_d_{d_period}"] = d_result
                    
                    new_features.extend([
                        f"{feature}_stoch_k_{k_period}",
                        f"{feature}_stoch_d_{d_period}"
                    ])
                    
            elif indicator == 'WILLIAMS_R':
                period = params.get('period', 14)
                
                if feature in ['price']:
                    # Simplified calculation
                    williams_result = out.groupby(group_keys)[feature].transform(
                        lambda x: calculate_williams_r(x, x, x, period)
                    )
                    out[f"{feature}_williams_r_{period}"] = williams_result
                    new_features.append(f"{feature}_williams_r_{period}")
                    
            elif indicator == 'CCI':
                period = params.get('period', 20)
                
                if feature in ['price']:
                    # Simplified calculation
                    cci_result = out.groupby(group_keys)[feature].transform(
                        lambda x: calculate_cci(x, x, x, period)
                    )
                    out[f"{feature}_cci_{period}"] = cci_result
                    new_features.append(f"{feature}_cci_{period}")
    
    print(f"[TECH_IND] Added {len(new_features)} technical indicator features")
    return out


def add_volume_indicators(df: pd.DataFrame, feature_cols: List[str],
                         volume_features: List[str] = ['VOL', 'DOLLARVOL']) -> pd.DataFrame:
    """
    Add volume indicators.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        volume_features: List of volume features.
    
    Returns:
        DataFrame with volume indicators added.
    """
    out = df.copy()
    print(f"[VOL_IND] Adding volume indicators...")
    
    # Ensure the data is sorted correctly
    if 'segment_id' in out.columns:
        sort_keys = ['accid', 'permno', 'segment_id', 'mdate']
        group_keys = ['accid', 'permno', 'segment_id']
    else:
        sort_keys = ['accid', 'permno', 'mdate']
        group_keys = ['accid', 'permno']
    
    out = out.sort_values(sort_keys).reset_index(drop=True)
    
    new_features = []
    
    for volume_feature in volume_features:
        if volume_feature not in out.columns:
            continue
            
        # Volume moving averages
        for window in [5, 10, 20]:
            feature_name = f"{volume_feature}_ma_{window}"
            out[feature_name] = out.groupby(group_keys)[volume_feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            new_features.append(feature_name)
        
        # Volume ratios
        for window in [5, 10, 20]:
            ma = out.groupby(group_keys)[volume_feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            feature_name = f"{volume_feature}_ratio_{window}"
            out[feature_name] = out[volume_feature] / (ma + 1e-8)
            new_features.append(feature_name)
    
    print(f"[VOL_IND] Added {len(new_features)} volume indicator features")
    return out


def apply_technical_indicators(df: pd.DataFrame, feature_cols: List[str],
                             config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Apply all technical indicator feature engineering steps.
    
    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with technical indicator features added.
    """
    if config is None:
        config = {
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
    
    print("=" * 60)
    print("APPLYING TECHNICAL INDICATORS FEATURE ENGINEERING")
    print("=" * 60)
    
    # Apply technical indicators
    if config.get('technical_indicators', {}).get('enabled', True):
        df = add_technical_indicators(
            df, feature_cols,
            price_features=['price'],
            volume_features=['VOL', 'DOLLARVOL'],
            config=config['technical_indicators']
        )
    
    # Apply volume indicators
    if config.get('volume_indicators', {}).get('enabled', True):
        df = add_volume_indicators(
            df, feature_cols,
            volume_features=config['volume_indicators'].get('volume_features', ['VOL', 'DOLLARVOL'])
        )
    
    print("=" * 60)
    print("TECHNICAL INDICATORS FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    
    return df
