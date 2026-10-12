# Advanced Feature Overview (Monthly Data)

This document describes four feature engineering modules, their output names and meanings, and the recommended processing order. All window sizes and periods refer to months.

## Pipeline order

1. Temporal features
2. Technical indicators
3. Statistical features
4. Interaction and combination features

Construct time and price/volume derivatives first, then compute statistical aggregates, and finally build interactions. This order helps limit noise accumulation and dimensionality growth from early combinations.

## Temporal features

- **Cyclical encoding:** `month_sin/cos`, `quarter_sin/cos`, `year_sin/cos`. Sine and cosine map calendar periods onto a circle to represent seasonality and cycles; year encoding uses a ten-year cycle relative to the first year in the data.
- **Lags:** `{feature}_lag_{1,3,6,12}`. Values from previous months represent temporal dependencies through shifts within each `accid, permno` group, including `segment_id` when present.
- **Trends:** `{feature}_trend_slope_{3,6,12}`, `{feature}_trend_r2_{3,6,12}`. Linear slopes and trend strength (R squared) within recent windows.
- **Seasonal indicators:** `is_q{1-4}`, `is_{jan..dec}`, and `month/quarter/year`. Calendar indicators for seasonal and month effects.
- **Time distances:** `months_since_start`, `months_since_group_start`, `months_to_group_end`. Elapsed months from the global/group start and months remaining until the group ends.

Lead features are implemented but disabled by default because they can leak future data. Only enable them when values would be available at prediction time, with release lags aligned explicitly. `months_to_group_end` also uses information about future group endpoints and requires evaluation of its availability.

## Technical indicators

- **RSI:** `price_rsi_14`, the Relative Strength Index comparing upward and downward price movements.
- **MACD:** `price_macd_{12}_{26}`, `price_macd_signal_9`, `price_macd_histogram`. The difference between 12- and 26-month EMAs, its 9-month signal line, and their difference describe momentum and turning points.
- **Bollinger Bands:** `price_bb_upper_20`, `price_bb_middle_20`, `price_bb_lower_20`, `price_bb_width_20`, `price_bb_position_20`. Width is `(upper-lower)/middle`; position measures the price's relative location within the bands.
- **Stochastic oscillator:** `price_stoch_k_14`, `price_stoch_d_3`.
- **Williams %R:** `price_williams_r_14`.
- **CCI:** `price_cci_20`, the Commodity Channel Index.
- **Volume indicators:** `VOL_ma_{5,10,20}` for moving averages and `VOL_ratio_{5,10,20}` for current volume divided by its moving average. The same naming applies to `DOLLARVOL`.

The current configuration uses monthly `price` data. Indicators requiring high, low, and close use price-based approximations. Replace these approximations with actual monthly `high/low/close` inputs when available.

## Statistical features

- **Rolling statistics:** `{feature}_roll_{3,6,12,24}_{mean,std,min,max,median,skew}`. Recent means, volatility, extremes, medians, and skewness describe the distribution and its stability.
- **Quantiles:** `{feature}_roll_{6,12,24}_q{10,25,75,90}`. Rolling quantile levels provide robust measures of relative position.
- **Ranks:** `{feature}_roll_{6,12,24}_{percentile,zscore}`. Percentile ranks and standardized deviations within each window.
- **Changes:** `{feature}_change_{1,3,6,12}`, `{feature}_pct_change_{1,3,6,12}`. Absolute and percentage changes over the specified monthly periods.
- **Volatility:** `{feature}_vol_{6,12,24}`, `{feature}_cv_{6,12,24}`. Standard deviations and coefficients of variation (`std/mean`) quantify variability.

Rolling statistics, ranks, and differences are computed within `accid, permno` groups, also using `segment_id` when present, after monthly sorting. New columns are constructed in batches and combined with `pd.concat(axis=1)` to reduce DataFrame fragmentation.

## Interaction and combination features

- **Ratios:** `{A}_div_{B}`. Financial, market, and macroeconomic pairs include `roa/roe`, `curr_ratio/quick_ratio`, `VOL/DOLLARVOL`, `GDP_g/con_g`, and `INFLATION/GDP_g`.
- **Polynomials:** `{feature}_pow_2`. Squared terms capture nonlinear relationships; selecting a small number of columns limits dimensionality.
- **Cross products and differences:** `{f1}_x_{f2}`, `{f1}_minus_{f2}`. Pairwise products and differences represent interactions and relative magnitudes.

## Integration and configuration

The neural pipeline uses `config/config_nn.py` to enable modules and configure their windows, statistics, indicators, and feature selections:

| Toggle | Configuration dictionary |
| --- | --- |
| `ENABLE_TEMPORAL_FEATURES` | `TEMPORAL_FEATURES_CONFIG` |
| `ENABLE_TECHNICAL_INDICATORS` | `TECHNICAL_INDICATORS_CONFIG` |
| `ENABLE_STATISTICAL_FEATURES` | `STATISTICAL_FEATURES_CONFIG` |
| `ENABLE_INTERACTION_FEATURES` | `INTERACTION_FEATURES_CONFIG` |

`main_nn.py` invokes these modules in the recommended order. The generators add columns to the DataFrame but do not currently extend the training `feature_cols` list; add the intended derived columns to that list before using them as model inputs. The advanced families are disabled by default. The current `main_lin.py` and `main_tr.py` do not invoke this advanced feature pipeline.

All periods assume monthly data. Price and volume inputs must match existing columns, currently `price`, `VOL`, and `DOLLARVOL`. Review prediction-time availability of every feature and the scope of preprocessing before using the outputs for evaluation.
