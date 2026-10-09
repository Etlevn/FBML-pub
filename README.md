# FBML: Features & Biases Machine Learning

**Predicting retail investor trading behavior with stock characteristics and behavioral biases.**

FBML is a behavioral-finance research project that turns monthly investor–stock panel data into a configurable machine-learning experiment system. It combines stock characteristics, behavioral-bias variables, and historical account states to study buy/sell decisions, comparing linear baselines, tree ensembles, and sequence neural networks under a rolling evaluation framework.

The public repository contains the Python modeling and experiment code. Investor-level data and original experiment records are supplied separately.

## Research questions

1. How much information do stock characteristics and behavioral-bias variables provide about subsequent retail trading behavior?
2. Do nonlinear models and historical sequence representations improve prediction relative to simpler baselines?
3. How do model choices change when accuracy, precision, recall, class-specific F1, and ROC AUC favor different architectures?

The historical study used approximately **10 years of monthly observations**, covering **72,000 retail investors**, **8,000 stocks**, and **92 stock-characteristic and behavioral-bias variables**, with a **36-month history window**. These figures describe the original study; the public code infers its feature set from the supplied data rather than enforcing a fixed 92-variable schema.

The research was informed by work on the “factor zoo” and “bias zoo,” including [Ghosh, Lu, Zhang, and Zhang, *A Tale of Two Zoos: Machine Learning Insights on Retail Investors*](https://abfer.org/component/edocman/main-annual-conference/a-tale-of-two-zoos-machine-learning-insights-on-retail-investors). FBML applies this research motivation to trading-behavior classification; it is an independent implementation, rather than a replication package for that paper.

## Research and engineering contributions

The author independently developed the computational pipeline under academic guidance: data preparation, supervised target construction, model implementations, configuration-driven experiments, rolling evaluation, hyperparameter experiments, feature importance, and result analysis. The advisor supplied the underlying investor data and some preconstructed stock-characteristic and behavioral-bias variables. The contribution is the modeling and experiment system; it does not claim authorship of the source data, every input factor, or the original research idea.

The implementation provides:

- **Shared data preparation:** monthly source alignment, configurable buy/sell labels, missing-value handling, and investor–stock sequence construction.
- **Three model families:** linear baselines; Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, and CatBoost; and FNN, CNN, RNN, LSTM, BiLSTM, GRU, and Transformer classifiers.
- **Imbalance-aware experiments:** optional class/sample weighting and neural losses including cross entropy, binary cross entropy, and Focal Loss.
- **Configurable sequence training:** a default 36-month history, BiLSTM aggregation/head options, early stopping, and optional learning-rate, gradient-clipping, and weight-decay controls.
- **Evaluation and interpretation:** per-window metrics, batch-result merging, aggregate summaries, and model-specific feature importance exports.

The modular Python scripts are the entry points for all public experiments.

## Research results

The metrics below are taken directly from the original experiment workbook, `SW_S.xlsx`, and rounded to four decimal places. The cross-model benchmark uses its model-comparison worksheet; the BiLSTM configuration study uses its separate configuration worksheet. The workbook and private datasets are not distributed with this repository, and the experiments have **not been rerun from this public snapshot**.

Model names describe the architecture and recorded configuration instead of internal version numbers. **Balanced** expands the workbook's `-b` suffix and denotes class-imbalance handling; **weighted** expands its `-w` suffix. In the public implementation, balancing uses class weights where supported and inverse-class-frequency sample weights for models such as XGBoost. Exact historical hyperparameters require the corresponding run configuration.

### Neural models

| Model (configuration) | Accuracy | Precision₁ | Recall₁ | F1₁ | F1₀ | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BiLSTM (early stopping) | 0.8320 | 0.8043 | 0.7958 | 0.7875 | 0.8394 | 0.8870 |
| LSTM | 0.8196 | 0.7875 | 0.7857 | 0.7737 | 0.8265 | 0.8749 |
| GRU | 0.8179 | 0.7814 | 0.7822 | 0.7713 | 0.8254 | 0.8722 |
| FNN | 0.7911 | 0.7341 | 0.7907 | 0.7513 | 0.7923 | 0.8491 |
| CNN | 0.7877 | 0.7435 | 0.7852 | 0.7513 | 0.7801 | 0.8514 |
| RNN | 0.7653 | 0.7079 | 0.7869 | 0.7324 | 0.7556 | 0.8308 |
| Transformer | 0.7310 | 0.6870 | 0.7942 | 0.7163 | 0.6879 | 0.8052 |

For LSTM, GRU, FNN, CNN, RNN, and Transformer, the comparison shows the highest recorded positive-class F1 within each family in the model-comparison worksheet. That worksheet provides model labels and metrics but no detailed configuration column, so additional settings are not inferred from version numbers. BiLSTM uses the explicitly documented early-stopping configuration from the configuration worksheet.

### Tree models

| Model (configuration) | Accuracy | Precision₁ | Recall₁ | F1₁ | F1₀ | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest (balanced) | 0.8163 | 0.7415 | 0.8825 | 0.7994 | 0.8096 | 0.9250 |
| Random Forest | 0.8011 | 0.7281 | 0.8709 | 0.7805 | 0.7917 | 0.9158 |
| Decision Tree (balanced) | 0.7811 | 0.6946 | 0.8941 | 0.7733 | 0.7682 | 0.8142 |
| Extra Trees (balanced) | 0.7712 | 0.6891 | 0.8982 | 0.7703 | 0.7460 | 0.9278 |
| Decision Tree | 0.7842 | 0.7062 | 0.8513 | 0.7635 | 0.7788 | 0.8104 |
| Extra Trees | 0.7657 | 0.6978 | 0.8597 | 0.7530 | 0.7454 | 0.9169 |
| XGBoost (balanced) | 0.7142 | 0.6491 | 0.8488 | 0.7221 | 0.6743 | 0.8246 |
| XGBoost | 0.7256 | 0.6719 | 0.8217 | 0.7208 | 0.6915 | 0.8172 |
| Histogram Gradient Boosting (balanced) | 0.7059 | 0.6493 | 0.8354 | 0.7149 | 0.6613 | 0.8137 |
| LightGBM (balanced) | 0.7003 | 0.6467 | 0.8397 | 0.7138 | 0.6535 | 0.8173 |
| LightGBM | 0.7194 | 0.6710 | 0.8106 | 0.7136 | 0.6835 | 0.8081 |
| CatBoost | 0.7191 | 0.6612 | 0.8199 | 0.7131 | 0.6840 | 0.8135 |
| CatBoost (weighted) | 0.7014 | 0.6316 | 0.8500 | 0.7126 | 0.6584 | 0.8289 |
| Histogram Gradient Boosting | 0.7220 | 0.6737 | 0.8069 | 0.7118 | 0.6878 | 0.8078 |
| Gradient Boosting | 0.6995 | 0.6514 | 0.8096 | 0.6983 | 0.6586 | 0.7956 |
| Gradient Boosting (weighted) | 0.6795 | 0.6187 | 0.8353 | 0.6958 | 0.6240 | 0.8035 |

### Linear baselines

| Model (configuration) | Accuracy | Precision₁ | Recall₁ | F1₁ | F1₀ | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge (balanced) | 0.5241 | 0.4514 | 0.7526 | 0.5236 | 0.3433 | 0.6778 |
| SGD (hinge loss, balanced) | 0.5241 | 0.4514 | 0.7526 | 0.5236 | 0.3433 | 0.6778 |
| OLS (weighted) | 0.4877 | 0.5233 | 0.7564 | 0.4806 | 0.2490 | 0.5216 |
| Ridge | 0.5795 | 0.4518 | 0.6214 | 0.4799 | 0.5110 | 0.6775 |
| SGD (modified Huber loss, balanced) | 0.4898 | 0.4287 | 0.4696 | 0.4113 | 0.5127 | 0.4972 |
| SGD (log loss, balanced) | 0.4891 | 0.4272 | 0.4618 | 0.4071 | 0.5131 | 0.4959 |
| SGD (hinge loss) | 0.5250 | 0.4353 | 0.3735 | 0.3729 | 0.5729 | 0.5021 |
| SGD (log loss) | 0.5224 | 0.4297 | 0.3589 | 0.3626 | 0.5794 | 0.5040 |
| SGD (modified Huber loss) | 0.5215 | 0.4266 | 0.3457 | 0.3540 | 0.5836 | 0.5011 |
| Logistic Regression (balanced) | 0.5439 | 0.4181 | 0.2901 | 0.2481 | 0.6470 | 0.5822 |
| Logistic Regression | 0.5673 | 0.4141 | 0.1464 | 0.1601 | 0.6926 | 0.5754 |
| OLS | 0.5640 | 0.6215 | 0.2024 | 0.1476 | 0.5869 | 0.5193 |

### BiLSTM configuration study

The configuration worksheet distinguishes the following experiments. The parenthetical label identifies the setting recorded for each run; it does not imply that every other hyperparameter was held constant.

| Model (configuration) | Accuracy | Precision₁ | Recall₁ | F1₁ | F1₀ | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BiLSTM (default) | 0.8225 | 0.7902 | 0.7881 | 0.7775 | 0.8302 | 0.8760 |
| BiLSTM (early stopping) | 0.8320 | 0.8043 | 0.7958 | 0.7875 | 0.8394 | 0.8870 |
| BiLSTM (attention) | 0.8209 | 0.7869 | 0.7877 | 0.7759 | 0.8278 | 0.8751 |
| BiLSTM (`fce`) | 0.8077 | 0.7609 | 0.8153 | 0.7798 | 0.8117 | 0.8718 |
| BiLSTM (focal loss) | 0.8253 | 0.7947 | 0.7873 | 0.7794 | 0.8340 | 0.8793 |
| BiLSTM (cosine annealing) | 0.8233 | 0.7917 | 0.7879 | 0.7782 | 0.8310 | 0.8763 |
| BiLSTM (gradient clipping) | 0.8223 | 0.7901 | 0.7880 | 0.7775 | 0.8300 | 0.8760 |
| BiLSTM (weight decay) | 0.8238 | 0.7917 | 0.7872 | 0.7783 | 0.8325 | 0.8765 |
| BiLSTM (combined settings: `ALL`) | 0.8316 | 0.8057 | 0.7884 | 0.7851 | 0.8408 | 0.8847 |
| BiLSTM (sequence aggregation: last) | 0.8287 | 0.8005 | 0.7874 | 0.7818 | 0.8381 | 0.8813 |
| BiLSTM (sequence aggregation: mean) | 0.8069 | 0.7784 | 0.7845 | 0.7652 | 0.8091 | 0.8659 |
| BiLSTM (sequence aggregation: max) | 0.8112 | 0.7802 | 0.7897 | 0.7697 | 0.8134 | 0.8702 |
| BiLSTM (deep classification head) | 0.8246 | 0.7871 | 0.7871 | 0.7784 | 0.8328 | 0.8782 |
| BiLSTM (classification-head batch normalization) | 0.8223 | 0.7919 | 0.7891 | 0.7779 | 0.8289 | 0.8773 |

Configuration names correspond to the controls in `config/config_nn.py` and their implementations in `main_nn.py` and `model/model_nn.py`: early stopping, attention, Focal Loss, cosine annealing, gradient clipping, weight decay, sequence aggregation, a deep classification head, and head batch normalization. The historical `default` label does not identify the current checked-in defaults.

The workbook also contains the labels `fce` and `ALL`. The available training code supports `ce`, `bce`, and `focal` losses, but has no `fce` option or single `ALL` switch. Those two labels are retained as recorded; the specific loss behind `fce` and the precise combination behind `ALL` cannot be uniquely recovered from the available code and table.

### Findings and model selection

- **BiLSTM (early stopping)** was the study's selected model for its combination of accuracy, positive-class precision, and performance across both classes: **0.8320 accuracy**, **0.8043 positive-class precision**, **0.7875 positive-class F1**, and **0.8870 AUC**. The arithmetic mean of its two reported class F1 values is **0.8135**.
- **Random Forest (balanced)** remains a competitive benchmark. Its positive-class recall (**0.8825**), positive-class F1 (**0.7994**), and AUC (**0.9250**) exceed the selected BiLSTM's values. Its mean of the two class F1 values is **0.8045**; model selection therefore depends on the task's precision/recall priorities.
- **Extra Trees (balanced)** has the highest reported AUC among the listed tree models (**0.9278**), while Random Forest (balanced) has the highest positive-class F1 in that group. The two metrics favor different models.
- **Balancing has metric-specific tradeoffs.** For XGBoost, the balanced run has higher recall and AUC than the standard run, while accuracy and precision are lower. The workbook does not establish a universal gain from balancing.
- **The BiLSTM study records differences across training settings.** Early stopping has the highest reported accuracy and positive-class F1 in its configuration table; the `ALL` run has slightly higher positive-class precision and class-0 F1. These comparisons do not establish statistical significance or improvement across every time window.

Subscripts denote class 1 and class 0. The historical action mapping, sample/window comparability, and metric aggregation still require reconciliation with the original runs. Reported F1 values are preserved as supplied rather than recomputed from rounded precision and recall. The current neural defaults combine settings and should not be treated as an exact reproduction of any historical row. See [Evaluation considerations](#evaluation-considerations) for implementation boundaries.

## Repository layout

```text
FBML-pub/
├── main_nn.py                 # Neural network training and evaluation
├── main_lin.py                # Linear model training and evaluation
├── main_tr.py                 # Tree model training, evaluation, and tuning
├── config/
│   ├── config_nn.py           # Neural network experiment settings
│   ├── config_lin.py          # Linear model experiment settings
│   └── config_tr.py           # Tree model experiment settings
├── model/
│   ├── model_nn.py            # Neural architectures and shared model constants
│   ├── model_lin.py           # Linear model factory and prediction utilities
│   └── model_tr.py            # Tree model factory and hyperparameter searches
├── modules/                   # Data preparation, features, losses, and metrics
├── importance/                # Model-specific feature importance methods
├── merge.py                   # Merge window-specific batch outputs
├── summary.py                 # Summarize merged batch outputs
├── FEATURES_NEW.md            # Advanced feature definitions and naming
├── PUBLICATION.md             # Source release and privacy notes
├── scripts/check_publication.py # Publication scan and source export
└── requirements.txt           # Python dependencies
```

Results are written to `out/` by default. Data, virtual environments, and generated model artifacts are not included in the repository.

## Setup

From the repository root, use a Python environment compatible with the packages in `requirements.txt`. Full experiments require the external datasets described below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Optuna is optional and is not listed in `requirements.txt`; install it separately if using Optuna searches:

```bash
python -m pip install optuna
```

Package versions are specified as lower bounds rather than a reproducible lockfile.

## Data configuration

Supply datasets separately in the ignored `data/` directory, or set `FBML_DATA_ROOT` to a private data directory before running any pipeline:

```bash
export FBML_DATA_ROOT="/path/to/private/datasets"
```

All three pipelines use this environment variable and default to the repository's `data/` directory. An `.env` file is not loaded automatically. Edit `FILES` in the matching configuration if your filenames or formats differ. Account-level datasets must not be committed or redistributed without authorization.

The default sources are:

| File | Format | Date column mapping |
| --- | --- | --- |
| `master_file.dta` | Stata | Existing `mdate` |
| `associative_memory.sas7bdat` | SAS | `public_date` → `mdate` |
| `AP_factors.dta` | Stata | `time_avail_m` → `mdate` |
| `stock_ret_controls.dta` | Stata | Existing `mdate` |

The loader also accepts CSV and Parquet entries. It normalizes dates to monthly values and left-merges sources on `permno` and `mdate`, using the first source as the base table.

The merged data must contain `accid`, `permno`, `mdate`, and the selected action column (`buy` or `sell`). With `TASK = 'bin'`, the target is 1 when the chosen action equals 1 and 0 otherwise. When both action columns exist, `NAN = True` fills missing action values with zero; `NAN = False` removes rows where both are missing.

Target preparation also supports `TASK = 'tri'` with labels -1, 0, and 1. The current neural heads and exported metrics are designed for binary classification, so the modular pipeline should use `TASK = 'bin'` unless those components are adapted for a multiclass experiment.

## Run an experiment

Run commands from the repository root after editing the matching configuration:

```bash
python main_nn.py
python main_lin.py
python main_tr.py
```

Each command runs its own pipeline. Experiment settings are printed before processing starts.

| Pipeline | Model selection setting | Available model types |
| --- | --- | --- |
| Neural network | `SEQ_MODEL_TYPE` | `fnn`, `rnn`, `lstm`, `bilstm`, `gru`, `transformer`, `cnn` |
| Linear | `LINEAR_MODEL_TYPE` | `ols`, `ridge`, `logistic`, `sgd_hinge`, `sgd_log`, `sgd_modified_huber` |
| Tree | `TREE_MODEL_TYPE` | `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `xgboost`, `lightgbm`, `catboost` |

`MODEL_LABEL` controls experiment labels and output filenames. Use a distinct label for each run: fold CSVs are appended to existing files, so reusing a label can mix results from different experiments.

### Windows and sequences

`T` is the sequence length in months (default: 36). For window index `i`, training rows cover months `max(0, i-T+1)` through `i`, and test rows cover `max(0, i-T+2)` through `i+1`. Sequences are grouped by account and stock, with `segment_id` included when enabled. Short sequences are padded at the front with zeros; the label comes from the final row of each sequence.

`USE_CUMULATIVE_TRAINING` builds training examples at successive months within the supplied training frame. The main scripts still obtain that frame from the window splitter; this option does not itself expand the frame to the entire history.

Set `BATCH_MODE = True` and provide zero-based, inclusive `START_WINDOW` and `END_WINDOW` values to process a subset of windows. `END_WINDOW` must be less than the total window count. When `BATCH_MODE = False`, both bounds are ignored.

### Neural network settings

The default neural experiment uses BiLSTM, Focal Loss, early stopping, and `concat_last_mean` sequence aggregation. Other settings include:

- `LOSS_TYPE`: cross entropy (`ce`), binary cross entropy (`bce`), or Focal Loss (`focal`).
- `BCE_POS_WEIGHT_MODE`, `FOCAL_ALPHA`, and `FOCAL_GAMMA`: loss-specific parameters.
- `ENABLE_DEEP_HEAD`, `HEAD_HIDDEN_SIZE`, `HEAD_NUM_LAYERS`, and `ENABLE_HEAD_BATCH_NORM`: BiLSTM classification head options.
- `MAX_EPOCHS`, `LR`, early stopping, cosine annealing, gradient clipping, and weight decay settings.
- `ENABLE_STANDARDIZATION` and `STANDARDIZE_MODE`: rolling, global, or per-window training-set scaling.

Shared architecture constants, training/evaluation batch sizes, and FNN/CNN/Transformer-specific constants live in `model/model_nn.py`. Transformer positional encoding must accommodate `T`. Device selection uses CUDA when available and CPU otherwise.

The neural pipeline optionally applies advanced features in this order: temporal features, technical indicators, statistical features, and interactions. These families are disabled by default. Their generators add DataFrame columns, but the current pipeline does not extend the model input `feature_cols` list to include them; that integration must be completed before training on the derived features. The linear and tree entry points currently use basic preparation and layered missing-value filling rather than invoking these advanced modules. See [FEATURES_NEW.md](FEATURES_NEW.md) for definitions.

### Tree hyperparameter optimization

Set `ENABLE_HYPERPARAMETER_OPTIMIZATION = True` in `config/config_tr.py` to run optimization on representative windows. This mode prints results and exits before the normal rolling evaluation loop.

- `HPO_METHOD`: `optuna` or `random` (RandomizedSearchCV).
- `HPO_N_TRIALS`, `HPO_CV_FOLDS`, and `HPO_SCORING`: search budget, TimeSeriesSplit folds, and a scikit-learn scoring name such as `f1`, `roc_auc`, or `average_precision`.
- `HPO_REPRESENTATIVE_SELECTION`: `auto`, `manual`, or `first_n`; configure the corresponding window indices or count.
- `HPO_VALIDATE_STABILITY`: compare numeric parameter variation across representative windows.

Built-in search spaces currently exist for Random Forest and Extra Trees; the default LightGBM configuration has no search space. The Random Forest factory forwards optimized parameters, while the Extra Trees factory currently ignores keyword overrides. Use Random Forest for the implemented tuning path until that factory is extended.

Optimization results are not automatically applied to normal training. Although the printed instructions refer to copying parameters into configuration, normal training does not currently read a best-parameter dictionary from `config/config_tr.py`. To use the results, wire the selected parameters into model creation or update the relevant factory defaults, then disable optimization mode. Optuna pruners are configured, but the objective currently does not report intermediate scores or request pruning.

## Results and feature importance

For binary experiments, fold exports include accuracy, precision, recall, F1 for both classes, and ROC AUC. Normal non-batch runs also export mean metrics across folds.

Feature importance methods vary by pipeline:

| Pipeline | Methods |
| --- | --- |
| Neural network | Gradient, DeepSHAP, Integrated Gradients |
| Linear | Absolute coefficients, permutation, SHAP |
| Tree | Built-in importance, tree-specific importance, permutation, SHAP |

Set `COMPUTE_IMPORTANCE`, `IMPORTANCE_METHODS`, and the method-specific options in the matching configuration. Linear and tree importance can be aggregated across time steps using `AGGREGATE_TEMPORAL_IMPORTANCE`. Optional summaries combine results across methods.

Typical filenames under `out/` are:

```text
<MODEL_LABEL>_fold.csv
<MODEL_LABEL>_sum.csv
<MODEL_LABEL>_imp_<method>_fold.csv
<MODEL_LABEL>_imp_<method>_sum.csv
<MODEL_LABEL>_imp_sum.csv
```

Batch fold files add a window suffix, for example `<MODEL_LABEL>_fold_0-10.csv`. After completing the batches, set the same `MODEL_LABEL` in both utility scripts and run:

```bash
python merge.py
python summary.py
```

`merge.py` combines batch metric and importance fold files; `summary.py` produces aggregate summaries from the merged files. Inspect the results before reusing files from overlapping or repeated batches.

## Evaluation considerations

Monthly ordering and training-set scaling do not by themselves guarantee that every preprocessing step avoids future information. Layered filling includes group means, and several feature transforms are applied to the full prepared dataset before window splitting. Global standardization, lead features, and `months_to_group_end` can expose future information. Review feature availability and preprocessing boundaries before interpreting results as an out-of-sample backtest.

Tree tuning applies TimeSeriesSplit to flattened, grouped sequence samples. Those splits follow sample order rather than explicit calendar folds; review their chronological meaning for your dataset. Early stopping in the neural pipeline uses the current test loader for monitoring, which can also affect evaluation independence.

Current summary exports take an unweighted arithmetic mean of each metric across windows. An averaged F1 need not equal the harmonic mean of averaged precision and recall. This code behavior does not establish the aggregation settings used in the historical workbook.

A BiLSTM encoding both directions within an available historical window does not itself use future observations; the critical checks are window boundaries, label alignment, and when each feature became available.

Full training requires the external datasets and installed dependencies. Source and syntax checks do not establish predictive performance or validate the experimental design.

This repository is intended for research and educational use. No standalone license file is currently included.

## Preparing a public source release

Run `python scripts/check_publication.py` to check the current source files for common secrets, personal paths, private artifacts, and excluded notebook files. Run `python scripts/check_publication.py --export` to produce `out/FBML-public-source.zip` from the checked files. The archive excludes Git history, local environments, private data, and generated results.

This check is a heuristic and does not establish data redistribution rights. Existing Git history can retain removed content and author emails; a clean working tree does not sanitize earlier commits or other branches. Use the clean archive for a new public repository, or sanitize every branch and tag before changing the existing repository's visibility.
