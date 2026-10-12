# FBML: Features & Biases Machine Learning

FBML predicts retail investors' future trades from stock characteristics and behavioral-bias variables. The question is how much those two kinds of information help, and whether prediction improves as models move from linear baselines to nonlinear and sequential models.

The supplied panel is about ten years of monthly investor–stock data, covering about 72,000 retail investors, 8,000 stocks, and 92 characteristic and bias variables. This repository is the training system for that study: shared preprocessing and labels, linear and tree baselines, neural and sequence models, rolling time windows, and comparable classification metrics.

Two comparisons organize the experiments:

1. How well do stock characteristics and investor behavioral biases predict later retail trading?
2. Do nonlinear machine-learning and sequential deep-learning models improve on linear methods under the same time split?

The modular scripts are the primary entry points. Place the private datasets in `data/` before training. A recorded summary, not a fresh run of this repository, selected **BiLSTM-V0.1 with early stopping** as the overall model on accuracy, class-1 precision, and both class F1 scores (accuracy 0.8320, precision₁ 0.8043, F1₁ 0.7875, F1₀ 0.8394, AUC 0.8870). Random Forest with class balancing remained higher on recall₁ (0.8825), F1₁ (0.7994), and AUC (0.9250). Those figures come from an external summary table; window aggregation and cross-model sample alignment still need to be checked against the evaluation code before treating them as a confirmed ranking.

## Repository layout

```text
FBML/
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
└── requirements.txt           # Python dependencies
```

Results are written to `out/` by default. Data, virtual environments, and generated model artifacts are not included in the repository.

## Setup

Use a Python environment compatible with the packages in `requirements.txt`.

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

Place datasets in the repository `data/` directory. All three pipelines read from that folder via `DATA_ROOT = <project>/data`. Edit `FILES` in the matching configuration if your filenames or formats differ. Keep private datasets out of version control.

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

Full training requires the external datasets and installed dependencies. Translation and syntax checks do not establish predictive performance or validate the experimental design.

This repository is intended for research and educational use. See LICENSE.
