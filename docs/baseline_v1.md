# TCP-CI Minimal Baseline v1

## Purpose

This document defines the minimal probabilistic
failure-prediction baseline used to validate the five-project
TCP-CI pilot pipeline.

The purpose of this baseline is pipeline validation rather
than final model selection.

## Evaluation protocol

Each project is modeled independently.

The chronological outer split defined by
`configs/temporal_split_v1.yaml` is used:

- model and preprocessing fit: training period only;
- prediction and evaluation: outer test period only.

No random row-level splitting is used.

## Input features

The logistic baseline uses the seven history features defined
in the minimal history-feature pipeline:

- `failure_rate_history`
- `recent_failure`
- `consecutive_failures`
- `consecutive_passes`
- `test_age_builds`
- `time_since_last_failure_builds`
- `execution_time_mean_history_ms`

Current-build `failed` is the target.

Current-build `execution_time_ms` is an observed realized cost
and is not supplied as a model feature.

## Missing history

Missing historical values are imputed using the median learned
from the training period only.

Missing-value indicators are added by the training-fitted
imputer.

The transformed features are standardized using statistics
learned from the training period only.

## Baselines

### Dummy prior

`DummyClassifier(strategy="prior")`

This model assigns the training failure prevalence as the
probability for every test observation.

It provides a trivial probabilistic reference point.

### Logistic regression

A standard logistic-regression model is fitted using:

- solver: `lbfgs`
- `C = 1.0`
- maximum iterations: 2000
- no class weighting

No hyperparameter search is performed.

## No tuning or calibration

Baseline v1 performs no:

- feature selection;
- hyperparameter tuning;
- SMOTE or resampling;
- probability calibration;
- threshold optimization.

The logistic-regression probabilities are therefore raw,
uncalibrated baseline probabilities.

They must not be interpreted as the final calibrated
probabilities required by the thesis risk-bounded selection
method.

## Metrics

The pilot baseline reports:

- ROC-AUC;
- Average Precision;
- Brier score;
- Log Loss.

Metrics are computed independently for each project.

A macro mean is also reported by averaging project-level
metrics with equal project weight.

## Pilot results

The minimal baseline was executed successfully on all five
frozen TCP-CI pilot projects.

### Per-project results

| Project key | Model | ROC-AUC | Average Precision | Brier Score | Log Loss |
|---|---|---:|---:|---:|---:|
| datasets/camunda@camunda-bpm-platform | Dummy prior | 0.500000 | 0.005100 | 0.005075 | 0.032091 |
| datasets/camunda@camunda-bpm-platform | Logistic regression | 0.918315 | 0.045693 | 0.004963 | 0.024080 |
| datasets/apache@shardingsphere | Dummy prior | 0.500000 | 0.000434 | 0.000436 | 0.004681 |
| datasets/apache@shardingsphere | Logistic regression | 0.958868 | 0.127711 | 0.000462 | 0.002140 |
| datasets/facebook@buck | Dummy prior | 0.500000 | 0.000063 | 0.000069 | 0.002959 |
| datasets/facebook@buck | Logistic regression | 0.813974 | 0.001031 | 0.000099 | 0.000744 |
| datasets/Angel-ML@angel | Dummy prior | 0.500000 | 0.096960 | 0.091877 | 0.364717 |
| datasets/Angel-ML@angel | Logistic regression | 0.976292 | 0.955044 | 0.023399 | 0.103535 |
| datasets/zolyfarkas@spf4j | Dummy prior | 0.500000 | 0.006308 | 0.006269 | 0.038280 |
| datasets/zolyfarkas@spf4j | Logistic regression | 0.895841 | 0.572516 | 0.003691 | 0.020820 |

### Macro mean across projects

| Model | ROC-AUC | Average Precision | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| Dummy prior | 0.500000 | 0.021773 | 0.020745 | 0.088546 |
| Logistic regression | 0.912658 | 0.340399 | 0.006523 | 0.030264 |

The history-based logistic baseline showed substantially better
discrimination than the dummy-prior baseline across all five
pilot projects.

The raw logistic probabilities did not dominate the dummy
baseline on every project and every probability-sensitive
metric. In particular, the logistic baseline produced slightly
higher Brier scores for `apache@shardingsphere` and
`facebook@buck`, despite substantially better ranking metrics.

This observed distinction between discrimination and
probability-sensitive performance is consistent with the
motivation for evaluating an explicit probability-calibration
stage in the thesis pipeline.

Because failure prevalence varies substantially across
projects, per-project results are treated as primary evidence.
Macro averages are reported only as an equal-project summary.

## Temporal state during testing

History features for a test-period build may use execution
history from earlier test-period builds when that information
is chronologically available.

The model itself is not refitted on test-period observations
in this baseline.

Therefore:

`history-state update != model retraining`

## Outputs

Generated outputs are stored under:

`results/pilot/baseline/`

The main files are:

- `baseline_summary.csv`
- `baseline_macro_summary.csv`

Each project also contains:

`test_predictions.csv`

with the observed target/cost and raw baseline probabilities.

## Scope

These results are intended only to demonstrate that the
five-project pilot prediction pipeline runs end-to-end under
the defined temporal and leakage constraints.

They are not the final RQ1 calibration results and are not used
to select the final thesis model.