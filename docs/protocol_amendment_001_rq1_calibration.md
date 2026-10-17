# Protocol Amendment 001 — RQ1 Calibration Design

## Status

Frozen before the RF/XGB calibrated RQ1 experiment.

## 1. Purpose

Protocol v1 requires probability calibration for RQ1 and
requires temporal ordering between model fitting, calibration,
and evaluation.

This amendment freezes the previously unspecified calibration
window, rare-event handling, calibrator input, metric
conventions, and pilot retraining behavior.

## 2. Existing decisions preserved

The existing pilot outer split remains unchanged.

For each project:

- the first 80% of chronological builds form the development
  period;
- the final 20% form the outer evaluation period;
- splitting occurs at build level;
- no random row-level split is permitted;
- the outer evaluation period is not used for model fitting,
  calibration fitting, feature selection, preprocessing fitting,
  or hyperparameter tuning.

The outer evaluation boundary is immutable.

## 3. Development-period partition

The initial development-period allocation is:

- first 80% of development builds: model-fit;
- final 20% of development builds: calibration.

This corresponds approximately to:

- 64% model-fit;
- 16% calibration;
- 20% evaluation;

when no rare-event adjustment is required.

The 64/16/20 proportions are therefore an initial allocation,
not a mandatory fixed proportion for every project.

## 4. Rare-event calibration-window expansion

The calibration partition must contain at least:

- 5 positive test observations;
- 5 negative test observations;
- 2 failing builds;
- 2 passing builds.

If the initial calibration partition does not satisfy these
requirements, its start boundary is expanded backward by one
chronological build at a time.

Expansion stops at the first window satisfying all support
requirements.

The outer evaluation period must never be consumed.

At least 100 builds must remain in the model-fit period.

If sufficient support cannot be obtained without violating
this floor, calibrated variants are marked:

`INSUFFICIENT_CALIBRATION_SUPPORT`

The uncalibrated variant remains eligible for evaluation.

## 5. Pilot feasibility evidence

The initial window satisfied the calibration support rule for
Camunda, ShardingSphere, Buck, and SPF4J.

Angel-ML initially contained four positive calibration
observations and therefore failed the pre-specified five-event
minimum.

Applying the deterministic expansion rule required seven
additional earlier builds.

The resulting Angel-ML partition contained:

- 189 model-fit builds;
- 57 calibration builds;
- 62 evaluation builds;
- 5 positive calibration observations;
- 5 failing calibration builds.

The outer evaluation boundary was unchanged.

No predictive-performance metric was used to determine the
seven-build expansion.

## 6. Probability calibration methods

RQ1 compares:

1. uncalibrated base probability;
2. Platt calibration;
3. isotonic calibration.

Calibration is project-specific and model-specific.

A calibrator is never shared across projects or across base
models.

## 7. Calibration input

Each base model exposes its positive-class probability:

`p_raw`

The uncalibrated condition uses `p_raw` directly.

Platt calibration uses:

`logit(clip(p_raw, 1e-6, 1 - 1e-6))`

as its scalar input and fits a sigmoid mapping using calibration
observations only.

Isotonic calibration uses `p_raw` directly and fits a
monotonic non-decreasing mapping using calibration observations
only.

This establishes one common probability-level interface across
LR, RF, and XGBoost.

## 8. Model fitting and preprocessing

Model preprocessing is fitted using model-fit observations
only.

The resulting preprocessing transformation is applied
unchanged to calibration and evaluation observations.

Each base model is fitted once per project for the five-project
pilot.

Each calibrator is fitted once per project/model combination.

Pilot RQ1 does not perform model retraining across evaluation
builds.

Final longitudinal thesis evidence remains subject to the
walk-forward protocol defined separately.

## 9. Hyperparameter policy

RQ1 v1 does not use result-driven hyperparameter tuning.

No outer evaluation data may be used to choose model
parameters.

Exact RF and XGBoost parameter values must be frozen before
their first RQ1 performance run.

## 10. RQ1 metrics

Discrimination:

- ROC-AUC;
- Average Precision.

Probability quality:

- Brier Score;
- Log Loss.

Calibration diagnostics:

- Expected Calibration Error;
- mean predicted probability;
- calibration bias;
- absolute calibration bias;
- reliability diagrams.

Threshold-dependent classification Recall is not computed in
RQ1 v1 because no binary classification threshold is defined.
Failure Recall is evaluated later under the explicit selection
rules used for RQ2.

## 11. ECE definition

ECE uses 10 equal-width probability bins over [0, 1].

For every non-empty bin:

- mean predicted probability is computed;
- observed failure frequency is computed;
- the absolute gap is weighted by the fraction of evaluation
  observations in that bin.

Empty bins do not contribute.

The existing pilot ECE implementation is retained.

## 12. Calibration-bias convention

Calibration bias is:

`mean predicted probability - observed failure prevalence`

A positive value indicates average over-estimation of failure
risk.

A negative value indicates average under-estimation.

Absolute calibration bias is the absolute value of the signed
bias.

## 13. Reporting

Per-project results are primary.

Rows from different projects must not be silently pooled as if
all observations were statistically independent.

For five projects, three models, and three calibration
conditions, the pilot RQ1 matrix contains at most 45
project-model-calibration conditions.

Skipped calibration conditions are retained in experiment
metadata with an explicit skip status.

## 14. Information available when this amendment was frozen

Existing raw Logistic Regression pilot diagnostics had already
been produced during the earlier pipeline-validation phase.

Those historical LR pilot metrics were not used to select the
calibration-window size, expansion rule, minimum event support,
Platt definition, isotonic definition, or ECE configuration.

No RF, XGBoost, Platt-calibrated, or isotonic-calibrated RQ1
performance results were inspected before these decisions were
frozen.

The calibration-window decision used only temporal ordering,
class/event availability, and implementation feasibility.

## 15. Change control

Any later change affecting:

- temporal partitioning;
- calibration-window expansion;
- minimum calibration support;
- calibrator input;
- Platt or isotonic definitions;
- RQ1 primary metrics;
- ECE binning;
- retraining behavior;

requires a documented protocol amendment before affected
experiments are rerun.
