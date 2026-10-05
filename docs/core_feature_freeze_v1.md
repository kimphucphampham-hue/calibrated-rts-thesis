# CORE Feature Freeze v1

## 1. Purpose

This document freezes the feature definitions used by the CORE
RQ1 pipeline before inspecting Random Forest or XGBoost model
performance.

Feature inclusion was determined from source availability,
semantics, pre-execution availability, temporal validity,
reproducibility, data coverage, and leakage risk.

Predictive model performance was not used for feature
selection.

## 2. Frozen feature groups

### F1 — Historical baseline

The existing historical baseline contains seven features:

- `failure_rate_history`
- `recent_failure`
- `consecutive_failures`
- `consecutive_passes`
- `test_age_builds`
- `time_since_last_failure_builds`
- `execution_time_mean_history_ms`

These features use only information available before the
target test execution.

### F2 — Change/context

CORE v1 contains one F2 feature:

- `changed_entity_count`

It is defined as the number of distinct `EntityId` values
changed across all commits associated with the target build.

All commits associated with a build are considered.

An F2 value is valid only when all matched change records
satisfy:

`CommitDate <= build_started_at`

If commit linkage fails or any matched change record violates
the temporal condition, the F2 value is missing rather than
zero.

Conflicting duplicate `(Commit, EntityId)` line-change records
were observed in the TCP-CI artifact. Therefore
`change_added_lines`, `change_deleted_lines`, and
`change_churn` are not included in CORE v1.

### F3 — Temporal features

CORE v1 contains:

- `build_index`
- `recent_failure_density_10`
- `execution_time_trend_10`

`build_index` is the deterministic chronological dense index
within each project.

The rolling temporal window is frozen to the previous 10
observed executions of the same test.

`recent_failure_density_10` is the mean failure indicator over
those 10 previous executions.

`execution_time_trend_10` is the linear slope of historical
execution time against build index over those 10 previous
executions.

The current target-build failure outcome and realized execution
time are excluded.

For both rolling F3 features, fewer than 10 prior executions
produces a missing value.

## 3. Window-selection rationale

The temporal window size was selected before model-performance
inspection.

The five-project pilot history audit showed that a 10-execution
history window remained available for at least approximately
95% of rows in every pilot project.

A 20-execution window reduced row coverage to approximately
91% in Angel-ML.

The window decision therefore used history availability and
reproducibility only. ROC-AUC, Average Precision, Brier score,
ECE, or other model-performance results were not inspected.

## 4. Frozen feature sets

### F1

7 historical features.

### F1 + F2

The seven F1 features plus:

- `changed_entity_count`

Total: 8 features.

### F1 + F2 + F3

F1 + F2 plus:

- `build_index`
- `recent_failure_density_10`
- `execution_time_trend_10`

Total: 11 features.

The full `F1_F2_F3` set is the primary CORE RQ1 feature set.

The three sets are frozen now so that later ablation analysis
does not redefine feature groups after model results are known.

## 5. Excluded candidates

The following candidates are excluded from CORE v1:

- `change_added_lines`
- `change_deleted_lines`
- `change_churn`
- `changed_files_count`
- `complexity`
- `dependency_score`
- `coverage_to_change`

Reasons include unresolved aggregation semantics, incomplete
all-build coverage, unavailable dependency artifacts, or
insufficiently established temporal semantics.

## 6. Leakage policy

For target build `b`, no predictor may use the target outcome
or realized target execution duration.

In particular:

- current `failed` is forbidden,
- current `execution_time_ms` is forbidden,
- current `verdict_raw` is forbidden,
- all historical rolling features use observations strictly
  earlier than the target build.

## 7. Missingness

Unavailable source data and insufficient temporal history
remain missing at feature-generation time.

Missing values must not be converted to zero merely to obtain a
complete matrix.

Missing-value handling is performed later inside the
training-fitted preprocessing pipeline.

## 8. Freeze status

CORE feature definitions are frozen as version 1 before
RF/XGB/calibration model-result inspection.

Any subsequent methodological change to these definitions must
be documented and versioned before affected experiments are
rerun.
