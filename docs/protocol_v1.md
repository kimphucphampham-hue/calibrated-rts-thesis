# Calibrated Risk-Bounded RTS Experimental Protocol v1

## 1. Protocol identity

**Protocol name:** Calibrated Risk-Bounded Regression Test
Selection Experimental Protocol

**Version:** 1.0

**Dataset:** TCP-CI v3

**Status:** Pilot pipeline frozen; core calibration,
risk-bounded selection, and longitudinal evaluation pending
implementation.

Protocol v1 freezes the methodological decisions known at the
completion of the five-project pilot.

A methodological change after this version must be documented
as a protocol amendment and, where substantive, must increment
the protocol version.

## 2. Purpose and scope

The purpose of this protocol is to define, before the final
core experiments, how TCP-CI data are transformed into
temporally valid regression-test observations, how failure
risk is estimated and calibrated, how risk-bounded test
selection will be evaluated, and how results will be
aggregated and reported.

The protocol separates two experimental roles:

1. the five-project pilot, used to validate the end-to-end
   implementation and data assumptions; and
2. the core longitudinal experiment, used as the primary
   thesis evidence.

Pilot results are not treated as final evidence for RQ1-RQ3.

Public real CI data are the primary evidence source. Legacy
synthetic data are restricted to pipeline sanity checking and
historical reproduction.

## 3. Research questions and hypotheses

### RQ1 — Probability estimation and calibration

Can machine-learning models estimate test-failure
probabilities accurately and with adequate calibration across
successive CI builds?

**H1:** Probability calibration reduces Brier Score and ECE
without materially degrading ranking/detection performance.

### RQ2 — Risk-bounded selection

Can risk-bounded regression-test selection reduce regression
testing cost while preserving failure-detection capability
relative to appropriate baselines?

**H2:** For rho >= 0.95, the proposed approach reduces runtime
and/or test-selection ratio while retaining high Failure Recall
relative to random and history-based baselines at comparable
budgets.

### RQ3 — Temporal durability and retraining

How stable is selector performance over time and across
projects, and how frequently should the model be retrained?

**H3:** Periodic or rolling retraining reduces performance
decay and variance relative to a static model with acceptable
additional training cost.

## 4. Data source

The primary benchmark is the published TCP-CI v3 main
dataset.

The downloaded artifact contains 25 discovered project
directories.

The canonicalization pipeline primarily uses:

- `exe.csv` for test execution history;
- `builds.csv` for build identity and chronological ordering;
- `id_map.csv` for interpretation of encoded test entities;
- additional TCP-CI artifacts only when their temporal
  semantics have been validated.

Across the downloaded artifact, `exe.csv` uses the common
schema:

`test, build, job, verdict, duration`

Observed verdict values are 0, 1, and 2.

The thesis binary mapping is:

`verdict == 0 -> failed = 0`

`verdict in {1,2} -> failed = 1`

The documented TCP-CI code 3 is treated as failure if it is
encountered, although it was not observed in the downloaded
v3 artifact.

## 5. Observation and decision units

Raw execution staging uses the `test_job_execution`
granularity with the key:

`(project_id, build_id, job_id, test_id)`

The canonical modeling observation is one test within one CI
build:

`test_build`

with primary key:

`(project_id, build_id, test_id)`

The decision unit is whether a test should be selected for a
target build before that test is executed.

For the five frozen pilot projects, no duplicate
`(build_id, test_id)` groups, multi-job groups, or conflicting
verdict groups were observed. This condition must be
revalidated when additional projects enter the core
experiment.

## 6. Canonical schema, target, and cost

The authoritative field-level definition is maintained in:

`configs/tcp_ci_schema_v1.yaml`

The core canonical fields are:

| Field | Role |
|---|---|
| `project_id` | project identifier |
| `build_id` | CI build identifier |
| `build_started_at` | temporal timestamp |
| `build_index` | deterministic chronological build index |
| `test_id` | encoded TCP-CI test/source entity identifier |
| `failed` | binary target |
| `execution_time_ms` | realized execution cost |

Builds are chronologically ordered within each project by
`build_started_at`, with `build_id` used as a deterministic
tie breaker.

Current-build `failed` and `execution_time_ms` are
post-execution quantities and cannot be used as predictors at
the pre-execution decision point.


## 7. Project inclusion and exclusion criteria

Project eligibility must be determined before inspecting
project-level model performance.

A TCP-CI project may enter the core experiment only if:

1. its required source artifacts can be parsed;
2. build execution records can be linked to chronological
   build metadata;
3. canonicalization succeeds without unresolved primary-key
   ambiguity;
4. the binary target is valid;
5. execution cost is available and usable;
6. sufficient chronological history exists for the required
   training/calibration/evaluation windows.

Projects must not be included or excluded because their model
performance is unusually high or low.

For the pilot chronological holdout implementation, the
minimum configured requirements are:

- 100 training builds;
- 20 test builds.

These pilot thresholds validate the implementation and do not
automatically define the final walk-forward eligibility
threshold.

The core study must execute at least 8 eligible projects end
to end, targets 10, and treats extension toward 15–25 projects
as a stretch objective.


## 8. Feature engineering

### 8.1 Pilot feature set F1 — historical features

The frozen pilot history feature set contains:

- `failure_rate_history`
- `recent_failure`
- `consecutive_failures`
- `consecutive_passes`
- `test_age_builds`
- `time_since_last_failure_builds`
- `execution_time_mean_history_ms`

For target build `b`, all history features for test `t` are
derived exclusively from executions of the same
`(project_id, test_id)` occurring in builds earlier than `b`.

### 8.2 Cold start

A previously unseen test has no historical execution record.

Missing historical statistics remain missing at feature
generation time and are handled by the training-fitted
preprocessing pipeline.

### 8.3 Future feature groups

Change/context features are not part of pilot v1.

Candidate future sources include `builds.csv`,
`entity_change_history.csv`, and temporally validated fields
from `dataset.csv`.

Any added feature must pass the same pre-execution
availability and leakage rules before entering core
experiments.

## 9. Leakage-control rules

The fundamental rule is:

> For target build `b`, every model input and every
> decision-time cost estimate must be derivable using
> information available before execution of the target test in
> build `b`.

Consequently:

- current-build `failed` is never a predictor;
- current-build realized `execution_time_ms` is never a
  predictor or selection-time cost;
- historical features use only builds earlier than the target
  build;
- preprocessing parameters are fitted only on the applicable
  training period;
- feature selection and hyperparameter tuning must not inspect
  the outer evaluation period;
- probability calibration must not use the target evaluation
  observations;
- no random row-level split is permitted for primary temporal
  evidence.

History state may be updated after a completed evaluation
build becomes chronologically observable. Such state updates
must not be confused with model retraining.

## 10. Temporal evaluation protocol

### 10.1 Pilot evaluation

The five-project pilot uses a deterministic chronological
outer holdout:

- first 80% of observed builds: training period;
- final 20%: outer test period;
- splitting is performed by build, never by individual row.

This split exists to validate the implementation.

It is not the primary longitudinal evaluation design for the
final thesis claims.

### 10.2 Core evaluation

Primary RQ evidence uses walk-forward evaluation.

For target build `B_t`:

`training/calibration information < B_t`

and:

`evaluation = B_t`

After the outcome of `B_t` becomes observable, historical
state may be updated for later target builds.

Random K-fold cross-validation is not used as primary
evidence.

## 11. Models and baselines

### 11.1 Implemented pilot baselines

Pilot v1 implements:

**Dummy prior**

`DummyClassifier(strategy="prior")`

This provides a constant train-prevalence probabilistic
reference.

**Logistic Regression**

The pilot configuration uses:

- solver: `lbfgs`
- C: `1.0`
- max_iter: `2000`
- no class weighting

No tuning or resampling is performed in baseline v1.

### 11.2 Planned core prediction models

The core RQ1 comparison is planned to include at least:

- Logistic Regression;
- Random Forest;
- Gradient/boosted-tree model.

Model expansion beyond the frozen CORE set requires a protocol
amendment.

## 12. Probability calibration protocol

Probability calibration is part of RQ1 and is not optional in
the final core experiment.

The planned comparison contains:

1. uncalibrated probability;
2. Platt/sigmoid calibration;
3. isotonic calibration.

The calibrator must never be fitted using the outer evaluation
observations.

Within each temporal evaluation cycle, data must respect the
ordering:

`model-fit period < calibration period < evaluation period`

or another time-aware calibration design providing the same
absence of look-ahead.

The exact calibration-window sizing will be frozen before the
core RQ1 experiment and recorded as a protocol amendment if it
is not already specified by this version.

Calibration quality is evaluated separately from ranking
quality.


## 13. Risk-bounded selection protocol

For a target build containing candidate tests `T`, each test
`i` has:

- predicted failure probability `p_i`;
- decision-time estimated execution cost `c_i`.

For a selected subset `S`, the risk-retention requirement is:

`sum_{i in S} p_i >= rho * sum_{i in T} p_i`

The optimization objective is:

`minimize sum_{i in S} c_i`

subject to the risk-retention constraint.

The planned risk-retention levels are:

- rho = 0.90
- rho = 0.95
- rho = 0.99

Current-build realized `execution_time_ms` must not be used as
the selection-time cost.

A historical cost estimate available before execution must be
used for selection.

Realized `execution_time_ms` may be used after execution only
to evaluate actual runtime savings.

Protocol v1 does not yet claim that an exact solver has been
implemented. The exact optimization implementation must be
frozen before RQ2 results are generated.


## 14. Metrics and reporting

### 14.1 Discrimination

- ROC-AUC
- Average Precision / PR-oriented performance

Ranking metrics measure whether failed tests tend to receive
higher scores. They do not establish probability calibration.

### 14.2 Probability quality and calibration

- Brier Score
- Log Loss
- Expected Calibration Error
- calibration bias
- absolute calibration bias
- reliability diagnostics

Calibration-oriented metrics are reported per project.

### 14.3 Selection quality

The RQ2 core experiment will report:

- Failure Recall (FR)
- Test Selection Ratio (TSR)
- Runtime Ratio (RTR)
- Missed Failure Rate (MFR)

where reliable cost measurements permit, realized CI time
saving is also reported.

### 14.4 Reporting unit

Per-project results are primary.

Cross-project summaries must not silently pool every test or
build as if they were independent observations.


## 15. Statistical analysis

The project is the primary independent unit for statistical
comparison.

For each method and metric, the analysis reports:

- project-level values;
- median;
- interquartile range;
- 95% bootstrap confidence intervals where applicable.

Paired method comparisons across projects use the
Wilcoxon signed-rank test when its use is appropriate for the
available project-level observations.

Effect size is reported using Vargha-Delaney A12 or Cliff's
delta as appropriate.

Multiple pairwise comparisons use Holm correction.

Individual CI builds are not treated as independent samples
for the primary cross-project significance test.

## 16. Reproducibility and artifact governance

The experiment is configuration-driven.

Protocol decisions and stage-specific parameters are stored in
version-controlled YAML files.

Source code, tests, documentation, and configuration files are
committed to Git.

Large raw/intermediate/generated artifacts may be excluded
from Git when they are reproducibly regenerable.

Each major pipeline stage must provide:

- explicit input paths;
- explicit output paths;
- deterministic project/build ordering;
- validation checks;
- automated tests where practical.

Per-project pilot output manifests provide traceable links
from project summaries to underlying generated artifacts.

The Git commit used for final experiments must be recorded with
the reported results.

## 17. Protocol amendments and deviations

Protocol v1 is intended to reduce post-hoc methodological
decision making.

After protocol v1 is committed:

- implementation bug fixes that preserve the intended method
  may be applied without redefining the research question;
- methodological changes must be documented explicitly;
- changes affecting project eligibility, temporal splitting,
  feature availability, calibration, selection objectives,
  rho values, primary metrics, or statistical tests require a
  protocol version increment or formal amendment.

Every amendment must record:

- what changed;
- why it changed;
- whether previous results were inspected before the change;
- which experiments must be rerun.

Final thesis results must identify the protocol version under
which they were generated.


## 18. Known issues and current limitations

### 18.1 Probability calibration not yet implemented

The pilot evaluates raw probabilistic baselines only.
Platt/sigmoid and isotonic calibration remain pending for the
core RQ1 experiment.

### 18.2 Risk-bounded optimization not yet implemented

The exact risk-bounded selector has not yet been implemented.
No pilot result is interpreted as evidence for RQ2.

### 18.3 Core walk-forward experiment not yet executed

The existing five-project 80/20 chronological holdout is an
implementation-validation pilot and not the final longitudinal
evaluation.

### 18.4 Test identifier semantics

For the five pilot projects, `exe.csv:test` was verified
against `id_map.csv:value` and is used as the canonical
`test_id`.

However, the encoded entity must not be interpreted as an
individual test-method identifier without additional evidence.

### 18.5 Change/context features

Pilot v1 uses the minimal history feature set only.

Change/context features have not yet been incorporated into the
pilot model and require temporal-validity screening before
CORE use.

### 18.6 Public CI data only

The primary evidence currently uses public TCP-CI projects.

Enterprise evaluation is optional and is not required for the
validity of the thesis core experiment.

### 18.7 External validity

The benchmark consists primarily of open-source CI projects.
Results may not directly generalize to proprietary systems,
different languages, CI platforms, or organizational testing
processes.