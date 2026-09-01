# TCP-CI Five-Project Pilot Report v1

**Dataset:** TCP-CI v3

**Pilot projects:** 5

**Protocol:** `docs/protocol_v1.md`

**Status:** Completed

## 1. Pilot objective and success criteria

## 1. Pilot objective and success criteria

The purpose of the five-project pilot is to validate whether
the proposed TCP-CI experimental pipeline can be executed
end-to-end on real CI data before scaling to the core thesis
experiment.

The pilot is intended to validate implementation feasibility,
data assumptions, temporal correctness, leakage controls, and
basic probabilistic evaluation.

It is not intended to establish the final answer to RQ1, RQ2,
or RQ3.

The pilot is considered successful if:

1. all five frozen projects can be canonicalized successfully;
2. chronological build ordering is valid;
3. the canonical `(project_id, build_id, test_id)` key is valid;
4. historical features can be generated without using
   current-build outcome or cost;
5. train/test splitting preserves chronological order;
6. baseline models can produce valid probabilities;
7. pilot metrics can be computed consistently;
8. one project can reproduce its result when run independently;
9. compact per-project output packages can be generated; and
10. no blocking data issue prevents extension toward the core
    multi-project experiment.


## 2. Dataset and pilot projects

The pilot uses the published TCP-CI v3 main dataset as the
primary real-CI benchmark.

The downloaded benchmark contains 25 discovered projects.

Five projects were frozen for the initial pilot:

1. `datasets/camunda@camunda-bpm-platform`
2. `datasets/apache@shardingsphere`
3. `datasets/facebook@buck`
4. `datasets/Angel-ML@angel`
5. `datasets/zolyfarkas@spf4j`

These projects are used to validate the pipeline before
expanding to the minimum core study size.

Project retention is not determined from model performance.
A project with unusually high or low predictive performance
is not removed solely because of that result.

## 3. Pilot pipeline and scope

The validated pilot pipeline is:

`TCP-CI raw artifacts`

→ `staging test_job_execution`

→ `canonical test_build`

→ `causal historical features`

→ `leakage guards`

→ `chronological build-level split`

→ `Dummy prior + Logistic Regression`

→ `raw probability evaluation`

→ `calibration-oriented diagnostics`

→ `per-project output packages`

The pilot uses a chronological 80/20 build-level holdout.

This split validates the implementation only.

The final core thesis experiment will use the walk-forward
evaluation protocol defined in `docs/protocol_v1.md`.

The pilot does not yet implement:

- probability calibration;
- risk-bounded test selection;
- the exact optimization solver;
- final retraining policies;
- the full change/context feature set.

## 4. Data quality and canonicalization findings

### 4.1 Project scale

| Project key | Canonical rows | Builds | Tests | Outer-test rows | Outer-test failures | Failure prevalence |
|---|---:|---:|---:|---:|---:|---:|
| datasets/camunda@camunda-bpm-platform | 472,765 | 822 | 1,021 | 128,237 | 654 | 0.005100 |
| datasets/apache@shardingsphere | 833,541 | 1,049 | 1,020 | 184,349 | 80 | 0.000434 |
| datasets/facebook@buck | 561,294 | 846 | 800 | 127,456 | 8 | 0.000063 |
| datasets/Angel-ML@angel | 10,133 | 308 | 46 | 1,908 | 185 | 0.096960 |
| datasets/zolyfarkas@spf4j | 68,787 | 587 | 168 | 12,999 | 82 | 0.006308 |

### 4.2 Canonicalization checks

The five pilot projects passed the canonicalization checks.

Across the pilot projects:

- the canonical `(project_id, build_id, test_id)` key is
  non-null and unique;
- all execution build identifiers link to the corresponding
  build metadata;
- build timestamps support deterministic chronological
  ordering;
- no duplicate `(build_id, test_id)` execution groups were
  observed;
- no multi-job groups requiring aggregation were observed;
- no conflicting verdict groups were observed;
- no missing or negative execution duration values were
  observed.

The published `exe.csv:duration` field is treated as
milliseconds and maps directly to `execution_time_ms`.
No additional multiplication by 1000 is applied.

### 4.3 Target and test identity

Observed `verdict` values are 0, 1, and 2.

The pilot target mapping is:

`verdict == 0 -> failed = 0`

`verdict in {1,2} -> failed = 1`

For all five pilot projects, `exe.csv:test` matched
`id_map.csv:value` and is used as the canonical `test_id`.

The encoded identifier is treated as a stable thesis entity
identifier.

However, the available evidence does not justify interpreting
it as an individual test-method identifier.

### 4.4 Failure imbalance

Failure prevalence differs substantially across projects.

In particular:

- `datasets/facebook@buck` contains only 8 failures among
  127,456 outer-test observations;
- `datasets/apache@shardingsphere` contains 80 failures among
  184,349 outer-test observations;
- `datasets/Angel-ML@angel` has a substantially higher failure
  prevalence of approximately 9.7%.

Therefore, probability and ranking metrics must be interpreted
per project.

Average Precision should be interpreted relative to each
project's failure prevalence rather than compared only by its
absolute value.


## 5. Pilot results

### 5.1 Logistic-regression diagnostic results

| Project key | ROC-AUC | Average Precision | Brier Score | Log Loss | ECE | Absolute calibration bias |
|---|---:|---:|---:|---:|---:|---:|
| datasets/camunda@camunda-bpm-platform | 0.918315 | 0.045693 | 0.004963 | 0.024080 | 0.001120 | 0.000916 |
| datasets/apache@shardingsphere | 0.958868 | 0.127711 | 0.000462 | 0.002140 | 0.000269 | 0.000050 |
| datasets/facebook@buck | 0.813974 | 0.001031 | 0.000099 | 0.000744 | 0.000080 | 0.000013 |
| datasets/Angel-ML@angel | 0.976292 | 0.955044 | 0.023399 | 0.103535 | 0.038954 | 0.037610 |
| datasets/zolyfarkas@spf4j | 0.895841 | 0.572516 | 0.003691 | 0.020820 | 0.002938 | 0.001268 |


### 5.2 Interpretation

The history-based logistic baseline produced non-trivial
ranking signal in all five pilot projects.

However, these results are not interpreted as evidence that
the model is optimal or consistently calibrated.

Probability-sensitive performance was project-dependent.

For example, the raw logistic model achieved strong ranking
performance in several projects while its Brier Score or ECE
did not uniformly dominate the dummy-prior reference.

This distinction supports the need for the explicit
probability-calibration experiment defined in Protocol v1.

Because failure prevalence differs substantially across
projects, per-project results are treated as the primary pilot
evidence.

Macro averages are considered descriptive summaries only.


### 5.3 Dummy-prior sanity baseline

The dummy-prior model behaved as expected:

- ROC-AUC was 0.5 for all five projects;
- Average Precision equaled outer-test failure prevalence;
- AP lift was 1.0.

These results provide a basic consistency check for the pilot
metric implementation.


## 6. Findings, resolved issues, and open limitations

### 6.1 Resolved during the pilot

The pilot resolved several data and implementation questions.

**Schema**

All 25 inspected `exe.csv` artifacts use the common schema:

`test, build, job, verdict, duration`

The observed `builds.csv` schema is:

`id, commits, started_at`

**Duration**

The published TCP-CI v3 `exe.csv:duration` value is treated as
milliseconds.

No further numerical conversion is applied during thesis
canonicalization.

**Build linkage**

Execution records can be linked to chronological build
metadata for the pilot projects.

**Pilot aggregation**

No repeated `(build_id, test_id)` execution groups,
multi-job groups, or conflicting verdict groups were observed
in the five pilot projects.

Therefore, no test-job aggregation was required for these
projects.


### 6.2 Open limitations

**Test identifier semantics**

The relationship between `exe.csv:test` and
`id_map.csv:value` has been verified for the pilot.

However, the exact semantic granularity of the encoded test
entity beyond this relationship remains limited.

**Extreme class imbalance**

Several projects contain very few failures.

This affects the interpretation and stability of PR-oriented
and calibration metrics.

**Change/context features**

The pilot uses only the minimal historical feature group.

Change/context features have not yet been screened and added
under the temporal leakage rules.

**Calibration**

The current probabilities are uncalibrated baseline
probabilities.

Platt/sigmoid and isotonic calibration are not yet
implemented.

**Risk-bounded selection**

The selector and exact optimization procedure have not yet
been implemented.

**Core temporal evaluation**

The final walk-forward experiment has not yet been run.

**Pipeline computational runtime**

Wall-clock runtime and memory consumption have not yet been
systematically benchmarked across the complete 25-project
dataset.

## 7. Decisions for the core phase — September 2026

### 7.1 Project retention

All five pilot projects are retained as valid pilot evidence.

No pilot project is excluded based on predictive performance.

The five projects remain candidates for the core project pool,
subject to the final project-eligibility rules required by the
walk-forward and calibration protocols.

The next expansion step is to screen additional TCP-CI
projects using the predefined eligibility criteria until at
least 8 projects can execute the full core pipeline, with 10
projects as the target.

### 7.2 Feature decision

The seven historical features are frozen as the F1 historical
baseline feature group:

- `failure_rate_history`
- `recent_failure`
- `consecutive_failures`
- `consecutive_passes`
- `test_age_builds`
- `time_since_last_failure_builds`
- `execution_time_mean_history_ms`

This freezes the historical baseline.

It does not imply that the final thesis feature set is limited
to these seven features.

Change/context features must be evaluated separately under the
same temporal-validity rules before inclusion in the core
experiment.


### 7.3 Pipeline decisions

The canonicalization, historical-feature, leakage-guard,
temporal-split, baseline, metric, smoke-run, and project-output
stages are retained.

No blocking redesign of these pilot stages is required.

The next core implementation priorities are:

1. implement temporally valid probability calibration;
2. implement the risk-bounded selector;
3. freeze the exact optimization method;
4. implement walk-forward evaluation;
5. implement retraining policies required for RQ3;
6. screen additional eligible TCP-CI projects;
7. evaluate temporally valid change/context features.

### 7.4 Checks required before calibration experiments

Before generating RQ1 calibration results, the following
decisions must be frozen:

- model-fit/calibration/evaluation window ordering;
- calibration-window size;
- minimum positive-event requirements for calibration;
- handling of projects/windows with extremely rare failures;
- Platt/sigmoid implementation;
- isotonic implementation;
- retraining behavior across walk-forward steps;
- whether calibration is model-specific at each temporal step;
- calibration metrics and reliability-diagram reporting rules.

Outer evaluation data must never be used to fit the
calibrator.

## 8. Reproducibility and artifacts

The pilot is reproducible through the version-controlled
pipeline configuration and source code.

Key documentation and configuration artifacts include:

- `configs/tcp_ci_schema_v1.yaml`
- `configs/history_features_v1.yaml`
- `configs/temporal_split_v1.yaml`
- `configs/baseline_v1.yaml`
- `configs/pilot_metrics_v1.yaml`
- `configs/project_outputs_v1.yaml`
- `configs/protocol_v1.yaml`
- `docs/protocol_v1.md`

Generated pilot summaries include:

- `results/pilot/baseline/baseline_summary.csv`
- `results/pilot/metrics/pilot_metrics.csv`
- `results/pilot/metrics/calibration_bins.csv`
- `results/pilot/projects/project_output_index.csv`

Per-project packages under `results/pilot/projects/` provide
compact summaries and manifests pointing to the underlying
generated artifacts.

The Git commit associated with this pilot report should be
recorded when the report is frozen.

**Pilot implementation baseline:** `0af54741e5df5fe4b8a59aa06d1d872dc3ca674c`


## 9. Conclusion

The five-project pilot successfully validates the feasibility
of the implemented TCP-CI experimental pipeline.

All five frozen pilot projects were processed through the
canonicalization, historical-feature, leakage-control,
chronological splitting, probabilistic baseline, metric, and
project-output stages without a blocking data failure.

The pilot also identified substantial cross-project variation
in project scale and failure prevalence, confirming that
per-project evaluation and time-aware analysis are necessary.

The current evidence supports the use of TCP-CI as the primary
benchmark for the next experimental phase.

The existing pipeline appears technically suitable for
expansion toward the required 8–10 eligible projects.

However, the pilot does not establish the final effectiveness
of probability calibration or risk-bounded regression-test
selection.

Those claims require the core walk-forward calibration,
selection, and retraining experiments defined in Protocol v1.