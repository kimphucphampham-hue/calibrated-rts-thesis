# Calibrated Risk-Bounded Regression Test Selection for CI/CD

A reproducible experimental repository for studying
probability-calibrated, risk-bounded regression test selection
in Continuous Integration / Continuous Delivery environments.

This repository supports the master's thesis:

**Calibrated Risk-Bounded Regression Test Selection for CI/CD
Pipelines: A Longitudinal Empirical Study**

---

## 1. Overview

Regression testing in CI/CD environments can become expensive
as test suites and build frequency grow.

Machine-learning approaches can estimate which tests are more
likely to fail, but ranking tests by a prediction score alone
does not directly provide a decision rule for balancing
failure risk against execution cost.

This project studies a calibration-aware regression-test
selection pipeline in which:

1. a model estimates the failure probability of each test;
2. predicted probabilities are evaluated and calibrated;
3. each test receives a decision-time execution-cost estimate;
4. a risk-bounded selector chooses a lower-cost subset of
   tests while retaining a specified fraction of predicted
   failure risk.

The primary empirical benchmark is the public TCP-CI dataset.

---

## 2. Research objective

The central objective is to investigate whether calibrated
test-failure probabilities can support regression-test
selection decisions that reduce CI testing cost while
retaining a controlled amount of predicted failure risk.

The study follows three principles:

- real CI data first;
- time-aware evaluation;
- explicit separation between probability estimation and
  downstream selection decisions.

Synthetic data inherited from earlier work are used only for
sanity checking and historical reproduction, not as the
primary thesis evidence.

---

## 3. Research questions

### RQ1 — Probability estimation and calibration

Can machine-learning models estimate test-failure
probabilities accurately and with adequate calibration across
successive CI builds?

### RQ2 — Risk-bounded selection

Can risk-bounded regression-test selection reduce regression
testing cost while preserving failure-detection capability
relative to appropriate baselines?

### RQ3 — Temporal durability and retraining

How stable is selector performance over time and across
projects, and how frequently should the prediction model be
retrained?

The complete hypotheses and experimental rules are defined in:

`docs/protocol_v1.md`

---

## 4. Core risk-bounded formulation

For a target build with candidate test set `T`, each test `i`
has:

- predicted failure probability `p_i`;
- decision-time estimated execution cost `c_i`.

For selected subset `S`, the intended risk-retention
constraint is:

`sum_{i in S} p_i >= rho * sum_{i in T} p_i`

while minimizing:

`sum_{i in S} c_i`

The planned risk-retention levels are:

- `rho = 0.90`
- `rho = 0.95`
- `rho = 0.99`

The current realized execution duration of a test is not
available at selection time and therefore cannot be used as
its decision-time cost.

Historical execution-time estimates are used instead.

---

## 5. Current project status

| Component | Status |
|---|---|
| Reproducible repository structure | Completed |
| TCP-CI inventory and Data Dictionary | Completed |
| Five-project pilot | Completed |
| Historical feature pipeline | Completed |
| Leakage guards | Completed |
| Chronological pilot split | Completed |
| Dummy-prior baseline | Completed |
| Logistic-regression baseline | Completed |
| Pilot probability metrics | Completed |
| Single-project reproducibility smoke run | Completed |
| Per-project pilot outputs | Completed |
| Experimental Protocol v1 | Completed |
| Pilot Report v1 | Completed |
| Probability calibration | Planned |
| Risk-bounded selector | Planned |
| Core walk-forward experiment | Planned |
| Retraining experiments | Planned |
| Final statistical analysis | Planned |

The completed five-project pilot validates pipeline
feasibility.

It is not treated as the final experimental evidence for
RQ1-RQ3.

---

## 6. Data

### Primary benchmark

The primary dataset is the published TCP-CI v3 benchmark.

The downloaded main dataset contains 25 project directories.

The main artifacts used by the current pipeline include:

- `exe.csv`
- `builds.csv`
- `id_map.csv`
- `entity_change_history.csv`
- `dataset.csv`

The observed `exe.csv` schema is:

`test, build, job, verdict, duration`

The observed `builds.csv` schema is:

`id, commits, started_at`

### Target mapping

Observed TCP-CI v3 verdict values are:

- `0`
- `1`
- `2`

The thesis binary target is:

`verdict == 0 -> failed = 0`

`verdict in {1, 2} -> failed = 1`

TCP-CI documentation also defines verdict `3` as an unknown
failure. It was not observed in the downloaded v3 artifact but
is treated as failure if encountered.

### Execution duration

Published `exe.csv:duration` values are treated as
milliseconds.

The canonical mapping is therefore:

`exe.csv:duration -> execution_time_ms`

with no additional multiplication by 1000.

---

## 7. Five-project pilot

The frozen pilot projects are:

1. `datasets/camunda@camunda-bpm-platform`
2. `datasets/apache@shardingsphere`
3. `datasets/facebook@buck`
4. `datasets/Angel-ML@angel`
5. `datasets/zolyfarkas@spf4j`

The pilot validates the end-to-end implementation before the
core experiment is expanded toward at least 8 eligible
projects, with 10 projects as the target.

The pilot projects are not retained or removed based on model
performance.

---

## 8. Pilot dataset summary

| Project | Canonical rows | Builds | Tests | Outer-test rows | Failure prevalence |
|---|---:|---:|---:|---:|---:|
| `datasets/camunda@camunda-bpm-platform` | 472,765 | 822 | 1,021 | 128,237 | 0.005100 |
| `datasets/apache@shardingsphere` | 833,541 | 1,049 | 1,020 | 184,349 | 0.000434 |
| `datasets/facebook@buck` | 561,294 | 846 | 800 | 127,456 | 0.000063 |
| `datasets/Angel-ML@angel` | 10,133 | 308 | 46 | 1,908 | 0.096960 |
| `datasets/zolyfarkas@spf4j` | 68,787 | 587 | 168 | 12,999 | 0.006308 |

The large variation in failure prevalence is one reason why
results are interpreted primarily at the project level.

---

## 9. Canonical data model

Raw TCP-CI execution records are staged at:

`test_job_execution`

granularity with key:

`(project_id, build_id, job_id, test_id)`

The canonical modeling observation is:

`test_build`

with primary key:

`(project_id, build_id, test_id)`

Core canonical fields include:

| Field | Role |
|---|---|
| `project_id` | project identifier |
| `build_id` | CI build identifier |
| `build_started_at` | chronological timestamp |
| `build_index` | deterministic chronological index |
| `test_id` | encoded test/source entity identifier |
| `failed` | binary target |
| `execution_time_ms` | realized execution cost |

The authoritative schema is:

`configs/tcp_ci_schema_v1.yaml`

---

## 10. Historical feature set

The pilot uses seven causal historical features:

- `failure_rate_history`
- `recent_failure`
- `consecutive_failures`
- `consecutive_passes`
- `test_age_builds`
- `time_since_last_failure_builds`
- `execution_time_mean_history_ms`

For a target build `b`, these features are computed using only
executions of the same `(project_id, test_id)` occurring in
builds earlier than `b`.

The seven features form the frozen F1 historical baseline.

Change/context features are planned for later evaluation under
the same temporal-validity requirements.

---

## 11. Leakage-control rule

The fundamental leakage rule is:

> For a target build `b`, every predictor and every
> decision-time cost estimate must be derivable from
> information available before execution of the target test in
> build `b`.

Therefore:

- current-build `failed` is never a predictor;
- current-build `execution_time_ms` is never a predictor;
- current realized execution cost is not used for selection;
- historical features use only earlier builds;
- preprocessing is fitted only on eligible training data;
- evaluation data cannot be used to fit probability
  calibrators;
- random row-level splitting is not used for primary temporal
  evidence.

Automated leakage tests are included in the test suite.

---

## 12. Temporal evaluation

### Pilot

The five-project pilot uses a deterministic chronological
build-level holdout:

- first 80% of observed builds: training;
- final 20%: outer testing.

This split exists for implementation validation.

### Core experiment

The primary thesis evaluation uses walk-forward evaluation.

For target build `B_t`:

`training/calibration information < B_t`

and:

`evaluation = B_t`

After `B_t` becomes observable, its outcome may update the
historical state used for later builds.

Random K-fold cross-validation is not used as the primary
experimental design.

---

## 13. Pilot models

### Dummy prior

The probabilistic sanity baseline uses:

`DummyClassifier(strategy="prior")`

It predicts the training-set class prior.

### Logistic Regression

The history-based pilot baseline uses Logistic Regression with
a training-fitted preprocessing pipeline.

Pilot configuration:

- solver: `lbfgs`
- `C = 1.0`
- `max_iter = 2000`
- no class weighting
- no hyperparameter tuning
- no resampling
- no probability calibration

Additional models are reserved for the core experiment.

---

## 14. Pilot evaluation metrics

The current pilot evaluates:

### Discrimination

- ROC-AUC
- Average Precision

### Probability quality

- Brier Score
- Log Loss

### Calibration diagnostics

- Expected Calibration Error (ECE)
- mean predicted probability
- signed calibration bias
- absolute calibration bias
- reliability-bin diagnostics

Selection metrics such as Failure Recall, Test Selection Ratio,
Runtime Ratio, and Missed Failure Rate are reserved for the
risk-bounded selection experiment.

---

## 15. Pilot result snapshot

The history-based Logistic Regression baseline produced the
following diagnostic results:

| Project | ROC-AUC | Average Precision | Brier | Log Loss | ECE |
|---|---:|---:|---:|---:|---:|
| `datasets/camunda@camunda-bpm-platform` | 0.918315 | 0.045693 | 0.004963 | 0.024080 | 0.001120 |
| `datasets/apache@shardingsphere` | 0.958868 | 0.127711 | 0.000462 | 0.002140 | 0.000269 |
| `datasets/facebook@buck` | 0.813974 | 0.001031 | 0.000099 | 0.000744 | 0.000080 |
| `datasets/Angel-ML@angel` | 0.976292 | 0.955044 | 0.023399 | 0.103535 | 0.038954 |
| `datasets/zolyfarkas@spf4j` | 0.895841 | 0.572516 | 0.003691 | 0.020820 | 0.002938 |

These values are pilot diagnostics only.

They are not interpreted as evidence that Logistic Regression
is the optimal model or that its raw probabilities are
consistently calibrated.

See:

`docs/pilot_report_v1.md`

for the full pilot interpretation.

---

## 16. Repository structure

```text
calibrated-rts-thesis/
├── configs/
│   ├── tcp_ci_schema_v1.yaml
│   ├── pilot_projects_v1.yaml
│   ├── history_features_v1.yaml
│   ├── temporal_split_v1.yaml
│   ├── baseline_v1.yaml
│   ├── pilot_metrics_v1.yaml
│   ├── project_outputs_v1.yaml
│   └── protocol_v1.yaml
│
├── data/
│   ├── raw/
│   │   └── tcp_ci/
│   ├── interim/
│   └── processed/
│
├── docs/
│   ├── protocol_v1.md
│   ├── pilot_report_v1.md
│   ├── history_features_v1.md
│   ├── temporal_split_v1.md
│   ├── baseline_v1.md
│   ├── pilot_metrics_v1.md
│   └── project_outputs_v1.md
│
├── results/
│   └── pilot/
│
├── src/
│   └── tcp_ci/
│       ├── adapt_pilots.py
│       ├── materialize_canonical.py
│       ├── build_history_features.py
│       ├── split_by_build.py
│       ├── run_baseline.py
│       ├── evaluate_pilot_metrics.py
│       └── export_project_outputs.py
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 17. Environment setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the repository:

```bash
python -m pytest -q
```

All tests should pass before running experimental stages.

---

## 18. Data placement

The TCP-CI v3 main dataset is not intended to be duplicated in
Git.

Place the downloaded source artifact under the repository's
raw-data location according to the existing TCP-CI data
configuration.

The current repository uses:

```text
data/raw/tcp_ci/
```

Raw, interim, processed, and generated result artifacts should
follow the repository's `.gitignore` and reproducibility
policy.

---

## 19. Reproducing the five-project pilot

The current pilot pipeline is executed in chronological stage
order.

### 19.1 Canonicalization

Run the TCP-CI pilot adapter and canonical materialization
stages defined under:

```text
src/tcp_ci/
```

The canonical output must satisfy the schema defined in:

```text
configs/tcp_ci_schema_v1.yaml
```

### 19.2 Historical features

```bash
python src/tcp_ci/build_history_features.py
```

### 19.3 Chronological split

```bash
python src/tcp_ci/split_by_build.py
```

### 19.4 Probabilistic baselines

```bash
python src/tcp_ci/run_baseline.py
```

### 19.5 Pilot metrics

```bash
python src/tcp_ci/evaluate_pilot_metrics.py
```

### 19.6 Per-project output packages

```bash
python src/tcp_ci/export_project_outputs.py
```

### 19.7 Validation

```bash
python -m pytest -q
```

Stage-specific configurations under `configs/` are the
authoritative source for experiment parameters.

---

## 20. Generated outputs

Important pilot outputs include:

```text
results/pilot/baseline/baseline_summary.csv
results/pilot/baseline/baseline_macro_summary.csv

results/pilot/metrics/pilot_metrics.csv
results/pilot/metrics/pilot_metrics_macro.csv
results/pilot/metrics/calibration_bins.csv

results/pilot/projects/project_output_index.csv
```

Each project package under:

```text
results/pilot/projects/<project>/
```

contains:

```text
project_summary.json
model_metrics.csv
calibration_bins.csv
artifact_manifest.csv
```

Large train/test/history/prediction artifacts remain at their
canonical generated locations instead of being duplicated into
each project package.

---

## 21. Documentation

The main project documents are:

### Experimental protocol

[Experimental Protocol v1](docs/protocol_v1.md)

Defines the frozen methodological rules for the pilot and
planned core experiments.

### Pilot report

[Pilot Report v1](docs/pilot_report_v1.md)

Summarizes pilot feasibility, data-quality findings, pilot
results, limitations, and decisions for the core phase.

### Canonical schema

[`configs/tcp_ci_schema_v1.yaml`](configs/tcp_ci_schema_v1.yaml)

Defines observation granularity, canonical fields, temporal
availability, leakage status, and candidate feature schema.

### Historical features

[`docs/history_features_v1.md`](docs/history_features_v1.md)

Defines the causal historical feature set and cold-start
behavior.

### Temporal splitting

[`docs/temporal_split_v1.md`](docs/temporal_split_v1.md)

Documents the chronological pilot holdout.

### Baselines

[`docs/baseline_v1.md`](docs/baseline_v1.md)

Documents the dummy-prior and Logistic Regression pilot
baselines.

### Pilot metrics

[`docs/pilot_metrics_v1.md`](docs/pilot_metrics_v1.md)

Defines the pilot discrimination and probability-quality
metrics.

### Per-project outputs

[`docs/project_outputs_v1.md`](docs/project_outputs_v1.md)

Defines the compact project-level output package.

---

## 22. Known limitations

The current repository has completed the five-project pilot,
but several components required for the final thesis
experiments remain pending.

In particular:

- Platt/sigmoid calibration is not yet implemented;
- isotonic calibration is not yet implemented;
- the risk-bounded selector is not yet implemented;
- the exact optimization method is not yet frozen in code;
- the final walk-forward experiment has not yet been executed;
- retraining policies for RQ3 have not yet been evaluated;
- the pilot uses only the F1 historical feature group;
- change/context features require temporal-validity screening;
- the exact semantic granularity of the encoded TCP-CI test
  entity beyond the verified `id_map.csv:value` relationship
  remains limited;
- computational runtime and memory requirements have not yet
  been systematically benchmarked over the complete
  25-project dataset.

These limitations are explicitly separated from resolved
pilot findings.

---

## 23. Roadmap

The next core research stages are:

1. Screen additional TCP-CI projects using the frozen
   eligibility criteria.
2. Expand the full pipeline toward at least 8 eligible
   projects, targeting 10.
3. Implement time-aware probability calibration.
4. Evaluate uncalibrated, Platt/sigmoid, and isotonic
   probabilities.
5. Implement and validate risk-bounded selection.
6. Freeze the exact selection optimization procedure.
7. Implement the core walk-forward experiment.
8. Evaluate retraining policies for RQ3.
9. Screen and integrate temporally valid change/context
   features.
10. Perform project-level statistical analysis.
11. Generate the final RQ1-RQ3 results.

Methodological changes after Protocol v1 must follow the
protocol amendment rules.

---

## 24. Reproducibility

The project follows a configuration-driven workflow.

Methodological decisions are stored in version-controlled
configuration and documentation files.

Before reporting final experimental results, the repository
will record:

- the exact Git commit;
- protocol version;
- dataset version;
- project set;
- model configuration;
- calibration configuration;
- selector configuration;
- temporal evaluation configuration.

This allows reported results to be associated with a specific
experimental implementation.

---

## 25. Research status

This repository is an active master's thesis research project.

The completed five-project pilot establishes implementation
feasibility and supports scaling the TCP-CI pipeline toward
the core experiment.

The pilot does not establish the final effectiveness of
probability calibration or risk-bounded regression-test
selection.

Those claims require the implementation and execution of the
core walk-forward calibration, risk-bounded selection, and
retraining experiments already specified in Protocol v1.



