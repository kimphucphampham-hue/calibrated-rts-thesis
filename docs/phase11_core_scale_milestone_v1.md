# Phase 11 Core-Scale Pipeline Milestone v1

## 1. Status

Phase 11 — Multi-project Core-Scale Pipeline Expansion

Status:

**CLOSED / PASS**

Phase 11 establishes multi-project implementation and
scaling evidence for the frozen RQ1 probability-estimation
and calibration pipeline.

It is not the primary final longitudinal RQ1 evidence.

The implementation baseline associated with this milestone is:

`4b9d6da61fe3a05de67db018e972f5acaa20bc06`

---

## 2. Scope

Phase 11 validates that the frozen CORE pipeline can be
executed reproducibly across the frozen ten-project TCP-CI v3
project set.

The Phase 11 scope is limited to:

- deterministic structural project screening;
- ten-project canonical data preparation;
- frozen F1/F2/F3 feature materialization;
- leakage validation;
- chronological development/evaluation splitting;
- RQ1 probability-model execution;
- Platt and isotonic calibration;
- probability-quality evaluation;
- implementation-scale resource validation;
- independent output auditing;
- reproducibility verification.

Phase 11 does not perform:

- final walk-forward evaluation;
- risk-bounded test selection;
- RQ2 effectiveness analysis;
- RQ3 retraining experiments;
- model selection from observed performance;
- calibration-method selection from observed performance;
- final thesis effectiveness claims.

---

## 3. Dataset and project selection

Dataset:

- TCP-CI
- version v3

Structural screening discovered:

- 25 projects;
- 25 structurally eligible projects.

The Phase 11 target was ten end-to-end projects.

The frozen ten-project execution set is:

1. `datasets/camunda@camunda-bpm-platform`
2. `datasets/apache@shardingsphere`
3. `datasets/facebook@buck`
4. `datasets/Angel-ML@angel`
5. `datasets/zolyfarkas@spf4j`
6. `datasets/CompEvol@beast2`
7. `datasets/EMResearch@EvoMaster`
8. `datasets/Graylog2@graylog2-server`
9. `datasets/JMRI@JMRI`
10. `datasets/SonarSource@sonarqube`

The first five are the previously frozen pilot anchors.

The additional projects were selected deterministically from
the structurally eligible projects according to the frozen
Phase 11 project-selection rule.

No model-performance metric was used to include or exclude a
project.

The remaining eligible projects were not classified as
ineligible; execution stopped after the frozen target count of
ten projects was reached.

---

## 4. Frozen RQ1 execution design

Primary feature set:

`F1_F2_F3`

Frozen models:

- Logistic Regression;
- Random Forest;
- XGBoost.

Frozen probability variants:

- uncalibrated;
- Platt calibration;
- isotonic calibration.

The Phase 11 execution grid therefore contains:

`10 projects × 3 models × 3 probability variants = 90 conditions`

Models and calibrators remain project-specific.

No model pooling or calibrator pooling is performed.

No performance-driven hyperparameter tuning, feature
selection, project selection, model selection, or calibration
selection is performed.

---

## 5. Chronological design

The Phase 11 implementation-validation split uses the frozen
chronological holdout design.

Outer split:

- first 80% of builds: development;
- final 20% of builds: evaluation.

Within the development period, the initial inner split is:

- first 80%: model fit;
- final 20%: calibration.

If the initial calibration period does not satisfy the frozen
minimum support requirements, the calibration window expands
backward one build at a time.

Minimum calibration support:

- positive rows >= 5;
- negative rows >= 5;
- failing builds >= 2;
- passing builds >= 2.

Minimum retained model-fit history:

- 100 builds.

Evaluation observations are never consumed for model fitting
or calibration fitting.

---

## 6. CORE data-pipeline validation

The ten-project CORE pipeline completed:

- raw-to-staging adaptation;
- canonical materialization;
- F1/F2/F3 history-feature generation;
- F2 raw-source verification;
- F3 independent recomputation;
- real-data leakage perturbation testing;
- chronological development/evaluation materialization.

The final chronological split contains:

- 10 projects;
- 14,541,882 total observations;
- 14,045 total builds.

The outer evaluation partitions contain:

- 3,217,027 unique evaluation observations.

Prediction artifacts were generated independently for all
three models, giving:

- 30 prediction files;
- 9,651,081 model-specific prediction rows.

---

## 7. Calibration-support validation

All ten selected projects satisfy the frozen calibration
support rule after applying the frozen backward-expansion
procedure.

Summary:

- sufficient projects: 10;
- insufficient projects: 0;
- projects requiring backward expansion: 2.

Projects requiring expansion:

- `datasets/Angel-ML@angel`: 7 builds;
- `datasets/Graylog2@graylog2-server`: 1,592 builds.

Graylog retains:

- 755 model-fit builds;
- 2,179 calibration builds.

This remains above the frozen 100-build model-fit floor.

Calibration support is not used as a project-selection
criterion.

---

## 8. RQ1 execution result structure

The ten-project CORE RQ1 implementation run completed
successfully.

Generated structure:

- projects: 10;
- models: 3;
- metric conditions: 90;
- model diagnostics: 30;
- prediction files: 30;
- calibration-bin rows: 390.

All 90 conditions have status:

`AVAILABLE`

All 30 model-level calibration diagnostics have status:

`SUFFICIENT`

The Graylog outer evaluation partition contains no observed
failure rows.

According to the frozen undefined-metric policy, its
ROC-AUC and Average Precision values are reported as NA.

Probability-quality and calibration metrics remain defined.

This condition does not affect project eligibility.

---

## 9. Independent output audit

Phase 11 outputs were independently audited after execution.

### 9.1 Prediction/test integrity

All 30 prediction artifacts were compared against their
corresponding outer evaluation partitions.

Verified:

- row identity;
- row ordering;
- build identity;
- test identity;
- target preservation;
- evaluation build count;
- failure-row count;
- probability finiteness;
- probability bounds `[0, 1]`.

Result:

**PASS**

Observed:

- prediction files audited: 30;
- unique evaluation rows: 3,217,027;
- prediction rows audited: 9,651,081;
- identity mismatches: 0;
- invalid probabilities: 0.

### 9.2 Independent metric recomputation

All RQ1 metrics were independently recomputed directly from
saved prediction artifacts without calling the RQ1 evaluator.

Verified metrics:

- ROC-AUC;
- Average Precision;
- Brier Score;
- Log Loss;
- ECE;
- mean predicted probability;
- calibration bias;
- absolute calibration bias.

Result:

**PASS**

Verified:

- conditions: 90;
- metric comparisons: 810;
- expected Graylog ROC-AUC NA values: 9/9 matched;
- expected Graylog Average Precision NA values: 9/9 matched.

Maximum numerical differences were at floating-point rounding
scale and below `1e-12`.

### 9.3 Calibration-bin audit

All stored non-empty calibration bins were independently
recomputed from saved predictions.

Verified:

- bin IDs;
- equal-width bin boundaries;
- counts;
- weights;
- mean predicted probabilities;
- observed failure rates;
- absolute gaps;
- weighted gaps;
- ECE aggregation.

Result:

**PASS**

Verified:

- RQ1 conditions: 90;
- calibration-bin rows: 390.

### 9.4 Diagnostics/calibration-split audit

Calibration splits and support diagnostics were independently
recomputed directly from development data.

Verified:

- initial model-fit boundary;
- backward expansion;
- final model-fit boundary;
- calibration boundary;
- row counts;
- build counts;
- chronological ordering;
- calibration support;
- diagnostic consistency across all three models.

Result:

**PASS**

Verified:

- projects: 10;
- diagnostic rows: 30;
- sufficient projects: 10;
- projects requiring expansion: 2;
- maximum expansion: 1,592 builds.

---

## 10. Resource validation

The production ten-project RQ1 run completed in:

- wall-clock time: 309.50 seconds;
- maximum resident set size: 4,115,660,800 bytes;
- peak memory footprint: 4,306,433,152 bytes;
- swaps: 0.

Generated CORE RQ1 artifacts occupy approximately:

- 990 MB.

A prior largest-project smoke using JMRI also completed all
three frozen models without swapping.

No concurrent project/model execution was required.

---

## 11. Regression validation

Before reproducibility rerun:

- CORE RQ1 targeted regression tests: 64 passed;
- complete repository regression suite: 223 passed;
- failed tests: 0.

Environment:

- Python: 3.9.6;
- architecture: arm64;
- NumPy: 2.0.2;
- pandas: 2.3.3;
- SciPy: 1.13.1;
- scikit-learn: 1.6.1;
- XGBoost: 2.1.4;
- PyYAML: 6.0.3.

Repository state before the reproduction run was clean and
synchronized with the upstream branch.

---

## 12. Full reproducibility rerun

A second complete ten-project RQ1 run was executed into an
isolated output directory without overwriting the production
artifacts.

The rerun completed:

- 10 projects;
- 30 model runs;
- 90 RQ1 conditions;
- 30 prediction files.

Rerun resources:

- wall-clock time: 301.24 seconds;
- maximum resident set size: 4,571,660,288 bytes;
- peak memory footprint: 4,043,582,400 bytes;
- swaps: 0.

Production and rerun artifacts were compared independently.

Results:

- prediction files compared: 30;
- prediction rows compared: 9,651,081;
- maximum prediction-probability difference: `0.000e+00`;
- maximum summary-metric difference: `0.000e+00`;
- maximum calibration-bin difference: `0.000e+00`;
- maximum diagnostics difference: `0.000e+00`.

Result:

**PASS**

The Phase 11 CORE RQ1 execution is exactly reproducible under
the recorded environment and frozen configuration.

---

## 13. Frozen artifact fingerprints

Production summary artifacts:

`results/core/rq1/rq1_metrics.csv`

SHA256:

`8828a2e30a95e8cc52046115671cbc6679a9cf97ef4e57f1dc223660396980b0`

`results/core/rq1/calibration_bins.csv`

SHA256:

`2f97994ef7b8d199fa40aa949b1fb8eeb33e6fe7a120843976215cc7c6bcb099`

`results/core/rq1/model_diagnostics.csv`

SHA256:

`74787ffd63072dc8ab287dd60c2a58c3c1fdfc0d28f77b100d45ae9f18174b2f`

Key frozen execution artifacts:

`configs/core_rq1_v1.yaml`

SHA256:

`4a7f3664d4ec4b39ee1ceee9dfda362f94e113016fac9fc6aca73a4dae643918`

`configs/core_projects_v1.yaml`

SHA256:

`6a2c035cedc1cbc87783908ff4ad8522bc2cb7a6a21b91dc9b7586954b18ccc2`

`configs/rq1_v1.yaml`

SHA256:

`af0e4636b948ccaf33732d4708923d1e387c184fc73b1f3b0b83c6b34d67baac`

`configs/rq1_models_v1.yaml`

SHA256:

`d380f67961f102e2ba2e9c2b126f43af5135a11bec3e69e472af73fc4d111877`

`configs/calibration_v1.yaml`

SHA256:

`69a63b00133a49fb2085c310c7d0a6748fc6d8e88b63e3b902184ab9d5f673af`

`configs/core_features_v1.yaml`

SHA256:

`d4ca8bda5846c01ba02878372059c4a708ce25d3b3305275cf58559d12906b50`

`src/tcp_ci/core_rq1_runner.py`

SHA256:

`90a5aa27ed0e77e66d87db590ad71f31b919e9ac1ee9add3233fd7f30964660d`

`src/tcp_ci/rq1_pipeline.py`

SHA256:

`b7c5f10111c056e3f383fd33f92e4e5cc903811da67714cd818048d1384d257c`

`src/tcp_ci/rq1_evaluator.py`

SHA256:

`ee5621fb5c3075cac4e2ecf4f4779f2c0c66622d4a4cb1f76179f54bcc4b1d24`

---

## 14. Interpretation boundary

Phase 11 demonstrates that the frozen CORE RQ1 implementation:

- scales from the five-project pilot to ten frozen projects;
- preserves chronology and leakage controls;
- executes all frozen probability models;
- executes both frozen calibration methods;
- produces structurally valid probability outputs;
- produces independently reproducible metrics and reliability
  bins;
- runs within the observed machine-resource envelope;
- is exactly reproducible under the recorded environment.

Phase 11 does **not** establish:

- which model is best;
- which calibration method is best;
- whether calibration improves RQ1 outcomes overall;
- operational test-selection effectiveness;
- cost reduction;
- failure recall under a selection budget;
- final longitudinal thesis evidence.

Those questions require the subsequent frozen walk-forward
and risk-bounded decision experiments.

---

## 15. Phase 11 closure

Phase 11 exit criteria are satisfied.

Completed milestones include:

- frozen structural project eligibility;
- frozen ten-project execution set;
- ten-project CORE data materialization;
- F1/F2/F3 leakage validation;
- chronological input preparation;
- ten-project RQ1 execution;
- resource validation;
- independent output auditing;
- full repository regression validation;
- independent exact reproducibility rerun.

Phase 11 is therefore closed as:

**Multi-project core-scale implementation validation — PASS**

The next research stage must preserve the Phase 11 frozen
methodology unless a documented protocol amendment explicitly
changes it.

The next primary experimental objective is to implement and
freeze the final longitudinal walk-forward evaluation design
before making final RQ1 claims or proceeding to downstream
risk-bounded RQ2 evidence.
