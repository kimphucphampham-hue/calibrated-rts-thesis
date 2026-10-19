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

---

## 16. Traceability addendum (Phase 12.F1 documentation reconciliation)

This section was appended on 2026-10-19 during Phase 12.F1.
Sections 1–15 are unchanged.

It reconciles this document with a later, uncommitted draft
milestone (`phase11_core_scale_milestone.md`), the Phase 12
freeze audit (`phase12_freeze_audit.md`) and the Phase 11.9D
Git audit (`phase11_9d_git_sync_audit.md`).

These review records are not part of the repository. They are
kept unmodified in a separate `thesis-review-logs/` folder
outside the Git working tree, together with the Phase 12.F1
reconciliation and pre-commit review reports. They are working
records, not thesis evidence; every statement this addendum
relies on is restated here.

### 16.1 Authoritative document

This document, `docs/phase11_core_scale_milestone_v1.md`, is
the authoritative Phase 11 milestone record.

The draft `phase11_core_scale_milestone.md` is superseded. Its
unique, verified content is transferred into §16.3–§16.5
below. The draft was never committed. It is retained
unmodified outside the repository in `thesis-review-logs/`
and is not a second authoritative record.

### 16.2 Commit provenance

| Role | Commit | Evidence |
| --- | --- | --- |
| Execution / implementation baseline | `4b9d6da61fe3a05de67db018e972f5acaa20bc06` (2026-10-18 17:04 +0700) | Last commit changing code or configs before the run. All 13 tracked files in the checksum manifest hash-match their blobs at this commit. Production result files (30 prediction CSVs and 3 summary CSVs) have modification times 2026-10-18 20:13–20:19, after this commit. |
| Milestone documentation commit | `98eba9eec483709f354066e16ca3e98b8e5bc2d3` (2026-10-18 20:57 +0700) | `git diff --stat 4b9d6da 98eba9e` adds only this document. Code and configs are byte-identical to `4b9d6da`. |
| Execution-config freeze | `4f044b7465a9396ba694daccb30dcf07981daefb` | Adds `configs/core_rq1_v1.yaml` and `tests/test_core_rq1_config.py`. |

The draft's label "pre-milestone implementation commit:
`98eba9e`" is superseded. `98eba9e` is a documentation-only
commit; the execution baseline remains `4b9d6da`, as stated
in §1.

### 16.3 Evidence-status flag

`primary_final_rq1_evidence = false`

Source: `configs/core_rq1_v1.yaml`, line 10.

Phase 11 results are implementation validation. They are not
final RQ1 evidence, not walk-forward evidence, and not a basis
for selecting a best model or calibration method (§14).

### 16.4 Sub-phase map

| Sub-phase | Content | Section |
| --- | --- | --- |
| 11.7A | Prediction/test integrity | §9.1 |
| 11.7B | Independent metric recomputation | §9.2 |
| 11.7C | Calibration-bin consistency | §9.3 |
| 11.7D | Diagnostics/calibration-split audit | §9.4 |
| 11.8A | Regression/reproducibility preflight | §11 |
| 11.8B | Independent full reproducibility rerun | §12 |

### 16.5 Additional statements transferred from the draft

Re-checked during Phase 12.F1 from the frozen
`results/core/rq1/*.csv` (gitignored, checksum-verified):

- `rq1_metrics.csv`: 90 rows, 10 projects, all `AVAILABLE`;
- `model_diagnostics.csv`: 30 rows, all `SUFFICIENT`;
- `calibration_bins.csv`: 390 rows;
- prediction files under `results/core/rq1/predictions/`: 30;
- ROC-AUC and Average Precision are NA for exactly 9
  conditions each, all `datasets/Graylog2@graylog2-server`;
- Graylog `observed_failure_rate` is 0 in all 21 stored bins;
- model-fit, calibration and evaluation build ranges do not
  overlap for any of the 30 diagnostic rows;
- model-fit starts at the first build and the three windows
  are contiguous, so the development period is fully covered;
- window sizes are identical across the three models of each
  project;
- backward expansion: Angel-ML 7 builds, Graylog 1,592 builds;
  the smallest retained model-fit window is 189 builds.

Carried over from the draft without re-verification in F1:

- the 11.8A full-suite run reported 14 dependency deprecation
  warnings alongside 223 passed tests.

### 16.6 Checksum manifest

The full SHA-256 manifest for this milestone is:

`docs/phase11_core_scale_checksums.sha256`

It has 46 entries: 7 configs, 6 source files, 3 summary CSVs
and 30 prediction files. Verify from the repository root with:

`shasum -a 256 -c docs/phase11_core_scale_checksums.sha256`

Provenance, coverage and exclusions are documented in
`docs/phase11_core_scale_checksums.md`.

The §13 fingerprints are a subset of this manifest and agree
with it.

### 16.7 Protocol amendments

Phase 11 executed under two protocol amendments whose files
both carry the title "Amendment 001". Unique register IDs are
assigned in `docs/protocol_amendments.md`:

- AMD-001: `docs/protocol_amendment_001_rq1_calibration.md`;
- AMD-002: `docs/protocol_amendment_001_core_project_eligibility.md`.

Both amendment files are unchanged by Phase 12.F1. AMD-002 was
created in `e6d00ff` and its eligibility rule 10 revised in
`f1a20a1`, before the project-set freeze `096d74c`; the
register records this revision history.

### 16.8 Documentation gaps carried forward

- Reliability diagrams for the CORE results have not been
  rendered, although `configs/calibration_v1.yaml` and AMD-001
  §10 require them. They can be drawn from the saved
  `results/core/rq1/calibration_bins.csv` without rerunning
  models. Deferred beyond Phase 12.F1.
- The 11.7 audit scripts and the 11.8B rerun outputs are not
  committed. Their evidence is the narrative in §9 and §12.
- The manifest does not cover the data-stage configs, the
  protocol or the amendments (see
  `docs/phase11_core_scale_checksums.md`).

### 16.9 Next stage

Consistent with §15 and `docs/protocol_v1.md` §10.2, the next
methodological stage is to specify, review and freeze the
final chronology-preserving walk-forward RQ1 protocol before
any walk-forward performance is inspected.

Its evaluation windows, retraining policy, calibration policy,
aggregation rules and project-eligibility rules must be frozen
first. Final RQ1 claims and risk-bounded RQ2 and retraining
RQ3 evidence follow it.

This ordering constrains evidence, not implementation.
Implementing and validating the risk-bounded selector (README
§23, items 5–6) need not wait for the walk-forward freeze.
Selector results obtained before that freeze are
implementation validation only, not final RQ2 evidence. This
addendum does not decide the order of the selector work and
the walk-forward freeze.

This addendum does not assign a phase number to that stage.
"Phase 12" in the RQ1 milestone schedule (01–20 September
2026) denotes "Document + freeze"; numbering the walk-forward
freeze as "Phase 12.0", as the draft did, would collide with
it. The phase number and date are left to the advisor-approved
schedule.

---

## 17. Post-F1 Reliability Evidence Reconciliation

This section was appended after Phase 12.F1 on 2026-10-19.

Sections 1–16 remain unchanged. In particular, §16.8 records
the documentation gaps that existed at the time of Phase 12.F1.
This section updates their subsequent status without rewriting
that historical record.

### 17.1 Reliability diagrams completed

The missing CORE RQ1 reliability diagrams identified in §16.8
were generated from the previously frozen:

`results/core/rq1/calibration_bins.csv`

using:

- `configs/calibration_v1.yaml`;
- `src/tcp_ci/rq1_plots.py`;
- `generate_reliability_figures()`.

The execution produced:

- 30 PNG diagrams;
- 10 projects;
- 3 models per project;
- 3 calibration variants per diagram;
- 1 figure index CSV.

No model retraining or modification of frozen prediction
artifacts was required.

### 17.2 Validation and integrity

The following checks passed:

- 18 RQ1 plotting unit tests;
- 8/8 structural figure validation checks;
- 30/30 PNG files generated;
- 31/31 figure and index SHA-256 checks;
- 46/46 original Phase 11 SHA-256 checks;
- visual review of 3 representative diagrams.

The complete set of 30 diagrams has not undergone independent
manual visual review.

The figure integrity manifest is:

`docs/phase11_core_reliability_figures.sha256`

Detailed provenance and interpretation limitations are recorded in:

`docs/phase11_core_reliability_provenance.md`

These two evidence documents were committed in:

`ac38440` — Document Phase 11 CORE reliability diagram evidence.

### 17.3 Evidence archive

The generated figures, figure index, source calibration-bin CSV
and figure checksum manifest were packaged into:

`phase11_reliability_evidence_2026-10-19.tar.gz`

SHA-256:

`b298958932aa958222da358294e0328a958a8efb913842ea023dc4b4e99a1d77`

The archived figure/index files passed 31/31 SHA-256 checks.
The archived calibration-bin CSV matched the local frozen source.

The archive is stored outside Git in the local
`../thesis-evidence-archive/` directory.

The repository stores the provenance document and checksums,
not the archived PNG data. Separate durable backup and archive
availability remain operational preservation requirements.

### 17.4 Independent audit and reproducibility records

The Phase 11.7A–D audit results and Phase 11.8B exact rerun
are documented in §9 and §12, respectively.

The separately preserved Phase 11 evidence ZIP also contains
the corresponding audit logs, rerun comparison report,
configuration and rerun outputs.

These records support the previously documented PASS results.
They do not establish that the original independent audit
scripts have been committed to Git or that the reproduction
procedure is portable to a new machine without adaptation.

The absence of committed audit scripts must not be confused
with an absence of prior audit execution.

### 17.5 Remaining documentation boundaries

The reliability-diagram generation gap in §16.8 is resolved.

The following limitations remain explicitly acknowledged:

- independent audit scripts are not preserved as tracked
  repository artifacts;
- the full rerun outputs are stored outside Git;
- local absolute paths in rerun configuration may require
  adaptation on another machine;
- the original 46-entry checksum manifest does not cover
  all data-stage configurations, protocols or amendments;
- a separate durable backup of the evidence archives
  has not yet been confirmed.

This reconciliation does not revise the frozen methodology,
project selection, models, calibration policy or results.

### 17.6 Scientific scope

The reliability diagrams and associated Phase 11 audit evidence
belong exclusively to CORE implementation validation on a
chronological holdout.

They are not final longitudinal walk-forward RQ1 evidence and
cannot establish a best model, best calibration method or final
research effectiveness claims.

Closure of the original Phase 12 Document + Freeze milestone
requires a separate review of its complete exit criteria.
This section does not independently declare that milestone closed.
