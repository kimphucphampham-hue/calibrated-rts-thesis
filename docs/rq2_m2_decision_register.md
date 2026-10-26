# RQ2 M2 Decision Register

**Status:** DRAFT — NOT M2 FROZEN

**Scope:** RQ2 risk-bounded regression test selection and baseline comparison.

**Authoritative protocol:** `docs/protocol_v1.md`

## Purpose

This document records unresolved methodological decisions before the final longitudinal walk-forward evaluation.

A successful prototype test suite does not constitute methodological freeze.

## Decision Register

| ID | Topic | Proposed direction | Status |
|---|---|---|---|
| M2-01 | Historical cost fallback | Strictly earlier-build positive-cost fallback; investigate zero values | PENDING |
| M2-02 | Zero total predicted risk | Conservative Retest All fallback for consideration | PENDING |
| M2-03 | MILP execution policy | Small-build oracle; bounded runtime; report solver failures | PENDING |
| M2-04 | Baseline matching | Matched count primary; budget-capped sensitivity | PENDING |
| M2-05 | Random repetitions | Stable seeds; prespecified repetitions and aggregation | PENDING |
| M2-06 | Historical cold-start | Explicit causally available fallback | PENDING |
| M2-07 | Static ML definition | Fixed model and calibrator before evaluation window | PENDING |
| M2-08 | Walk-forward observability | Separate retrospective full-information evaluation from partial-observation simulation | PENDING |

## Freeze Acceptance Criteria

For every decision:

1. Record the final policy and rationale.
2. Confirm compatibility with Protocol v1 or document a formal amendment.
3. Identify the implementation and configuration affected.
4. Add unit/integration tests for important edge cases.
5. Define diagnostic reporting for excluded or exceptional builds.
6. Review and approve the policy before running final experiments.

## Evidence Boundaries

- Phase 11 chronological holdout results are implementation-validation evidence.
- RQ2 selector and baseline prototypes are not final longitudinal results.
- Final RQ2/RQ3 claims require chronological walk-forward evaluation.
- No decision in this draft is considered frozen solely because an implementation already exists.

## M2-01 — Historical Cost Policy: Evidence Review

**Status:** CANDIDATE POLICY — NOT FROZEN

### Evidence

The ten-project CORE evaluation audit identified:

- 3,217,027 evaluation observations.
- 778 missing historical costs.
- 61,869 zero historical costs.
- No negative or infinite historical costs.

All 778 missing historical costs occurred at `test_age_builds = 0`.

The Facebook/Buck investigation identified:

- 433,838 training observations.
- 127,456 evaluation observations.
- 61,838 zero historical costs in evaluation.
- 61,804 of these also had zero realized duration.
- All 61,838 had positive test age.
- No duplicate project/build/test keys within either split.
- No build indices mapped to multiple build IDs within either split.
- All 88,618 positive training durations and all 26,658 positive evaluation durations were multiples of 1,000 ms.
- Minimum observed positive duration: 1,000 ms.

Canonical materialization preserves upstream duration values without additional rounding. Historical mean is computed from previous execution observations.

### Candidate Primary Policy

1. Preserve strictly positive, finite historical test costs.
2. For missing historical costs, use a fallback derived exclusively from eligible strictly earlier builds of the same project.
3. For zero historical costs, consider a strictly positive floor informed by the project's observed duration resolution.
4. Treat 1,000 ms as the candidate floor for Facebook/Buck, not automatically for other projects.
5. Reject invalid negative or infinite costs.
6. If a required fallback is unavailable, fail explicitly and record the affected build.
7. Never use target-build realized duration to determine selection-time cost.

### Sensitivity Analysis

Compare at least:

- Resolution-informed positive-floor treatment of historical zeros.
- The existing project-level earlier-positive-duration median fallback for zeros.

Keep the missing-cost policy unchanged between these conditions when assessing the impact of zero-cost handling.

Report the impact on selected tests, estimated costs, and realized evaluation metrics.

### Remaining Acceptance Requirements

- Verify duration-resolution characteristics across the remaining CORE projects.
- Specify the final positive-floor rule and its causal availability.
- Review whether the policy requires a formal protocol amendment.
- Implement and test the approved rule.
- Validate decision-time temporal boundaries.
- Complete sensitivity evaluation without using evaluation outcomes to choose the primary method.

**Decision:** PENDING

**M2 Freeze Approval:** NOT GRANTED

### Cross-Project Duration Resolution Audit

**Status:** COMPLETED — 10 CORE project training partitions audited.

The minimum observed strictly positive duration was:

- 1 ms: Angel-ML, EvoMaster, Graylog2, JMRI, SonarSource, ShardingSphere, Camunda, and spf4j.
- 39 ms: CompEvol/beast2.
- 1,000 ms: Facebook/Buck.

All positive training durations in Facebook/Buck were multiples of 1,000 ms. Other projects included positive durations not divisible by 1,000 ms.

The minimum observed positive duration is not necessarily the instrument's measurement resolution.

### Candidate Zero-Cost Policy

**Status:** PROPOSED — NOT APPROVED.

Preserve finite, strictly positive historical estimates.

Handle missing historical estimates separately using a causally available same-project fallback.

For historical zero estimates, evaluate a project-specific positive floor derived from development-period data available before the evaluation target.

Candidate Phase 11 development floors are 1 ms for eight projects, 39 ms for CompEvol, and 1,000 ms for Facebook/Buck.

The primary floor rule must be prespecified and its temporal availability verified before final walk-forward evaluation.

### Required Sensitivity Comparison

Compare:

1. Project-specific positive floor for historical zeros.
2. Earlier-positive-duration project median fallback for historical zeros, as implemented in the current prototype.

Keep other selection inputs and policies unchanged.

Evaluate differences in selected test IDs, estimated costs, Failure Recall, Test Selection Ratio, Runtime Ratio, and Missed Failure Rate.

### Decision Status

- Cross-project duration audit: COMPLETE.
- Primary cost policy: CANDIDATE.
- Sensitivity protocol: PHASE 11 PROTOTYPE EXECUTED; final longitudinal protocol PENDING.
- Implementation update: COMPLETED (candidate prototype; not M2-frozen).
- Acceptance tests: Unit and regression tests PASS (337 tests); final longitudinal sensitivity validation PENDING.
- M2-01 freeze: NOT APPROVED.

## M2-01 — WP5.5 Sensitivity Evidence

**Status:** EMPIRICAL PROTOTYPE EVIDENCE — NOT FROZEN.

### Experiment Scope

- Evidence type: Phase 11 chronological holdout sensitivity pilot.
- Projects: Facebook/Buck and Angel-ML.
- Model: Logistic Regression.
- Calibration: Platt.
- Risk coverage levels: 0.90, 0.95, and 0.99.
- Chronological sampling: 12 evenly spaced evaluation builds per project.
- Total: 24 unique builds and 72 paired policy comparisons.
- Policies: project-specific positive floor and earlier-build positive-duration median.
- Missing historical cost treatment is shared between policies.
- Evaluation history uses retrospective full-information observations.
- Final RQ2/RQ3 longitudinal walk-forward evidence is not yet available.

### Reproducibility and Code Validation

- Cost adapter and candidate policy committed in db91195.
- Sensitivity runners and integration tests committed in abff05d.
- Full regression suite: 337 tests passed.
- Four CSV artifacts reproduced with identical SHA-256 hashes
  across the original artifacts and two repeated runs.
- Recorded runtime accounting was reconciled on all three
  changed-selection cases.
- These checks validate the prototype, not methodological freeze.

### Paired Sensitivity Findings

- Completed paired comparisons: 72.
- Comparisons with changed selected test IDs: 3.
- Distinct affected builds: 2.
- Comparisons with changed estimated selected costs: 35.
- Skipped builds: 0.
- Paired comparisons containing actual failures: 18; all were from Angel-ML.
- No sampled Facebook/Buck build contained actual failures; therefore failure-detection effects of the zero-cost policy remain unestablished.
- All three changed-selection comparisons occurred in Facebook/Buck.
- Angel-ML had no historical-zero costs in the sampled builds,
  and selection was identical under the two policies.

Changed-selection cases:

| Project | Build | Rho | Floor selected | Median selected | RTR delta (floor minus median) |
|---|---:|---:|---:|---:|---:|
| Facebook/Buck | 692 | 0.95 | 498 | 296 | -0.350105 |
| Facebook/Buck | 692 | 0.99 | 695 | 680 | -0.058281 |
| Facebook/Buck | 754 | 0.90 | 687 | 678 | -0.044543 |

### Runtime Accounting and Interpretation

In the three changed-selection comparisons:

- All tests selected only by the positive-floor policy
  had recorded realized duration of zero.
- Tests selected only by the historical-median policy
  had positive aggregate recorded duration.
- The RTR differences were reconciled against per-test
  realized execution durations.
- All three affected build-policy comparisons had zero
  actual failed tests.

The results establish that zero-cost handling can alter
estimated costs and selected test sets in this pilot.

They do not establish improved failure detection or actual
CI wall-clock savings. Recorded zero execution duration must
not be interpreted as zero real-world scheduling, startup,
or orchestration overhead.

### Outstanding Methodological Requirements

1. Prespecify and justify the primary zero-cost policy
   independently of evaluation outcome comparisons.
2. Verify availability of floor inputs at each decision time
   under the final longitudinal evaluation design.
3. Assess sensitivity across the ten CORE projects,
   including projects with few or no historical zeros.
4. Evaluate costs and failure-detection metrics across
   appropriate longitudinal windows.
5. Resolve zero-total-predicted-risk policy (M2-02).
6. Resolve historical cold-start behavior (M2-06).
7. Specify retrospective full-information evaluation
   separately from partial-observation simulation (M2-08).
8. Review protocol compatibility and final diagnostic
   reporting before freeze sign-off.

### Decision

**Primary policy:** NOT SELECTED.

**M2-01 Freeze Approval:** NOT GRANTED.

**Evidence disposition:** Retain as reproducible
implementation-validation and sensitivity evidence.

## Freeze Sign-off

**Decision review:** PENDING

**Code/test review:** PENDING

**Protocol compatibility review:** PENDING

**M2 Method Freeze:** NOT APPROVED
