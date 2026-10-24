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
- Sensitivity protocol: PROPOSED.
- Implementation update: COMPLETED (candidate prototype; not M2-frozen).
- Acceptance tests: Unit and regression tests PASS (329 tests); longitudinal sensitivity validation PENDING.
- M2-01 freeze: NOT APPROVED.

## Freeze Sign-off

**Decision review:** PENDING

**Code/test review:** PENDING

**Protocol compatibility review:** PENDING

**M2 Method Freeze:** NOT APPROVED
