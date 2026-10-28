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

## M2-01 — WP5.7 Methodological Review

**Status:** REVIEW RECOMMENDATION — NOT APPROVED.

### Cross-Project Causal Availability

The WP5.7.3 audit examined all ten CORE project
train/evaluation partitions.

- All ten training partitions ended strictly before
  their respective evaluation partitions.
- All ten training partitions contained eligible
  strictly positive execution durations.
- Training-derived positive minimum and median
  costs were available for all ten projects.
- Evaluation historical-zero observations: 61,869.
- Evaluation missing-cost observations: 778.
- These totals reconcile with the earlier M2 audit.

Historical-zero observations were distributed as follows:

- Facebook/Buck: 61,838.
- SonarQube: 30.
- ShardingSphere: 1.
- All other CORE projects: 0.

Facebook/Buck therefore accounts for approximately
99.95% of the historical-zero evaluation observations.

The audit establishes temporal availability of
training-derived floors for the Phase 11 holdout
partitions. It does not establish measurement
resolution or complete longitudinal observability.

### Review Recommendation

**Recommended primary candidate:**
historical positive-duration median.

**Recommended sensitivity comparator:**
project-specific positive floor.

**Decision status:** RECOMMENDED FOR REVIEW ONLY.

This recommendation favors an explicitly defined
earlier-build positive-duration median because its
meaning does not depend on interpreting the minimum
observed positive duration as a verified measurement
resolution.

The median remains an estimate rather than ground
truth. It may overestimate the cost of very fast tests.

The positive floor remains important for sensitivity
analysis because WP5.5 demonstrated changes in
estimated costs and selected test identities.

Neither policy is selected because it produced
better observed TSR, RTR, or Failure Recall in the
Phase 11 pilot.

### Proposed Primary Cost Rules

Subject to M2 review:

1. Preserve finite, strictly positive historical
   per-test estimated costs.
2. For missing historical costs, consider a
   same-project median of eligible strictly
   positive durations from earlier builds.
3. For zero historical costs, consider the same
   earlier-build positive-duration median.
4. Never use current-build realized duration to
   calculate selection-time costs.
5. Explicitly report targets without eligible
   positive history; the cold-start rule remains
   unresolved under M2-06.
6. Preserve project-level results and report the
   positive-floor sensitivity comparison.

These are proposed rules, not an approved method.

### Timing and Interpretation

This methodological recommendation was developed
after inspection of Phase 11 holdout sensitivity
results but before final longitudinal evaluation.

It must not be described as prespecified before
the Phase 11 pilot.

The final primary policy, comparator, and decision
criteria must be recorded before confirmatory
longitudinal evaluation is executed.

### Protocol and Observability Boundaries

Protocol v1 requires selection-time costs to be
derived using information available before
execution of the target test.

For the Phase 11 holdout, the project-specific
training-derived floor is temporally available.

Final longitudinal walk-forward evaluation must
verify cost availability at each relevant cutoff.

The retrospective full-information assumption
does not establish partial-observation deployability.
The latter remains a separate M2-08 issue.

The existing RQ2 Method Contract is DRAFT and
already describes the earlier-build median rule
as a candidate. Any final choice must undergo
formal Method Contract compatibility review.

### Outstanding Decision Gates

- Confirm the primary cost rule and tie it to
  a specified information-availability model.
- Resolve M2-06 for absent positive history.
- Resolve M2-08 observability assumptions.
- Verify final longitudinal causal boundaries.
- Review zero-duration measurement limitations.
- Review required protocol and contract changes.
- Approve the primary and sensitivity policies
  before final confirmatory RQ2 evaluation.

### Review Disposition

**Recommended primary candidate:** Historical Median.

**Recommended comparator:** Positive Floor.

**Primary policy:** NOT SELECTED.

**M2-01 Freeze Approval:** NOT GRANTED.

## M2-01 — WP5.8 Acceptance Gate Review

**Status:** TECHNICAL VALIDATION COMPLETED;
METHODOLOGICAL APPROVAL PENDING.

### Technical Evidence

The existing historical cost adapter was reviewed
at source level and tested with synthetic,
decision-time-controlled observations.

Behavioral checks confirmed:

- Finite positive historical per-test costs
  are preserved without modification.
- Missing historical costs use the median
  of eligible positive earlier-build durations.
- Under the default policy, historical
  zero costs use that same median.
- Under the candidate floor policy, historical
  zeros use the configured positive floor,
  while missing costs still use the median.
- Current and future build durations
  are excluded from historical fallback.
- Other-project durations cannot influence
  same-project fallback.
- Test order and DataFrame index are preserved.
- Invalid floor configurations are rejected.
- Missing eligible positive history raises
  an explicit error when median is required.

The controlled behavioral audit passed.
The previously executed regression suite
contained 337 passing tests.

### Acceptance Gate Assessment

| Gate | Status | Evidence or limitation |
|---|---|---|
| Positive historical cost preservation | PASS | Source and behavioral checks |
| Missing-cost median fallback | PASS | Source and behavioral checks |
| Zero-cost median fallback | PASS | Source and behavioral checks |
| Strictly earlier same-project history | PASS for tested paths | Adapter checks and temporal audits |
| Ten-project Phase 11 training boundary | PASS | WP5.7.3 |
| Positive-floor comparator implementation | PASS | WP5.5 and WP5.8.2 |
| Pilot reproducibility | PASS | Four CSV hashes reproduced |
| Exceptional-build diagnostics | PENDING | Reporting policy not yet approved |
| No-positive-history treatment | PENDING | M2-06 interaction |
| Partial-observation interpretation | PENDING | M2-08 |
| Primary method approval | NOT GRANTED | Formal review outstanding |

### Interpretation Boundaries

The adapter's explicit error when no eligible
positive history exists is validated behavior,
not an approved final cold-start policy.

The Phase 11 retrospective full-information
history is distinct from history observable
under deployed regression-test selection.

The source-level adapter checks do not,
by themselves, establish upstream historical
feature correctness in every longitudinal cycle.

The recommended Historical Median primary
candidate was documented after inspection
of Phase 11 sensitivity evidence and before
final confirmatory longitudinal evaluation.

No policy is selected based on favorable
Phase 11 runtime or failure-recall outcomes.

### Outstanding Approval Requirements

1. Specify the scope and format of diagnostics
   for builds where cost estimation fails.
2. Review the no-positive-history condition
   jointly with M2-06.
3. Define the information-availability model
   required by M2-08.
4. Check compatibility of the final proposed
   cost policy with Protocol v1 and the
   RQ2 Method Contract.
5. Record and approve the primary policy
   and sensitivity comparator before
   final confirmatory evaluation.

### Acceptance Disposition

**Technical adapter validation:** PASS
for the tested scope.

**Recommended primary candidate:** Historical Median.

**Recommended sensitivity comparator:** Positive Floor.

**Primary policy:** NOT SELECTED.

**M2-01 Freeze Approval:** NOT GRANTED.

## M2-06 — WP5.9 Cold-Start Evidence and Review

**Status:** EVIDENCE RECORDED — POLICY PENDING.

### Ten-Project Cold-Start Exposure Audit

The WP5.9.1 audit examined the training partitions
of all ten CORE projects.

- First recorded training build index: 1
  in every project.
- First build containing a positive realized
  execution duration: build 1 in every project.
- Early training builds without eligible
  strictly earlier positive history: 10.
- Training observations requiring zero/missing
  cost fallback in those builds: 6,518.
- Projects with this exposure: 10.
- Projects with positive training history
  before Phase 11 evaluation: 10.

Breakdown of exposed training observations:

| Project | Exposed observations |
|---|---:|
| JMRI | 4,245 |
| ShardingSphere | 762 |
| Facebook/Buck | 579 |
| Camunda | 362 |
| SonarQube | 288 |
| spf4j | 91 |
| EvoMaster | 81 |
| beast2 | 53 |
| Graylog2 | 30 |
| Angel-ML | 27 |
| Total | 6,518 |

These are historical training observations,
not failed final RQ2 selection attempts.

The first build cannot use its own realized
durations to construct decision-time costs.

Under retrospective full-information history,
positive project-level durations are available
after the completion of build 1.

### Candidate Operational Policy

**Recommendation for review:**
Conservative Retest All when an evaluation build
requires a historical positive-duration median
but has no eligible earlier positive history.

The current cost adapter should continue
to fail explicitly when median estimation
is impossible.

If approved, the evaluation orchestration
layer should intercept this specific condition
and select all candidate tests.

This is a proposed exception policy,
not an approved M2-06 decision.

### Proposed Diagnostic Requirements

Every affected evaluation build should report:

- Project and build identifiers.
- Decision-time history availability.
- Reason for unavailable cost fallback.
- Selected fallback action.
- Total candidate test count.
- Failure-detection and runtime metrics,
  where ground-truth observations permit.
- Explicit fallback status in aggregate reporting.

Evaluation builds must not be silently removed
from denominators because of cold-start.

Training or warm-up builds should be reported
separately from confirmatory evaluation builds.

### Interpretation and Dependencies

Retest All avoids fabricating unavailable
selection-time cost estimates.

It does not demonstrate cost savings or
optimized test selection.

The proposal must be assessed for compatibility
with Protocol v1 and the RQ2 Method Contract.

M2-08 must separately define retrospective
full-information and partial-observation
history semantics.

The zero-total-predicted-risk rule in M2-02
is a distinct decision and must not be
implicitly settled by this proposal.

### Decision Status

**M2-06 recommended fallback:** Conservative Retest All.

**M2-06 approval:** PENDING.

**M2-01 primary policy:** NOT SELECTED.

**M2-06 decision status:** PENDING — NOT FROZEN.

## Freeze Sign-off

**Decision review:** PENDING

**Code/test review:** PENDING

**Protocol compatibility review:** PENDING

**M2 Method Freeze:** NOT APPROVED
