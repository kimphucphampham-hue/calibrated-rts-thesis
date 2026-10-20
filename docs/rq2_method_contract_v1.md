# RQ2 Risk-Bounded Selection — Method Contract v1

Status: DRAFT — NOT FROZEN
Milestone: M2 Method Freeze
Scope: RQ2 implementation and pre-experiment validation

## 1. Scientific boundary

RQ2 minimizes estimated test execution cost subject to
retaining at least rho of total predicted failure risk.

Frozen risk-retention levels:
- 0.90
- 0.95
- 0.99

This document does not report final RQ2 results or
replace the longitudinal walk-forward protocol.

## 2. Inputs

Decision-time fields:
- project_id
- build_id
- build_index
- test_id
- predicted failure probability
- historical estimated execution cost
- rho

RQ1 prediction variants:
- p_uncalibrated
- p_platt
- p_isotonic

No final model/calibration winner is selected from
Phase 11 holdout results.

The decision function must not receive current-build:
- failed
- execution_time_ms

These fields are used only for post-selection evaluation.

## 3. Historical cost

Existing feature:
execution_time_mean_history_ms

Source:
data/processed/tcp_ci/core_split/<project>/test.csv

The feature is calculated from prior executions of
the same test. Build-level temporal ordering must
be verified before treating it as leakage-safe.

Audit observations across 10 CORE projects:
- 3,217,027 evaluation rows
- 778 missing historical costs
- 61,869 zero historical costs
- 0 negative or infinite historical costs

All 778 missing costs have test_age_builds = 0.

Open decisions:
- cold-start fallback from strictly earlier data
- zero-cost handling consistent with c_i > 0
- behavior when historical fallback is unavailable
- positive cost floor, if adopted

No current-build realized execution time may be used
to fill or modify decision-time costs.

## 4. Selector

Objective:
minimize sum(selected estimated costs)

Constraint:
sum(selected predicted probabilities)
>= rho * sum(all predicted probabilities)

Planned implementations:
- RiskDensitySort heuristic, O(n log n)
- SciPy MILP exact optimization where feasible

The heuristic is not claimed to be globally optimal.
Solver status must distinguish proven optimal solutions
from time-limited or otherwise non-optimal outcomes.

Open decisions:
- deterministic tie-breaking: prototype sorts by descending risk density, then ascending test_id; requires M2 review
- numerical tolerance and epsilon
- zero-total-risk behavior
- invalid-probability handling
- MILP size/time limits
- solver failure/fallback policy

## 5. Baselines

Planned comparisons from the advisor proposal:
- Retest All
- Random Selection
- Failure-History
- History/Duration
- Static ML selector
- Uncalibrated risk selector

DANTE is optional, subject to compatibility and scope.

The baseline comparison protocol must define matching
selection counts or budgets without outcome leakage.

## 6. Evaluation

Required RQ2 metrics:
- Failure Recall (FR)
- Test Selection Ratio (TSR)
- Runtime Ratio (RTR)
- Missed Failure Rate (MFR)

Actual outcomes and realized runtimes are read only
after a selection is determined.

Per-project results are primary.
Final RQ2 evidence requires frozen walk-forward execution.

## 7. Validation gates

Before method freeze:
- build-level temporal ordering verified
- joins validated for the CORE dataset
- cost policy documented and unit-tested
- risk-retention constraint tests pass
- numerical and empty-build edge cases covered
- heuristic compared with exact solutions on small cases
- baseline behavior and fairness rules tested
- RQ1 regression tests remain passing
- Chapter 3 method draft reviewed
- implementation/configuration provenance recorded

## 8. Freeze boundary

This DRAFT is not M2 approval.

No final RQ2 comparisons should be interpreted as
confirmatory before method decisions, test criteria,
and walk-forward setup are frozen and reviewed.

Any change to frozen scientific rules must follow
the protocol amendment procedure.


## 9. Implementation and Cost Policy Review

Status: DRAFT — PENDING M2 REVIEW

### 9.1 Solver implementation evidence

The selector prototype contains:
- RiskDensitySort with deterministic test_id tie-breaking.
- SciPy 0-1 MILP with a normalized risk constraint.
- Independent post-solution risk-retention validation.

Normalization uses p_i / sum(p_i) for positive total risk.
This preserves the mathematical constraint while improving
numerical conditioning for small probabilities.

46 selector unit tests passed during implementation
validation. This is not final longitudinal evidence.

### 9.2 Proposed historical cost handling

Candidate decision-time cost policy:

1. Use finite, strictly positive historical per-test mean
   when available.
2. For missing historical mean (cold start), use the median
   of strictly positive realized durations from earlier
   builds of the same project.
3. Apply the same earlier-build fallback to historical
   zero-cost observations.
4. Reject negative and nonfinite supplied cost values.
5. When no eligible positive historical duration exists,
   fail explicitly; do not use target-build duration.

The fallback must use build_index strictly smaller than
the target build_index, including when multiple tests
belong to the same build.

These rules remain a proposal until reviewed and frozen.

### 9.3 Known methodological sensitivity

Zero historical costs are concentrated in Facebook/Buck.
Replacing zero values with a project-level historical
median can materially affect estimated selection cost.

Review the interpretation and potential sensitivity
analysis before M2 freeze.

No Phase 11 artifacts or frozen RQ1 features are modified.

## 10. Prototype Integration Validation

Status: IMPLEMENTATION VALIDATION ONLY

### 10.1 Unit tests

The RQ2 selector and cost adapter passed 55/55
unit tests before the full regression-suite review.

### 10.2 Real-data historical fallback smoke test

Project: datasets/Angel-ML@angel
Project folder: datasets__Angel-ML__angel

Prediction source: Logistic Regression, p_platt
Target build_index: 279
Candidate tests: 25
Earlier historical observations: 9,345
Eligible positive historical durations: 9,336
Missing historical cost values: 6
Historical fallback median: 24,064 ms
Policy rho: 0.95

RiskDensitySort:
- Selected tests: 10
- Retained predicted risk: 0.39998302569941485
- Estimated selected cost: 166600.8717318128 ms
- Status: heuristic

SciPy MILP:
- Selected tests: 7
- Retained predicted risk: 0.39924888490325755
- Estimated selected cost: 166355.22018348624 ms
- Status: optimal

Required predicted risk: 0.3960660730240417

Validation assertions passed:
- Correct project_id mapping
- Strictly earlier historical builds
- Historical fallback matches eligible positive median
- Existing positive historical costs unchanged
- Positive selector input costs
- Risk-retention constraint for both selectors

These observations come from a researcher-executed,
read-only Terminal smoke test, not a frozen
longitudinal evaluation run.

### 10.3 Interpretation boundary

The p_platt/Logistic Regression combination was used
only for technical integration validation.

No best model, calibration, or selector is selected
from this smoke test.

The test uses the full recorded prior-build dataset,
not an explicitly simulated partially observable
history under earlier test-selection decisions.

The missing/zero historical cost fallback remains
a proposed policy, not an approved M2 method.

The Facebook/Buck zero-cost sensitivity and the
walk-forward history-update policy remain open
before formal method freeze.
