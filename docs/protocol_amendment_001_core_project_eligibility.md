# Protocol Amendment 001
## Phase 11 Core Project Eligibility Operationalization

### Status

Frozen before Phase 11 screening and before any prediction or
calibration performance is generated for additional TCP-CI
projects.

### Protocol relationship

This document is a formal amendment to Protocol v1.

It operationalizes the existing Protocol v1 project
inclusion/exclusion criteria for the Phase 11 core-scale
implementation-validation experiment.

It does not replace the final walk-forward project-eligibility
freeze.

## 1. What changed

Protocol v1 already requires project eligibility to be decided
before project-level model performance is inspected and
requires at least 8 eligible projects end to end, targeting 10.

Protocol v1 does not fully operationalize how the additional
TCP-CI projects are selected for the current chronological
holdout implementation.

This amendment freezes the operational Phase 11 rules in:

`configs/core_project_screening_v1.yaml`

A project is structurally eligible only when:

1. required source artifacts parse successfully;
2. raw execution staging-key fields are non-null;
3. every execution test identifier maps to `id_map.csv:value`;
4. every execution build links to unambiguous chronological
   build metadata with unique build identifiers;
5. build timestamps are parseable;
6. canonical `(project_id, build_id, test_id)` observations
   have no unresolved primary-key ambiguity;
7. the binary failure target is valid;
8. realized execution cost is available and usable;
9. sufficient chronological history exists for the frozen
   Phase 11 outer and inner partitions;
10. the model-fit partition contains both target classes.

The current Phase 11 implementation does not define a new
aggregation rule for duplicate `(build_id, test_id)`
observations. Projects containing unresolved repeated
test-build observations are therefore not eligible for the
Phase 11 execution set.

Calibration support and outer-evaluation class support are
recorded after project selection and do not determine project
eligibility.

## 2. Phase 11 project-set rule

The target execution-set size is 10 projects.

The minimum acceptable end-to-end size is 8 projects.

The previously frozen five pilot projects are retained only if
they satisfy the Phase 11 structural eligibility rules.

Additional structurally eligible projects are ordered by
`project_key` ascending and added until the target size of 10
is reached.

Project ordering does not use target-model performance,
calibration performance, selection effectiveness, retained
risk, or cost saving.

## 3. Why this change is required

The original five-project screening configuration was created
for implementation-feasibility testing.

Its project-size, failure-support, computational-tractability,
and owner-diversity rules were explicitly documented as
pilot-specific and must not automatically determine the
core-scale project set.

Phase 11 therefore needs a deterministic implementation of
the broader eligibility requirements already stated in
Protocol v1.

## 4. Previous results inspected before this amendment

Yes.

Prediction and calibration results have already been inspected
for the five previously frozen pilot projects during earlier
implementation-validation phases.

Those five projects were selected and frozen before those
downstream results were generated.

No RQ1 model or calibration performance has been generated or
inspected for the additional TCP-CI projects as part of the
Phase 11 selection process before this amendment.

The five pilot projects are retained as implementation anchors,
subject to the same Phase 11 structural eligibility checks,
and not because of their observed predictive performance.

## 5. Experiments affected

The following experiments must use this amendment:

- Phase 11 structural project screening;
- Phase 11 core-scale project-set freeze;
- Phase 11 multi-project RQ1 implementation-validation run.

Earlier five-project pilot results, Phase 9 targeted
validation, and Phase 10 one-project smoke results do not need
to be rerun because this amendment does not alter their frozen
project sets or experimental contracts.

The final walk-forward experiment requires a separate explicit
eligibility freeze because Protocol v1 states that pilot
holdout thresholds do not automatically define final
walk-forward eligibility.

## 6. Interpretation boundary

Phase 11 remains implementation/scaling evidence.

The Phase 11 chronological-holdout results are not primary
final RQ1 evidence.

No project, model, feature set, or calibration method may be
selected from Phase 11 performance results.
