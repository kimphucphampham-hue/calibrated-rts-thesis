# TCP-CI Temporal Train/Test Split v1

## Purpose

This document defines the chronological outer train/test
holdout used for the initial five-project TCP-CI pilot.

The split is performed at CI-build granularity rather than at
individual test-row granularity.

## Split unit

The atomic split unit is a build.

All test rows belonging to the same build must be assigned to
the same split.

No build may appear in both training and testing data.

## Ordering

Observed builds are ordered by the canonical `build_index`,
which was derived from chronological build metadata.

The split is deterministic and does not depend on input row
order.

## Boundary

For each pilot project:

- the first 80% of observed builds are assigned to `train`;
- the remaining 20% are assigned to `test`;
- the number of training builds is computed using floor
  rounding.

The boundary is determined without inspecting failure labels,
execution costs, model performance, or downstream evaluation
metrics.

## Temporal constraint

The following invariant must hold:

`max(train.build_index) < min(test.build_index)`

Therefore, every training build precedes every test build.

## No row-level randomization

Random row-level train/test splitting is prohibited.

Label-based stratification is also prohibited because it can
destroy the chronological CI structure and may allow tests
from the same build or future periods to influence training.

## History features during testing

History features remain causal at every target build.

For test build `b`, a feature may use observations from builds
strictly before `b`.

This means that later test-period builds may use execution
history produced by earlier test-period builds because those
observations are already available at that later decision
time.

This does not permit model fitting on the test period.

Feature-state updating and model retraining are separate
operations.

## Holdout policy

The outer test period must not be used for:

- model fitting;
- calibration fitting;
- hyperparameter tuning;
- feature selection.

If a probability-calibration subset is required, it must be
constructed from the training period only.

## Pilot scope

This 80/20 chronological holdout is intended to validate the
initial end-to-end pilot pipeline.

The final thesis evaluation may additionally use expanding or
walk-forward temporal evaluation and explicitly defined
retraining policies.

## Outputs

Generated split datasets are stored under:

`data/processed/tcp_ci/pilot_split/`

Each pilot project contains:

- `train.csv`
- `test.csv`
- `build_split.csv`

Diagnostics are stored under:

`results/pilot/temporal_split/`