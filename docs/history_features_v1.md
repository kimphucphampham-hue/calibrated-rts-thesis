# TCP-CI History Features v1

## Purpose

This document defines the minimal history-based feature set
used for the initial TCP-CI pilot pipeline.

All history features are constructed at the canonical
`test_build` granularity.

## Temporal rule

For a target build `b`, a history feature may use only
executions of the same test from builds strictly before `b`.

Current-build outcome and execution duration must not
contribute to features used for that same build.

Formally:

`history(test, b) = executions(test, build_index < b)`

## Feature set

### `failure_rate_history`

Fraction of prior observed executions of the same test that
failed.

`prior_failure_count / prior_execution_count`

The value is missing when no prior execution exists.

### `recent_failure`

Binary outcome of the most recent observed execution of the
same test before the target build.

The value is missing for the first observed execution.

### `consecutive_failures`

Number of consecutive failures immediately preceding the
target execution.

The value is zero when the previous execution passed or when
no prior execution exists.

### `consecutive_passes`

Number of consecutive passes immediately preceding the target
execution.

The value is zero when the previous execution failed or when
no prior execution exists.

### `test_age_builds`

Difference between the target `build_index` and the first
observed `build_index` of the same test.

The first observed execution therefore has age zero.

### `time_since_last_failure_builds`

Difference between the target `build_index` and the most
recent prior build in which the same test failed.

The value is missing when the test has never failed before the
target build.

### `execution_time_mean_history_ms`

Arithmetic mean of `execution_time_ms` over all prior observed
executions of the same test.

The current execution duration is never included.

The value is missing when no prior execution exists.

## Audit columns

The generated feature dataset also retains:

- `history_available`
- `prior_execution_count`
- `prior_failure_count`

These columns are primarily used to verify temporal feature
construction and cold-start behavior.

They are not automatically part of the model feature set.

## Cold-start policy

For the first observed execution of a test:

- `history_available = 0`
- `prior_execution_count = 0`
- `prior_failure_count = 0`
- `failure_rate_history = missing`
- `recent_failure = missing`
- `consecutive_failures = 0`
- `consecutive_passes = 0`
- `test_age_builds = 0`
- `time_since_last_failure_builds = missing`
- `execution_time_mean_history_ms = missing`

Missing-value imputation is intentionally deferred to the
modeling protocol.

## Leakage boundary

`failed` and `execution_time_ms` in the current canonical row
are post-execution observations.

They may be retained as target and realized-cost fields for
evaluation, but must not be supplied directly as pre-execution
model inputs for the same build.

Only their historical values from builds strictly before the
target build may contribute to history features.

## Outputs

History-enriched pilot files are generated under:

`data/processed/tcp_ci/pilot_history/`

Generated diagnostics are stored under:

`results/pilot/history_features/`