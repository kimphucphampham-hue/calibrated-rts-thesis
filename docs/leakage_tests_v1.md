# Temporal Leakage Guards v1

## Purpose

This document records the automated leakage guards applied to
the TCP-CI history-feature pipeline.

The goal is to ensure that features used for a target
test-build observation are constructed only from information
available before execution of that test in the target build.

## Leakage boundary

For test `t` in target build `b`:

`history(t, b) = observations of t with build_index < b`

The current row contains post-execution observations such as:

- `failed`
- `execution_time_ms`

These fields may be retained for target/evaluation purposes,
but they must not contribute to pre-execution history features
for the same row.

## Automated guards

### Current-row perturbation invariance

The current row's `failed` and `execution_time_ms` values are
perturbed after a baseline feature set is constructed.

All history features for that same target row must remain
unchanged.

This guards against direct current-outcome and current-cost
leakage.

### Future perturbation invariance

Post-execution values in a future build are modified.

Features already constructed for earlier builds must remain
unchanged.

This guards against future-information leakage.

### Prefix invariance

History features computed using a temporal prefix are compared
with the corresponding rows produced when later builds are
also present.

The prefix feature values must be identical.

This is a direct causal consistency check: adding future data
must not alter features for the past.

### Project isolation

Projects may contain overlapping encoded test identifiers.

History is therefore grouped by:

`(project_id, test_id)`

A test identifier from one project must never contribute to
the history of the same identifier in another project.

### Input-order invariance

The input canonical rows are shuffled before feature
construction.

After temporal sorting, the resulting history features must
match those produced from the original input order.

This guards against accidental dependence on file row order.

### Feature allowlist guard

The configured model-history feature list must not contain
current post-execution fields such as:

- `failed`
- `execution_time_ms`
- `verdict_raw`
- `duration_raw`

### Schema availability guard

Every configured history feature must be marked:

- `available_at = pre_test_execution`
- `leakage_risk = SAFE`

in the canonical TCP-CI schema.

## Scope

These tests guard the history-feature construction stage.

They do not yet prove that later model training, probability
calibration, hyperparameter tuning, or evaluation splits are
free of temporal leakage.

Separate temporal-split and calibration leakage guards must be
applied when those pipeline stages are implemented.