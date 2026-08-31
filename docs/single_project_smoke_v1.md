# TCP-CI Single-Project Smoke Run v1

## Purpose

This smoke run verifies that the pilot pipeline can be
executed independently for a single frozen TCP-CI project and
reproduce the corresponding result from the five-project run.

The smoke run is a reproducibility and integration check. It
is not a separate experiment and does not contribute an
additional project-level observation to the thesis results.

## Smoke project

The selected smoke project is:

`datasets/Angel-ML@angel`

The project is used because its pilot artifact is relatively
small and therefore provides a fast end-to-end integration
check.

Its selection for the smoke test does not imply that it is
representative of the other TCP-CI projects.

## Pipeline

The smoke run executes the following existing stages:

1. history-feature construction;
2. chronological build-level train/test split;
3. dummy-prior and logistic-regression baselines;
4. minimal probabilistic metric evaluation.

No algorithm, feature definition, model hyperparameter,
temporal boundary rule, or metric definition is changed for
the smoke run.

## Isolation

Temporary smoke-run configurations and generated artifacts are
written outside the repository under:

`/tmp/calibrated-rts-step11/`

The primary five-project artifacts are therefore not
overwritten.

## Reproducibility checks

The smoke run verifies that the single-project execution
reproduces the Angel subset of the main five-project run.

The checks include:

- identical chronological build assignments;
- identical number of outer-test observations;
- identical observed targets;
- equivalent dummy-prior probabilities;
- equivalent logistic-regression probabilities;
- equivalent project-level evaluation metrics.

Floating-point probability and metric comparisons use strict
numerical tolerances rather than formatted decimal strings.

## Observed smoke result

The Angel outer test period contains:

- 1,908 test observations;
- 185 failed observations.

The logistic-regression baseline reproduces the previously
recorded pilot performance, including approximately:

- ROC-AUC: 0.976292;
- Average Precision: 0.955044;
- Brier Score: 0.023399;
- Log Loss: 0.103535;
- ECE: 0.038954;
- calibration bias: -0.037610;
- absolute calibration bias: 0.037610.

The single-project results match the corresponding Angel
results from the five-project pilot run within the configured
numerical tolerance.

## Interpretation

Passing this smoke run demonstrates that the implemented
pipeline does not require the presence of the other four pilot
projects in order to reproduce the Angel result.

It does not establish cross-project generalization, model
optimality, or probability calibration quality.

## Scope

This smoke run is an integration/reproducibility checkpoint
for the pilot pipeline.

It is not an additional statistical experiment and should not
be counted separately from the five frozen pilot projects.