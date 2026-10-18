# RQ1 One-Project End-to-End Smoke v1

## Role

Phase 10 validates that the frozen RQ1 pipeline executes
end-to-end on one real TCP-CI project.

This smoke run is implementation evidence only and is not
primary final RQ1 evidence.

## Project

`datasets/Angel-ML@angel`

## Feature set

Primary frozen CORE feature set:

`F1_F2_F3`

## Models

- Logistic Regression
- Random Forest
- XGBoost

## Probability variants

- uncalibrated
- Platt
- isotonic

## End-to-end outputs

The smoke run successfully generated:

- per-model evaluation predictions;
- RQ1 metric summary;
- reliability bins;
- temporal/calibration diagnostics;
- reproducibility manifest.

## Structural result

- development rows: 8225
- evaluation rows: 1908
- evaluation failure rows: 185
- model runs: 3
- metric conditions: 9
- available conditions: 9
- reliability groups: 9
- prediction files: 3
- prediction rows per model: 1908

Temporal structure:

- model-fit builds: 189
- calibration builds: 57
- evaluation builds: 62
- calibration expansion: 7 builds
- calibration positive rows: 5

All calibration variants were available.

## Provenance

Experiment Git commit:

`8ff4d90ffbb4c22116de5a6db74abc377e7a0fc0`

The external Phase 10 manifest records SHA256 hashes for:

- development data;
- evaluation data;
- frozen RQ1 configuration;
- model configuration;
- calibration configuration;
- CORE feature configuration;
- output contract;
- three prediction files;
- metrics summary;
- reliability bins;
- diagnostics summary.

## Interpretation boundary

No model or calibration method is selected from this smoke run.

The smoke run does not establish the final longitudinal
effectiveness of probability calibration.

Primary RQ1 evidence still requires the frozen longitudinal
evaluation protocol defined for the thesis.
