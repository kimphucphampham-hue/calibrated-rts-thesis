# RQ1 Targeted Feature-Ablation Validation v1

## 1. Role and scope

This milestone records the supplementary five-project
targeted feature-ablation validation for RQ1.

It is not primary final thesis evidence.

The primary CORE RQ1 feature set remains:

`F1_F2_F3`

No feature set, model, or calibration method is selected as a
winner from this targeted validation.

The primary longitudinal RQ1 claims remain reserved for the
future walk-forward experiment.

## 2. Frozen comparison

The targeted validation compares the three previously frozen
CORE feature sets:

1. `F1`
2. `F1_F2`
3. `F1_F2_F3`

The comparison is a sensitivity / ablation analysis only.

The primary CORE feature set is not changed by these results.

## 3. Models and calibration variants

Frozen models:

- Logistic Regression
- Random Forest
- XGBoost

Frozen probability variants:

- uncalibrated
- Platt
- isotonic

No result-driven hyperparameter tuning, feature selection, or
calibration-method selection is performed.

## 4. Temporal design

This targeted validation reuses the frozen five-project RQ1
pilot temporal design.

For every project and model run:

`model-fit period < calibration period < outer evaluation period`

The outer evaluation period is not used for:

- model fitting;
- preprocessing fitting;
- calibration fitting;
- feature selection;
- hyperparameter selection.

The targeted validation does not enable retraining.

## 5. Project set

The five frozen pilot projects are:

- `datasets/camunda@camunda-bpm-platform`
- `datasets/apache@shardingsphere`
- `datasets/facebook@buck`
- `datasets/Angel-ML@angel`
- `datasets/zolyfarkas@spf4j`

Project retention is not based on model performance.

## 6. Experimental grid

The complete targeted-validation grid is:

- projects: 5
- feature sets: 3
- models: 3
- calibration variants: 3

Therefore:

`5 x 3 x 3 x 3 = 135 conditions`

The run produced:

- 45 model runs;
- 135 metric conditions;
- 135 available conditions;
- 135 reliability-condition audits;
- 135 predefined feature-set contrasts.

## 7. Frozen feature contrasts

The contrast definition is:

`candidate - baseline`

The predefined feature contrasts are:

1. `F1_F2 - F1`
2. `F1_F2_F3 - F1_F2`
3. `F1_F2_F3 - F1`

These contrasts were frozen before result interpretation.

They do not imply winner selection.

## 8. Validation results

Phase 9.6 structural and result audit status:

`PASS`

The following checks passed:

- complete experimental grid;
- evaluation-target invariance;
- numerical metric validity;
- temporal/calibration invariance;
- reliability-bin reconstruction;
- predefined contrast generation.

No project was removed based on performance.

No pooled-test-row result is used as the primary comparison.

## 9. Reproducibility provenance

Targeted-validation design freeze commit:

`89c36d5`

Targeted-validation runner commit:

`1372d87785727d5d7320d9d072f27f204cfe29f3`

The Phase 9.5 and Phase 9.6 experimental artifacts were
generated from commit:

`1372d87785727d5d7320d9d072f27f204cfe29f3`

Phase 9.6 input SHA256 values:

- audit policy:
  `3cb35c0f19614e4b1c22392dc162bb52e7c26f54bd425965a380acf4b4b4d81a`
- Phase 9.5 source manifest:
  `2f6821a165b50ff61656fd2f7a63bd04af4efe1e038d2d9d31229bdc59be9f80`
- metrics summary:
  `298eb27c31354d90b02dd39b90ef766b04081b97d3fe05d6c3cbcdfbf610e0d8`
- reliability bins:
  `383b6c58b32c510196d396b727ca7a3df2d575647668186eb9d10b12802ebf1a`
- diagnostics summary:
  `9131a241b30cbe005f5c685939974a5507f87231c52c90913ee1165b4b63a028`

Phase 9.6 output SHA256 values:

- reliability audit:
  `26b0e12f413a30738f82e93600cb6d49f9374e78fb5a5c6b60139a542c1a4645`
- feature contrasts:
  `dfd1be0cc272528405c771ff418e6aa1c5d4c549f4a3b73f293eba4b5c4294e6`
- project condition counts:
  `d6ada31e4487d3f220ee67256654df04adb2c59284c967f0839cc440c1ca484c`

## 10. Protocol relationship

This targeted validation does not redefine the primary CORE
feature set or the temporal/calibration protocol.

The three feature sets were frozen before this experiment for
later ablation analysis.

Therefore this milestone is recorded as a supplementary
sensitivity analysis rather than a replacement of the primary
RQ1 experimental protocol.

Any future change to project eligibility, feature definitions,
feature temporal availability, temporal splitting,
calibration rules, primary metrics, or statistical analysis
must follow the protocol amendment rules.

## 11. Interpretation boundary

These five-project chronological-holdout results must not be
presented as the final longitudinal answer to RQ1.

In particular:

- the final CORE walk-forward experiment has not yet been run;
- project-level failure prevalence differs substantially;
- rare-event projects require cautious interpretation;
- calibration may improve or degrade probability quality
  depending on project/model/window support.

Phase 9 establishes supplementary implementation and
sensitivity evidence only.
