# Phase 11 Core Project Set v1

## Role

This document freezes the exact TCP-CI project set used for
Phase 11 multi-project implementation validation.

Phase 11 results are not primary final longitudinal RQ1
evidence.

## Screening result

All 25 discovered TCP-CI v3 projects passed the frozen Phase 11
structural eligibility checks.

The screening process did not generate model predictions and
did not inspect model or calibration performance.

The Phase 11 target execution-set size is 10 projects.

## Selection rule

The selection rule was frozen before structural screening:

1. retain the five previously frozen pilot anchors when they
   remain structurally eligible;
2. sort all other structurally eligible projects by
   `project_key` ascending;
3. add projects in that order until the target count of 10 is
   reached.

Projects that are structurally eligible but fall after the
target-count stopping point remain eligible. They are not
classified as exclusions.

## Frozen pilot anchors

1. `datasets/camunda@camunda-bpm-platform`
2. `datasets/apache@shardingsphere`
3. `datasets/facebook@buck`
4. `datasets/Angel-ML@angel`
5. `datasets/zolyfarkas@spf4j`

## Deterministically selected additional projects

1. `datasets/CompEvol@beast2`
2. `datasets/EMResearch@EvoMaster`
3. `datasets/Graylog2@graylog2-server`
4. `datasets/JMRI@JMRI`
5. `datasets/SonarSource@sonarqube`

## Frozen Phase 11 execution set

The exact 10-project Phase 11 set is therefore:

- `datasets/camunda@camunda-bpm-platform`
- `datasets/apache@shardingsphere`
- `datasets/facebook@buck`
- `datasets/Angel-ML@angel`
- `datasets/zolyfarkas@spf4j`
- `datasets/CompEvol@beast2`
- `datasets/EMResearch@EvoMaster`
- `datasets/Graylog2@graylog2-server`
- `datasets/JMRI@JMRI`
- `datasets/SonarSource@sonarqube`

## Interpretation boundary

The project set was not chosen using ROC-AUC, Average
Precision, Brier Score, Log Loss, ECE, calibration quality,
risk-retention performance, or cost saving.

The remaining 15 structurally eligible TCP-CI projects are not
considered ineligible. They are outside the Phase 11 execution
set solely because the pre-frozen deterministic rule stops when
the target count of 10 is reached.

Final walk-forward project eligibility is frozen separately
before the primary longitudinal experiment.
