# TCP-CI Pilot Project Selection

## Purpose

This document records the selection protocol for the initial
five-project TCP-CI pilot.

The pilot is intended to validate the end-to-end thesis
pipeline before the full multi-project experiment.

Pilot-project selection is based on dataset feasibility and
must not depend on downstream model performance.

## Eligibility criteria

A project is eligible for the initial pilot when it satisfies:

- at least 200 unique CI builds;
- at least 40 unique tests;
- at least 100 observed failing execution records;
- execution failure rate of at least 0.1%;
- at most 1,000,000 execution records for pilot tractability;
- no unrecognized execution-outcome values;
- no missing execution-duration records;
- fully parseable build timestamps;
- verified execution-to-build linkage.

The execution-row upper bound is a pilot-only computational
constraint and is not automatically reused in the full thesis
experiment.

## Diversity rule

At most one project from the same repository owner is selected
for the initial five-project pilot.

This rule is intended to exercise the pipeline across more
heterogeneous projects.

## Ranking

Eligible projects are ranked by:

1. number of observed failure executions, descending;
2. number of unique builds, descending;
3. project key, ascending.

Projects are selected in ranking order while applying the
owner-diversity rule.

## Leakage and selection policy

Pilot projects are selected before model training.

The selection process does not use:

- AUC;
- Brier score;
- calibration error;
- failure-detection effectiveness;
- retained predicted risk;
- execution-cost savings.

These downstream experimental results must therefore not
influence pilot-project inclusion.

## Selected projects

The screening process evaluated 25 inventoried TCP-CI projects.
Eighteen projects satisfied the pilot eligibility criteria.

After applying the predefined ranking and repository-owner
diversity rule, the following five projects were selected:

| Rank | Project | Owner | Builds | Tests | Failure rows | Failure rate | Execution rows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `datasets/camunda@camunda-bpm-platform` | camunda | 822 | 1021 | 2112 | 0.004467 | 472765 |
| 2 | `datasets/apache@shardingsphere` | apache | 1049 | 1020 | 1364 | 0.001636 | 833541 |
| 3 | `datasets/facebook@buck` | facebook | 846 | 800 | 1128 | 0.002010 | 561294 |
| 4 | `datasets/Angel-ML@angel` | Angel-ML | 308 | 46 | 442 | 0.043620 | 10133 |
| 5 | `datasets/zolyfarkas@spf4j` | zolyfarkas | 587 | 168 | 398 | 0.005786 | 68787 |

The final pilot-project list is frozen in:

`configs/pilot_projects_v1.yaml`

The generated screening results are stored under:

`results/pilot/screening/`