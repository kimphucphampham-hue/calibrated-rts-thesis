# TCP-CI Per-Project Pilot Outputs v1

## Purpose

This document defines the compact per-project output package
for the five frozen TCP-CI pilot projects.

The package consolidates project-level summaries and
diagnostics while preserving the existing generated artifacts
as their single source of truth.

## Output structure

Generated packages are stored under:

`results/pilot/projects/`

Each frozen pilot project receives one directory containing:

- `project_summary.json`
- `model_metrics.csv`
- `calibration_bins.csv`
- `artifact_manifest.csv`

A cross-project index is stored as:

`results/pilot/projects/project_output_index.csv`

## Project summary

`project_summary.json` records compact project-level
information from the completed pilot pipeline, including:

- history row/build/test counts;
- cold-start and history coverage;
- chronological train/test counts;
- train/test failure rates;
- temporal split boundary;
- dummy-prior metrics;
- logistic-regression metrics;
- calibration-oriented diagnostics.

The summary is derived from existing generated artifacts and
does not trigger model training or metric recomputation.

## Model metrics

`model_metrics.csv` contains exactly two rows for each pilot:

- `dummy_prior`
- `logistic_regression`

The values are filtered directly from the master
`pilot_metrics.csv` artifact.

Therefore the project-level metric files are views of the
master results rather than independently recomputed
experiments.

## Calibration bins

`calibration_bins.csv` contains the non-empty calibration bins
for the corresponding project and both pilot models.

The rows are filtered directly from the master
`calibration_bins.csv` artifact.

## Artifact manifest

Large generated artifacts are not duplicated into each
project package.

Instead, `artifact_manifest.csv` references the existing:

- history-feature dataset;
- training split;
- test split;
- build assignment table;
- test prediction file.

Each manifest row records the repository-relative artifact
path and its file size.

This design avoids redundant copies of large generated data
while retaining an auditable link from each project package
to its underlying artifacts.

## Cross-project index

`project_output_index.csv` contains one row per frozen pilot
project.

It provides a compact view of:

- row/build/test counts;
- outer-test size and failure prevalence;
- logistic-regression ROC-AUC;
- Average Precision;
- Brier score;
- Log Loss;
- ECE;
- absolute calibration bias.

Per-project results remain the primary pilot evidence. The
index is intended for navigation and audit rather than as a
new statistical aggregation.

## No additional experiment

The packaging stage performs no:

- feature construction;
- train/test splitting;
- preprocessing fitting;
- model fitting;
- probability calibration;
- metric recomputation;
- threshold optimization;
- test-selection optimization.

It only organizes existing validated pilot artifacts.

## Duplication policy

Large history, split, and prediction files remain at their
canonical generated locations.

The project output package contains compact summaries and
references only.

This avoids storing multiple physical copies of the same
experimental data.

## Validation

The packaging stage verifies that:

- exactly five project packages are created;
- every project contains exactly two model metric rows;
- packaged metrics reproduce the master metric table;
- packaged calibration bins reproduce the master bin artifact;
- every manifest target exists;
- all five projects appear in the cross-project index.

## Scope

These packages provide an auditable project-level view of the
five-project pilot pipeline.

They do not introduce additional experimental observations
and should not be interpreted as a new evaluation protocol.