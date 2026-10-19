# Phase 11 CORE RQ1 — Reliability Diagram Provenance

Date: 2026-10-19

## 1. Scientific scope

These diagrams belong to Phase 11 CORE RQ1 implementation
validation on a chronological holdout.

They are NOT final longitudinal walk-forward RQ1 evidence,
and must not be used to select a final best model or
calibration method.

## 2. Inputs and implementation

Source:
- results/core/rq1/calibration_bins.csv

Configuration:
- configs/calibration_v1.yaml

Plot implementation:
- src/tcp_ci/rq1_plots.py
- generate_reliability_figures()

Implementation execution baseline:
- 4b9d6da61fe3a05de67db018e972f5acaa20bc06

Documentation baseline before figure generation:
- 255d2c85900172e2014b48509c7747ea274285b6

## 3. Plot configuration

- 10 equal-width probability bins over [0, 1]
- X-axis: mean predicted probability
- Y-axis: observed failure frequency
- Reference line: y=x
- Empty bins omitted
- Variants: uncalibrated, Platt, isotonic
- Output: PNG, 300 DPI

## 4. Generated evidence

- 10 projects
- 3 models per project
- 30 PNG reliability diagrams
- 1 reliability_figures_index.csv

Figure directory:
results/core/rq1/reliability_diagrams/

Figure checksum manifest:
docs/phase11_core_reliability_figures.sha256

Verification:
- 31/31 figure/index SHA-256 checks passed
- 46/46 original Phase 11 checksums passed
- 8/8 structural validation checks passed
- 18 RQ1 plotting unit tests passed
- 3 representative diagrams visually reviewed

Full independent visual review of all 30 diagrams
has not been performed.

## 5. Archive

Archive filename:
phase11_reliability_evidence_2026-10-19.tar.gz

Local archive location:
../thesis-evidence-archive/

Archive SHA-256:
b298958932aa958222da358294e0328a958a8efb913842ea023dc4b4e99a1d77

Archive contents:
- 30 PNG files
- 1 index CSV
- 1 source calibration_bins.csv
- 1 figure checksum manifest

Verification:
- 31/31 archived figure/index files verified
- Archived calibration_bins.csv matches the local source

The archive is stored outside Git and must be preserved
and backed up separately. The checksum manifest alone
does not contain the image files.

## 6. Interpretation limitations

- Some probability bins contain very few observations.
- Lines between plotted points do not imply observations
  exist in omitted empty bins.
- Graylog has zero observed failures in its evaluation
  partition, limiting calibration interpretation.
- All diagrams describe Phase 11 implementation validation,
  not the final longitudinal evaluation.

## 7. Freeze boundary

No models were retrained for these figures.

The frozen source CSV, model outputs, configurations,
original protocol and original 46-entry checksum manifest
were not modified.

This document records post-execution visualization
evidence; it does not amend the frozen methodology.
