# Original Phase 12 — Document + Freeze
## Final Exit Criteria Review

Date: 2026-10-19

Status: READY FOR HUMAN CLOSURE REVIEW

## 1. Scope

This review assesses completion of the original
Phase 12 Document + Freeze milestone for the
RQ1 implementation-validation stage.

It does not declare final longitudinal RQ1,
risk-bounded RQ2, or retraining RQ3 results complete.

## 2. Acceptance criteria

| Criterion | Verdict | Evidence |
| --- | --- | --- |
| RQ1 pilot and core milestone reporting | PASS | docs/pilot_report_v1.md; docs/phase11_core_scale_milestone_v1.md |
| Frozen RQ1 models and configurations | PASS | Frozen YAML configs and implementation baseline |
| CORE ten-project RQ1 validation | PASS | Phase 11 milestone §§3–8 |
| Independent output audits | PASS (reported) | Phase 11 milestone §9; preserved audit logs |
| Full reproducibility rerun | PASS (reported) | Phase 11 milestone §12; preserved rerun evidence |
| Reliability diagrams | PASS | 30 PNGs; figure index |
| Figure structural validation | PASS | 8/8 checks |
| Figure checksum verification | PASS | 31/31 checks |
| Original artifact checksum verification | PASS | 46/46 checks |
| Protocol amendment traceability | PASS | docs/protocol_amendments.md |
| Documentation reconciliation | PASS | Phase 11 milestone §§16–17 |
| Git commit and synchronization | PASS | 9e72e3e; main synchronized with origin/main |
| Independent archive backup | USER-CONFIRMED | Two archives uploaded to Google Drive |
| Formal milestone closure decision | PENDING | Human approval |

## 3. Freeze and provenance

Implementation baseline:

4b9d6da61fe3a05de67db018e972f5acaa20bc06

Reliability documentation commit:

ac38440

Milestone reconciliation commit:

9e72e3e

Production checksum manifest:

docs/phase11_core_scale_checksums.sha256

Reliability checksum manifest:

docs/phase11_core_reliability_figures.sha256

Reliability provenance:

docs/phase11_core_reliability_provenance.md

## 4. Evidence preservation

Two archives are preserved outside the Git repository:

1. phase11_core_scale.zip
2. phase11_reliability_evidence_2026-10-19.tar.gz

Both archives remain on the local Mac.

The researcher also confirms that both were uploaded
to Google Drive for independent backup.

The integrity of the copied Google Drive objects has
not been independently verified after upload.

The reliability evidence archive SHA-256 is:

b298958932aa958222da358294e0328a958a8efb913842ea023dc4b4e99a1d77

## 5. Known limitations

- The original independent Phase 11.7 audit scripts
  are not preserved as tracked Git artifacts.
- Full Phase 11.8B rerun outputs remain outside Git.
- Rerun configurations may contain machine-specific paths.
- The original 46-entry checksum manifest does not cover
  all data-stage configs, protocol files or amendments.
- Three representative reliability diagrams were manually
  reviewed; all 30 passed structural validation.
- Google Drive copies have not been independently verified
  against local archive hashes.

These limitations are disclosed and must not be presented
as evidence of final longitudinal effectiveness.

## 6. Scientific boundary

Phase 11 CORE RQ1 outputs are implementation validation
on chronological holdout data.

They do not establish:

- the best final probability model;
- the best calibration method;
- overall longitudinal improvement from calibration;
- RQ2 runtime savings or failure-recall effectiveness;
- RQ3 retraining policy effectiveness.

Final claims require the subsequent frozen
walk-forward and downstream experiments.

## 7. Exit recommendation

Technical and documentation acceptance criteria are
supported by the recorded evidence.

Recommendation:

READY FOR HUMAN CLOSURE REVIEW.

Formal closure requires researcher approval of this review,
including acceptance of the documented limitations.

This document does not automatically close Phase 12
or change the frozen research methodology.

---

## 8. Evidence Inventory

### 8.1 Preserved archives

**Phase 11 core-scale evidence**

Archive: `phase11_core_scale.zip`

SHA-256:

`45e8b993a46adc41ad0367d8eeb23c923dcf8c1b9de11724b38b7c63ae208750`

This fingerprint was computed from the ZIP provided for
independent review.

Important evidence paths inside the ZIP, relative to
`phase11_core_scale/`:

- `materialization/phase11_7a_prediction_integrity_audit.txt`
- `materialization/phase11_7b_metric_recomputation_audit.txt`
- `materialization/phase11_7c_calibration_bin_audit.txt`
- `materialization/phase11_7d_diagnostics_split_audit.txt`
- `materialization/phase11_8a_regression_repro_preflight.txt`
- `materialization/core_rq1_repro_20261018_204249/`

The rerun directory contains:

- `core_rq1_repro.yaml`
- `full_rerun_terminal.txt`
- `production_vs_rerun_audit.txt`
- `pre_rerun_state.txt`
- `rq1/rq1_metrics.csv`
- `rq1/calibration_bins.csv`
- `rq1/model_diagnostics.csv`
- 30 rerun prediction CSV files

The preserved comparison report records 30 prediction
files and 9,651,081 prediction rows compared with zero
reported maximum probability difference.

The three rerun summary CSV SHA-256 values were also
independently checked against the production fingerprints
recorded before the rerun: 3/3 matched.

The prediction-level result remains supported by the
stored comparison audit, not a newly executed comparator.

**Reliability diagram evidence**

Archive: `phase11_reliability_evidence_2026-10-19.tar.gz`

SHA-256:

`b298958932aa958222da358294e0328a958a8efb913842ea023dc4b4e99a1d77`

Contains 30 PNG diagrams, one index CSV,
the source calibration-bin CSV, and the figure manifest.

The archived figure/index files passed 31/31 SHA-256
checks. The archived source CSV matched the local
frozen source.

### 8.2 Storage and retrieval

Both archives are kept outside the Git repository on
the researcher's Mac.

The researcher confirms that both archives have also
been uploaded to Google Drive.

Retrieve archives by the exact filenames above.
Check archive SHA-256 after retrieval before use.

The downloaded Google Drive copies have not yet
been independently hash-verified.

No claim is made that the archives are included
in the Git repository.

## 9. Reproducibility Procedure

### 9.1 Historical execution environment

The Phase 11.8A preflight records:

- Implementation commit: `4b9d6da61fe3a05de67db018e972f5acaa20bc06`
- Python: 3.9.6
- Platform architecture: arm64
- NumPy: 2.0.2
- pandas: 2.3.3
- SciPy: 1.13.1
- scikit-learn: 1.6.1
- XGBoost: 2.1.4
- PyYAML: 6.0.3

The frozen execution configuration is:

`configs/core_rq1_v1.yaml`

The archived rerun configuration is:

`materialization/core_rq1_repro_20261018_204249/core_rq1_repro.yaml`

### 9.2 Required inputs and preparation

An independent rerun requires:

1. The historical implementation commit in an isolated
   checkout, without modifying the current Git branch.
2. Compatible Python dependencies.
3. TCP-CI v3 source data and the corresponding processed
   chronological project splits.
4. The frozen project, feature, model, calibration, and
   temporal-split configurations.
5. A separate writable output directory.

The original runner expects processed input under:

`data/processed/tcp_ci/core_split`

The evidence ZIP does not independently provide the
complete raw and processed input datasets.

The archived rerun YAML contains absolute paths from
the original Mac. Those paths must be adapted in a copy
of the rerun configuration for a different environment.

Do not edit the frozen production configuration or
overwrite the production results to perform this step.

### 9.3 Execution and verification boundary

The Phase 11.8A preflight confirms that the CORE RQ1
runner exposes a `--config` command-line argument.

The preserved output log documents the completed rerun,
but does not independently establish every original
shell command and setup step.

A future independent reproducer must first verify
the runner interface in the isolated checkout, then
execute with the copied configuration and isolated
output paths.

Expected implementation-validation output structure:

- 10 projects
- 30 model runs
- 90 model/calibration conditions
- 30 prediction CSV files
- 3 summary CSV files

Verify current preserved production artifacts using:

`shasum -a 256 -c docs/phase11_core_scale_checksums.sha256`

Verify current preserved reliability figures using:

`shasum -a 256 -c docs/phase11_core_reliability_figures.sha256`

Both commands require the corresponding Git-ignored
artifacts to be present at their expected relative paths.

The historical rerun reported exact agreement with
production; this documentation update does not repeat
that computational experiment.

### 9.4 Outstanding reproducibility limitations

- Original standalone Phase 11.7 audit scripts are not
  preserved as tracked Git files.
- Full rerun evidence is distributed separately from Git.
- The full original dataset preparation and portable
  end-to-end rerun have not been independently re-executed
  on a different machine during this review.
- The original comparator's prediction-level result
  is supported by stored audit logs.
- The Google Drive backups are user-confirmed, but their
  downloaded copies have not been hash-verified.

These limitations are accepted as disclosed historical
implementation-validation limitations only if approved
at the final human closure gate.

They do not alter the frozen scientific methodology and
do not establish final longitudinal RQ1-RQ3 results.
