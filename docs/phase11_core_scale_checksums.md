# Phase 11 Core-Scale Checksum Manifest — Provenance Note

## 1. Manifest

`docs/phase11_core_scale_checksums.sha256`

This note describes the manifest. The manifest itself is not
modified, so its own hash and meaning stay stable.

Manifest SHA-256 (recorded 2026-10-19):

`9d36d62a8a48e2cee0ad517aa8d8a9eb9d664910079764af520b885622900c9f`

## 2. Provenance

- Milestone: Phase 11 multi-project core-scale implementation
  validation (`docs/phase11_core_scale_milestone_v1.md`).
- Execution baseline: `4b9d6da61fe3a05de67db018e972f5acaa20bc06`.
- Run: Phase 11 production CORE RQ1 run
  (`execution_id: PHASE11-CORE-RQ1-v1`), not the 11.8B
  isolated rerun.
- Manifest file written: 2026-10-18 21:31 (local time), after
  the documentation commit `98eba9e`.
- The manifest does not itself record a commit or run ID;
  this note supplies them.

## 3. Format

`shasum -a 256` check format, one entry per line, paths
relative to the repository root.

Verify from the repository root:

```bash
shasum -a 256 -c docs/phase11_core_scale_checksums.sha256
```

Expected result: 46 lines ending in `OK`.

## 4. Coverage

| Group | Entries | Tracked in Git? |
| --- | ---: | --- |
| Configs (`configs/`) | 7 | yes |
| Source (`src/tcp_ci/`) | 6 | yes |
| Summary CSVs (`results/core/rq1/`) | 3 | no (gitignored) |
| Prediction files (`results/core/rq1/predictions/`) | 30 | no (gitignored) |
| **Total** | **46** | |

Configs:

- `configs/core_rq1_v1.yaml`
- `configs/core_projects_v1.yaml`
- `configs/core_temporal_split_v1.yaml`
- `configs/rq1_v1.yaml`
- `configs/rq1_models_v1.yaml`
- `configs/calibration_v1.yaml`
- `configs/core_features_v1.yaml`

Source:

- `src/tcp_ci/core_rq1_runner.py`
- `src/tcp_ci/rq1_pipeline.py`
- `src/tcp_ci/rq1_evaluator.py`
- `src/tcp_ci/calibration_split.py`
- `src/tcp_ci/model_runner.py`
- `src/tcp_ci/probability_calibration.py`

Results:

- `results/core/rq1/rq1_metrics.csv`
- `results/core/rq1/calibration_bins.csv`
- `results/core/rq1/model_diagnostics.csv`
- 3 model files (`logistic_regression.csv`, `random_forest.csv`,
  `xgboost.csv`) for each of the 10 project directories under
  `results/core/rq1/predictions/`.

All 13 tracked entries hash-match their blobs at `4b9d6da`.

`results/` is gitignored, so for the 33 result entries this
manifest is the only freeze record. The result files must be
kept outside Git (or archived) for the manifest to remain
checkable.

## 5. Exclusions

Not covered by this manifest:

- data-stage configs: `configs/core_adapter_v1.yaml`,
  `configs/core_history_features_v1.yaml`,
  `configs/core_project_screening_v1.yaml`;
- intermediate data under `data/interim/tcp_ci/core` and
  `data/processed/tcp_ci/core` (gitignored);
- the protocol (`docs/protocol_v1.md`) and amendments
  (`docs/protocol_amendments.md`);
- tests (`tests/`);
- the 11.8B rerun outputs.

The tracked exclusions are still frozen by Git history at
`4b9d6da`. Extending coverage would require a new manifest
version, which is out of scope for Phase 12.F1.
