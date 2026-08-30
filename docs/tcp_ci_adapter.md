# TCP-CI Canonical Adapter

## Purpose

The TCP-CI adapter transforms the published TCP-CI v3
execution artifacts into the canonical schema used by the
thesis pipeline.

The adapter is implemented in two stages:

1. evidence-preserving staging;
2. validated canonical materialization.

## Inputs

The adapter uses:

- `exe.csv`
- `builds.csv`
- `id_map.csv`

for the five frozen pilot projects defined in:

`configs/pilot_projects_v1.yaml`

## Staging granularity

The staging representation preserves the published
test-job-execution records.

Fields include:

- `project_id`
- `build_id`
- `build_started_at`
- `build_index`
- `job_id`
- `test_id`
- `commit_hashes`
- `verdict_raw`
- `failed`
- `duration_raw`

## Verdict mapping

TCP-CI verdicts are mapped to the binary thesis target as:

- `0` -> pass (`failed = 0`)
- `1` -> fail (`failed = 1`)
- `2` -> fail (`failed = 1`)
- `3` -> fail (`failed = 1`)

Only verdict values `0`, `1`, and `2` were observed in the
local TCP-CI v3 artifact used for the pilot.

## Canonical granularity

The thesis canonical granularity is `test_build`.

Canonical key:

`(project_id, build_id, test_id)`

Across the five pilot projects, no duplicate
`(build_id, test_id)` groups, multi-job groups, or conflicting
verdict groups were observed.

Therefore, no row aggregation is required for the pilot
canonicalization.

## Duration

Upstream implementation inspection confirmed that the
published execution duration is converted to milliseconds
before an `ExecutionRecord` is created.

The canonical mapping is therefore:

`exe.csv:duration` -> `execution_time_ms`

The mapping is an identity transformation.

The current-build execution time is a post-execution observed
cost and must not be used directly as a pre-execution feature
for selecting tests in that same build.

## Temporal ordering

`builds.csv:started_at` is parsed as the build timestamp.

`build_index` is assigned in chronological order and was
verified to preserve monotonically increasing build time for
all five pilot projects.

## Canonical output

Canonical files are generated under:

`data/processed/tcp_ci/pilot/`

Each project contains:

`test_build.csv`

Core fields are:

- `project_id`
- `build_id`
- `build_started_at`
- `build_index`
- `test_id`
- `execution_time_ms`
- `failed`

Additional provenance metadata is retained:

- `job_id`
- `commit_hashes`
- `verdict_raw`

## Validation

For all five pilot projects:

- staging and canonical row counts are identical;
- canonical keys are non-null and unique;
- failure counts are preserved;
- execution duration has no missing values;
- execution duration has no negative values;
- build ordering is chronological.

Generated diagnostic summaries are stored under:

`results/pilot/adapter/`