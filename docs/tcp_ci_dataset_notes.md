# TCP-CI Dataset Reading Notes

## 1. Dataset identity

- Dataset: TCP-CI
- Version: v3
- Subjects: 25 open-source projects
- Approximate builds: 21.5k
- Approximate failed builds: 2.5k
- Local artifact: `TCP-CI-main-dataset.tar.gz`

## 2. High-level data model

Main concepts:

- CI build
- build log
- test case
- source code
- commit
- fault

High-level relationship:

`commit/change -> build -> test execution -> execution result`


Additional source-code information includes static metrics,
change history, and dependencies.

High-level purpose:

TCP-CI was designed for machine-learning-based Test Case
Prioritization in Continuous Integration.

Its feature data combines information from multiple sources, including:

- CI/build execution data
- test execution history
- source code
- source-code change history
- static/historical dependencies

## 3. Raw input versus published output

The TCP-CI repository documents two CI input files used by the dataset-generation tool:

- `<repo>-full.csv`
- `<repo>-builds.csv`

These are inputs to the original TCP-CI dataset-generation
pipeline and must not be confused with the published dataset outputs.

The downloaded `TCP-CI-main-dataset` primarily contains the
outputs produced by that pipeline, so these raw input files
must not be assumed to exist locally until verified.


### Published dataset outputs

- `dataset.csv`
- `exe.csv`
- `entity_change_history.csv`
- `id_map.csv`
- `builds.csv`
- `contributors.csv`
- `analysis/`

### `contributors.csv`

Contains contributor information for the project.

It is retained as source metadata but is not currently part of
the CORE pilot feature schema.

## 4. Important artifacts

### `dataset.csv`

Contains precomputed test-case features associated with CI builds.

Inventory findings:

- all 25 projects contain `dataset.csv`;
- all observed `dataset.csv` files share one common schema;
- the published dataset contains precomputed test-level feature
  groups related to history, complexity, change, coverage, and
  dependency information.

Still open:

- the exact row semantics of `dataset.csv` when it is used as
  a source of precomputed features for the thesis pipeline;
- the precise source-code/test-class semantics represented by
  the encoded `Test` entity beyond the verified pilot mapping;
- whether `dataset.csv` contains only failed builds;
- which precomputed features are temporally valid and suitable
  for the CORE pilot feature set.

### `exe.csv`

Contains test execution history.

Documented `test_result` values:

- `0`: pass
- `1`: exception failure
- `2`: assertion failure
- `3`: unknown failure

Documented binary interpretation:

- `0 -> failed = 0`
- `{1, 2, 3} -> failed = 1`

The downloaded v3 artifact uses the column name `verdict`
rather than `test_result`. Local verification of the observed
artifact-level mapping is recorded in Section 9.

### Execution-history completeness caveat

The set of test cases observed in `exe.csv` may be smaller than
the set of test cases present in the source code because not all
tests were necessarily executed in the available Travis CI
build logs.

This must be considered when defining the eligible test set for
each build during the pilot experiments.

### Execution duration

`exe.csv:duration` is present without missing values across
the 25 projects inspected.

The physical unit was subsequently resolved during pilot
canonicalization through upstream implementation inspection.

The published `exe.csv:duration` value is treated as
milliseconds.

No numerical conversion is required when mapping it to
`execution_time_ms`.


### `builds.csv`

Contains CI build information.

The observed published-v3 schema is:

- `id`
- `commits`
- `started_at`

Potential thesis use:

- `id` -> build identity;
- `started_at` -> temporal ordering;
- `commits` -> commit/change context.

Local inventory confirmed that every build identifier observed
in `exe.csv:build` maps to a corresponding `builds.csv:id`
within the same project.

This linkage is accepted for the downloaded TCP-CI v3 artifact.

### `entity_change_history.csv`

Contains source-code entity change history based on Git commits.

Potential use:

- F2 change/context features.

### `id_map.csv`

Maps source-code paths to entity identifiers.

Historical paths may share one ID when files are renamed or
moved.

### `analysis/`

The TCP-CI documentation describes build-level
static/dependency artifacts under `analysis/`.

However, no `analysis/` directory was observed in any of the
25 projects in the downloaded main-dataset package.

Therefore, these raw analysis artifacts are not directly
available in the currently used artifact.

Precomputed static, change, coverage, and dependency-related
features are still present in `dataset.csv` and will be
evaluated separately during pilot feature screening.

## 5. Temporal rule

For target build `b`, model inputs must be available before
test execution in build `b`.

Current-build execution outcomes must not be used as predictors
for the same build.

## 6. Thesis mapping hypothesis

### F1 — History

Potential source:

- `exe.csv`

Candidates:

- failure history;
- recent failure;
- execution-duration history;
- test age/recency.

### F2 — Change/context

Potential sources available in the downloaded artifact:

- `builds.csv`;
- `entity_change_history.csv`;
- precomputed `dataset.csv`.

The TCP-CI documentation also describes raw static/dependency
artifacts under `analysis/`, but these directories are not
present in the downloaded main-dataset package.

### F3 — Temporal

Potential sources:

- build order;
- historical execution records.

## 7. Inventory questions

### Resolved

- 25 project directories were discovered.
- All 25 projects contain `exe.csv`.
- `exe.csv` has one common schema:
  `test`, `build`, `job`, `verdict`, `duration`.
- `verdict` contains only observed values `0`, `1`, and `2`.
- No missing `duration` values were observed.
- `started_at` timestamps are parseable for all inspected builds.
- No `analysis/` directory exists in the downloaded main-dataset package.
- Every build identifier observed in `exe.csv:build` maps to a
  corresponding `builds.csv:id` within the same project.
- No duplicate `builds.csv:id` values were observed within a project.
- For all five pilot projects, `exe.csv:test` matches
  `id_map.csv:value` with rate 1.0 and does not match
  `id_map.csv:key`.
- No repeated `(build_id, test_id)` execution groups were
  observed in the five pilot projects.
- No multi-job `(build_id, test_id)` groups or conflicting
  verdict groups were observed in the five pilot projects.
- The canonical `(project_id, build_id, test_id)` key was
  verified to be non-null and unique across all five pilots.
- `exe.csv:duration` was confirmed to be represented in
  milliseconds in the inspected TCP-CI implementation.

### Still open

- Does `dataset.csv` contain only failed builds?
- Which F1/F2/F3 fields will be retained in the CORE pilot
  feature set?
- What precise source-code/test-class semantics should be
  attached to the encoded test entity beyond the verified
  `id_map.csv:value` relationship?


### Test granularity caveat

The documented CI input format identifies a test using the full
test class name (`testName`) so that it can be matched to its
corresponding source file.

Pilot verification established that `exe.csv:test` contains the
encoded entity identifier corresponding to `id_map.csv:value`.

This identifier is therefore used as the stable thesis
`test_id` for canonicalization.

However, it must not be interpreted as an individual
test-method identifier without additional evidence.


## 8. Difference from the original TCP-CI study

The original TCP-CI study focuses primarily on Test Case
Prioritization.

This thesis reuses TCP-CI as an empirical CI benchmark for:

- failure-risk estimation;
- probability calibration;
- cost-aware and risk-bounded test selection.

Therefore, the original TCP-CI learning and feature pipeline is
not adopted unchanged. Its artifacts are treated as data sources
that must be mapped into the thesis canonical schema under the
thesis temporal protocol.

## 9. Local inventory findings

### Project coverage

The downloaded TCP-CI v3 main-dataset contains 25 discovered
project directories.

After increasing Python's CSV field-size limit for a large
`commits` field, all inventoried CSV artifacts could be read
successfully.

### `exe.csv`

All 25 projects share one observed `exe.csv` schema:

- `test`
- `build`
- `job`
- `verdict`
- `duration`

Observed schema signature:

`c6061e94dcfe`

The downloaded artifact therefore differs in column naming
from the raw/input-oriented names described in parts of the
TCP-CI repository documentation.

The observed execution-outcome field is `verdict`.

Across all 25 projects, only verdict values `0`, `1`, and `2`
were observed. No unrecognized verdict values were found.

The binary thesis target mapping is:

- `verdict == 0 -> failed = 0`
- `verdict in {1, 2} -> failed = 1`

TCP-CI documentation additionally defines result code `3` as
an unknown failure. Code `3` was not observed in the local v3
artifact but will be treated as failure if encountered.

### `builds.csv`

All 25 projects share one observed `builds.csv` schema:

- `id`
- `commits`
- `started_at`

Observed schema signature:

`fcd49b345641`

The `started_at` field was parseable for all inspected build
records, giving a build-time parse rate of 1.0 for all
25 projects.

All build identifiers observed in `exe.csv:build` were found
in `builds.csv:id` within the same project, and no duplicate
build identifiers were observed.

Therefore, `exe.csv:build` and `builds.csv:id` are accepted as
the execution-to-build metadata linkage for the downloaded v3
artifact.

### Execution duration

`exe.csv:duration` is present without missing values across
the 25 projects inspected.

At the inventory stage, the physical unit of the published
`duration` field had not yet been verified. Therefore, the
inventory process preserved the raw values without applying
any numerical conversion.

The unit was subsequently resolved during pilot
canonicalization, as documented below.

### Static/dependency analysis artifacts

No `analysis/` directory was observed in any of the 25 project
directories in the downloaded main-dataset package.

This does not imply that all static, change, or dependency
features are unavailable. The common `dataset.csv` schema
contains many precomputed history, complexity, change,
coverage, and dependency-related feature fields.

### Raw-input-only fields

The fields `count`, `failures`, `errors`, and `skipped`
documented for the raw CI input format were not observed in
the published v3 `exe.csv` schema.

They are therefore marked `UNAVAILABLE` for the currently used
main-dataset artifact rather than treated as confirmed source
fields.


### Duration unit resolution

The physical unit of `exe.csv:duration` was verified against
the upstream TCP-CI implementation.

`TrTorrentProcessor` extracts the numeric duration reported by
the CI test output. During execution-record construction,
`RtpTorrentExtractor` explicitly multiplies the aggregated
duration by `1000.0`:

`int(row[ExecutionRecord.DURATION] * 1000.0)`

The resulting value is stored by `ExecutionRecord` without
further numerical conversion and is subsequently written to
`exe.csv` by `DataCollectionService`.

A repository-wide inspection of the upstream implementation
identified a single `ExecutionRecord` construction site in the
inspected code path.

For all five selected pilot projects,
`dataset.csv:Duration` also matched `exe.csv:duration` exactly
for every matched `(Build, Test)` record.

Therefore, the thesis pipeline treats `exe.csv:duration` and
`dataset.csv:Duration` as milliseconds.

Canonical mapping:

`exe.csv:duration` -> `execution_time_ms`

No numerical conversion is applied during canonicalization.


### Pilot canonical granularity verification

The five pilot projects were inspected for repeated
`(build_id, test_id)` execution records.

Across all five pilots:

- duplicate `(build_id, test_id)` groups: 0
- multi-job `(build_id, test_id)` groups: 0
- conflicting-verdict groups: 0

Therefore, no aggregation is required when transforming the
pilot execution records from the published execution-level
representation to the thesis `test_build` canonical
granularity.

The canonical key is:

`(project_id, build_id, test_id)`

This key was empirically verified to be non-null and unique
across all five pilot datasets.


### Test identifier verification

For all five pilot projects, the values of `exe.csv:test`
matched `id_map.csv:value` with a rate of `1.0`, while the
match rate against `id_map.csv:key` was `0.0`.

Therefore, the thesis adapter treats `exe.csv:test` as the
encoded entity identifier corresponding to
`id_map.csv:value`.

The identifier is not interpreted as an individual test-method
identifier without additional evidence.