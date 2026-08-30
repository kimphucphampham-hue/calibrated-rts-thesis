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

Contains test-case features across failed CI builds.

Inventory findings:

- all 25 projects contain `dataset.csv`;
- all observed `dataset.csv` files share one common schema;
- the published dataset contains precomputed test-level feature
  groups related to history, complexity, change, coverage, and
  dependency information.

Still open:

- the exact row granularity used for thesis normalization;
- the semantic role of the `Build` and `Test` identifiers;
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

In the documented raw CI input format, `duration` is expressed
in seconds.

The thesis intends to use a canonical execution-cost measure,
but the unit of `exe.csv:duration` in the downloaded published
v3 artifact has not yet been verified.

Therefore, the raw-input rule must not yet be applied directly
to the published `exe.csv` artifact, and no unit conversion is
performed at this stage.


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

### Still open

- What is the exact semantic identity represented by `exe.csv:test`?
- Is `test` stable across builds and how does it relate to `id_map.csv`?
- Can the same test appear more than once in a build across multiple jobs?
- What is the confirmed unit of `exe.csv:duration`?
- Does `dataset.csv` contain only failed builds?
- Which F1/F2/F3 fields will be retained in the CORE pilot feature set?


### Test granularity caveat

The documented CI input format identifies a test using the full
test class name (`testName`) so that it can be matched to its
corresponding source file.

Therefore, `test_id` must not yet be assumed to represent an
individual test method. The exact semantic identity and
granularity of `exe.csv:test` will be verified during pilot
normalization and test-identity analysis.


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

However, the unit of `duration` in the published v3 artifact
has not yet been frozen.

Therefore, no conversion to milliseconds is performed at the
inventory stage.

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
