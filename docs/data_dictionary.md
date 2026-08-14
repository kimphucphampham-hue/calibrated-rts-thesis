# Data Dictionary

## Purpose

This data dictionary defines the semantics, provenance,
temporal availability, and usage status of fields used in the
calibrated risk-bounded regression test selection experiments.

The dictionary covers both:

- the legacy synthetic dataset used for reproduction and
  sanity checking;
- the TCP-CI datasets used as the primary empirical evidence
  for the master's thesis.

The machine-readable dictionary is stored at:

`docs/data_dictionary.csv`

## Dictionary schema

Each row in `data_dictionary.csv` describes one field.

| Column | Description |
|---|---|
| `dataset` | Dataset to which the field belongs |
| `source_field` | Original field name in the source dataset |
| `canonical_field` | Standardized field name used by the thesis pipeline |
| `granularity` | Observation level, e.g. project, build, test_build, test_case |
| `description` | Semantic meaning of the field |
| `data_type` | Logical data type |
| `unit` | Measurement unit, if applicable |
| `role` | Role in the pipeline: identifier, context, feature, cost, target, timestamp, or metadata |
| `source` | Technical origin of the value |
| `derivation` | Rule or formula used to derive the value |
| `available_at` | Time at which the value is available for a build decision |
| `history_scope` | Temporal history used to compute the field |
| `allowed_values` | Valid values or expected range |
| `missing_policy` | Policy for missing values |
| `leakage_risk` | Temporal/data-leakage assessment |
| `required` | Whether the field is required by the canonical schema |
| `status` | Current field status |

## Controlled values

### Role

- `identifier`
- `context`
- `feature`
- `cost`
- `target`
- `timestamp`
- `metadata`

### Availability

- `pre_build`: available before the current build starts
- `pre_test_execution`: available before test execution and therefore potentially usable at the selection decision point
- `post_test_execution`: available only after the current test has executed

### History scope

`history_scope` specifies the temporal range used to obtain or
derive a field.

For a target build `b`, historical predictors must use only
information from builds strictly earlier than `b`.

Examples:

- `same test, builds < b`
- `most recent execution before b`
- `current build context`
- `current test outcome`
- `N/A`

A field derived from the outcome of build `b` must not be used
as a predictor for the selection decision of the same build.


### Leakage risk

- `SAFE`: the field can be available at the selection decision
  point when computed according to its documented derivation.

- `CONDITIONAL`: the field may be safe or may introduce
  look-ahead leakage depending on its source or derivation.

- `LEAKAGE`: the value is available only after the current test
  execution and must not be used as a predictor or decision
  input for the same build.

Fields marked `LEAKAGE` may still be valid targets, evaluation outcomes, or historical inputs for later builds.

### Status

The `status` field records the current evidence and
governance state of each field.

- `CONFIRMED`: the field or derivation has been accepted into
  the current schema. A confirmed field may be an identifier,
  feature, cost, target, timestamp, or metadata field.

- `CANDIDATE`: the field is relevant to the research design
  but has not yet been confirmed for the core experimental
  feature set. Candidate fields must be evaluated during the
  TCP-CI pilot phase.

- `UNAVAILABLE`: the desired field cannot be obtained or
  reliably derived from the available data source.

- `EXCLUDED`: the field exists but is deliberately excluded
  from the modeling or experimental pipeline.

Status is independent of temporal leakage classification.

For example, a current-build test outcome can be
`CONFIRMED` as a target while also being marked `LEAKAGE`
for use as a predictor of the same build.

Status is also independent of the `required` flag. An optional
metadata field may be `CONFIRMED`, while a proposed feature may
remain `CANDIDATE`.

### Status lifecycle

The current dictionary represents a pre-pilot schema version.

The expected status transition is:

`CANDIDATE -> CONFIRMED`

when a field can be reliably obtained or derived and is
accepted into the pilot/core feature set.

A candidate may instead become:

`CANDIDATE -> UNAVAILABLE`

when the required information is not available or cannot be
derived reliably from TCP-CI.

A field may become:

`CONFIRMED -> EXCLUDED`

when it exists in the source data but is deliberately removed
from the modeling protocol, for example because of leakage,
poor reliability, redundancy, or scope-control decisions.

Status changes made during the pilot phase must be documented
rather than silently overwriting the original design rationale.

## Temporal rule

For a prediction made for build `b`, historical features must
only use information available before build `b`.

The outcome of a test execution in build `b` must not be used
as a predictor for the same build.

## Legacy Synthetic Dataset

### Scope

The legacy synthetic dataset is stored at:

`data/raw/synthetic/synthetic_data.csv`

It contains 5,000 test/Sprint observations covering 50
synthetic Sprints and 100 recurring test-case identifiers.

The dataset is used only for legacy reproduction and pipeline
sanity checking. It is not used as the primary empirical
evidence for the master's thesis.

### Observation granularity

Each row represents one synthetic test case within one
synthetic Sprint:

`(Sprint_ID, TestCase_ID)`

The corresponding data-dictionary granularity is:

`test_sprint`

- `test_sprint`: one test case within one legacy synthetic Sprint
- `test_job_execution`: one TCP-CI test execution within one CI job
- `test_build`: one normalized test case within one CI build

### Legacy fields

The dataset contains nine fields:

- `Sprint_ID`
- `TestCase_ID`
- `Code_Churn`
- `Cyclomatic_Complexity`
- `Dependencies`
- `Failure_History`
- `Execution_Time_ms`
- `Test_Age`
- `Label`

### Important provenance note

The semantic descriptions of the synthetic features represent
the software-engineering concepts that they were designed to
approximate. However, the values were generated synthetically
rather than extracted from an actual CI system.

In particular, `Failure_History` and `Test_Age` should not be
interpreted as true longitudinal histories reconstructed from
earlier Sprint observations. Their values were sampled directly
by the legacy synthetic-data generator.

The exact generation mechanism is documented in the
`derivation` column of `docs/data_dictionary.csv`.

## TCP-CI Canonical Schema

### Purpose

TCP-CI stores CI information across multiple source files and
uses source-specific field names. The thesis pipeline therefore
normalizes TCP-CI records into a canonical schema before feature
engineering or model training.

The machine-readable schema is stored at:

`configs/tcp_ci_schema_v1.yaml`

### Canonical observation

The target modeling granularity is:

`test_build`

Each normalized observation is conceptually identified by:

`(project_id, build_id, test_id)`

Raw execution data may contain job-level records with the
granularity:

`test_job_execution`

represented by:

`(project_id, build_id, job_id, test_id)`

The rule for consolidating multiple job-level executions of the
same test within one build is defined separately in the
experimental protocol.

### Core canonical fields

| Canonical field | Role | Required |
|---|---|---|
| `project_id` | identifier | yes |
| `build_id` | identifier | yes |
| `build_started_at` | timestamp | yes |
| `build_index` | temporal context | yes |
| `test_id` | identifier | yes |
| `execution_time_ms` | execution cost | yes |
| `failed` | target | yes |

### Feature groups

Candidate features are organized into three groups:

- **F1 — History**
- **F2 — Change/context**
- **F3 — Temporal**

Not every candidate feature is required for the initial pilot.
Actual availability is assessed separately for each TCP-CI
project before the feature set is frozen.

### Naming convention

Canonical fields use lowercase `snake_case`.

Source-specific names are preserved in the `source_field`
column of the data dictionary so that every transformation from
raw TCP-CI data to the canonical schema can be traced.