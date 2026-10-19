# Protocol Amendment Register

## 1. Purpose

This register assigns a unique, stable ID to every amendment
of `docs/protocol_v1.md`.

Two amendment files both carry the title "Protocol Amendment
001". Their filenames are referenced by frozen configs and
tests, so they are not renamed and their contents are not
changed. The register ID below is the authoritative
identifier; the in-file title "001" is historical.

IDs are assigned in chronological order of the freezing
commit.

## 2. Register

| ID | Title | File | Freezing commit | Date |
| --- | --- | --- | --- | --- |
| AMD-001 | RQ1 Calibration Design | `docs/protocol_amendment_001_rq1_calibration.md` | `6699ba3` | 2026-10-17 |
| AMD-002 | Phase 11 Core Project Eligibility Operationalization | `docs/protocol_amendment_001_core_project_eligibility.md` | `e6d00ff` (created); `f1a20a1` (eligibility rule 10 revised) | 2026-10-18 |

New amendments take the next free ID (AMD-003, …). A new
amendment file should use its register ID in its filename
and title.

## 3. AMD-001 — RQ1 Calibration Design

**File:** `docs/protocol_amendment_001_rq1_calibration.md`
(unchanged)

**Governs:**

- development-period model-fit/calibration partition;
- rare-event backward calibration-window expansion and
  minimum calibration support;
- Platt and isotonic calibration methods and calibrator input;
- RQ1 metrics including reliability diagrams (§10), ECE
  definition and calibration-bias convention;
- per-project reporting (§13).

**Implemented by:** `configs/calibration_v1.yaml`,
`configs/rq1_v1.yaml`, `src/tcp_ci/calibration_split.py`,
`src/tcp_ci/probability_calibration.py`.

**Referenced by path from:** no config, test or document.
The Phase 11 execution chain reaches it only indirectly
through `configs/calibration_v1.yaml`, which
`configs/core_rq1_v1.yaml` references. This register and
`docs/phase11_core_scale_milestone_v1.md` §16.7 close that
traceability gap.

**Protocol §17 compliance:**

| Requirement | Present | Where |
| --- | --- | --- |
| What changed | yes | §1, §3–§13 |
| Why it changed | yes | §1 |
| Previous results inspected | yes | §14 |
| Experiments to rerun | partial | §15 states that later changes require an amendment before affected reruns; it does not list experiments to rerun because of this amendment |

**Supersedes:** `docs/protocol_v1.md` §18.1 ("Probability
calibration not yet implemented"), as an implementation
status. Protocol v1 is frozen and is not edited.

**Used by:** Phase 11 CORE RQ1 run
(`docs/phase11_core_scale_milestone_v1.md`).

## 4. AMD-002 — Phase 11 Core Project Eligibility Operationalization

**File:** `docs/protocol_amendment_001_core_project_eligibility.md`
(legacy filename, unchanged)

**Governs:**

- structural project-eligibility rules for Phase 11;
- the deterministic ten-project execution-set rule;
- the boundary that Phase 11 results are not primary final
  RQ1 evidence.

**Implemented by:** `configs/core_project_screening_v1.yaml`,
`configs/core_projects_v1.yaml`,
`src/tcp_ci/screen_core_projects.py`.

**Referenced by path from:**

- `configs/core_project_screening_v1.yaml`
- `configs/core_projects_v1.yaml`
- `configs/core_rq1_v1.yaml`
- `tests/test_core_project_screening_config.py`

**Protocol §17 compliance:**

| Requirement | Present | Where |
| --- | --- | --- |
| What changed | yes | §1, §2 |
| Why it changed | yes | §3 |
| Previous results inspected | yes | §4 |
| Experiments to rerun | yes | §5 |

**Supersedes:** nothing in `docs/protocol_v1.md`; it
operationalizes §7 for Phase 11 only. It does not define
final walk-forward eligibility.

**Revision history:**

| Commit | Time | Change |
| --- | --- | --- |
| `e6d00ff` | 2026-10-18 14:50 +0700 | File created ("Freeze Phase 11 core project eligibility"). |
| `f1a20a1` | 2026-10-18 15:05 +0700 | Eligibility rule 10 reworded from "the model-fit partition contains both target classes" to "the frozen minimum model-fit prefix contains both target classes, so backward calibration-window expansion cannot leave the base model with a single-class fitting partition". Committed together with the screening implementation (`src/tcp_ci/screen_core_projects.py`), before the project-set freeze `096d74c` (15:14 +0700). No other change to the file. |

**Screening results inspected before the `f1a20a1`
revision:** UNVERIFIED. No record states whether any
screening output was seen before rule 10 was reworded. The
screening script is first committed in `f1a20a1` itself. The
only surviving screening outputs, kept outside the
repository, have modification times of 15:06 +0700, after
`f1a20a1`. That timing does not rule out an earlier,
overwritten run, so it is not treated as evidence.

**Used by:** Phase 11 screening, project-set freeze and CORE
RQ1 run.

## 5. Protocol §18 status

`docs/protocol_v1.md` §18 lists known issues as of the pilot.
Protocol v1 is frozen; current status is recorded here.

| Protocol §18 entry | Status | Source |
| --- | --- | --- |
| 18.1 Probability calibration not yet implemented | Implemented (implementation validation) | AMD-001; Phase 11 milestone |
| 18.2 Risk-bounded optimization not yet implemented | Still open | — |
| 18.3 Core walk-forward experiment not yet executed | Still open | — |
| 18.4–18.7 | Unchanged | — |

## 6. Change control

This register is updated whenever a protocol amendment is
added. Adding a register entry does not change any frozen
methodology.
