"""Pre-freeze F2 temporal missing-timestamp regression tests.

Unit-scope synthetic CSV fixtures only. The commit-token parser is stubbed
solely to isolate the as-of timestamp branch. No real CORE data used.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.tcp_ci import build_history_features as bhf


CUTOFF = "2026-01-02T00:00:00Z"
COMMIT = "commit-X"


def _feature(tmp_path, dates, *, canonical_time=CUTOFF, linked_commit=COMMIT):
    raw_dir = tmp_path
    pd.DataFrame({
        "id": ["build-1"],
        "commits": [linked_commit],
        "started_at": [CUTOFF],
    }).to_csv(raw_dir / "builds.csv", index=False)
    pd.DataFrame({
        "Commit": [COMMIT] * len(dates),
        "CommitDate": dates,
        "EntityId": [f"entity-{i}" for i in range(len(dates))],
    }).to_csv(raw_dir / "entity_change_history.csv", index=False)
    canonical = pd.DataFrame({
        "build_id": ["build-1"],
        "build_started_at": [canonical_time],
    })
    with patch.object(
        bhf, "parse_commit_tokens",
        side_effect=lambda raw: [str(raw)] if pd.notna(raw) else [],
    ):
        result = bhf.compute_changed_entity_count(canonical, raw_dir)
    assert list(result.columns) == ["build_id", "changed_entity_count"]
    assert result["build_id"].tolist() == ["build-1"]
    return result.loc[0, "changed_entity_count"]


@pytest.mark.parametrize(
    "dates,canonical_time,linked_commit,expected",
    [
        (["2026-01-01T00:00:00Z"], CUTOFF, COMMIT, 1.0),
        ([CUTOFF], CUTOFF, COMMIT, 1.0),
        (["2026-01-03T00:00:00Z"], CUTOFF, COMMIT, None),
        (["2026-01-01T00:00:00Z"], CUTOFF, "unmatched", None),
        ([None], CUTOFF, COMMIT, None),
        (["2026-01-01T00:00:00Z"], None, COMMIT, None),
        (["2026-01-01T00:00:00Z", None], CUTOFF, COMMIT, None),
        ([None], None, COMMIT, None),
    ],
    ids=[
        "past-allowed", "equal-cutoff-allowed", "future-rejected",
        "unmatched-commit-rejected", "missing-CommitDate-rejected",
        "missing-canonical-start-rejected", "partially-missing-CommitDate-rejected",
        "both-timestamps-missing-rejected",
    ],
)
def test_f2_timestamp_provenance_fail_closed(
    tmp_path, dates, canonical_time, linked_commit, expected,
):
    result = _feature(
        tmp_path, dates,
        canonical_time=canonical_time,
        linked_commit=linked_commit,
    )
    if expected is None:
        assert pd.isna(result), "Unknown/future temporal provenance must yield missing F2"
    else:
        assert float(result) == expected
