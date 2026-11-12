"""Synthetic, decision-time tests for the existing frozen CORE history transform."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.tcp_ci.build_history_features import FEATURE_COLUMNS, compute_history_features
from src.tcp_ci.rq2_causal_feature_materializer import (
    DECISION_COLUMNS,
    materialize_target_preexecution_features as materialize,
)


def _timestamp(build_index):
    return (pd.Timestamp("2026-01-01T00:00:00Z")
            + pd.Timedelta(days=build_index - 1)).isoformat()


def _fixture():
    history = pd.DataFrame([
        dict(project_id="synthetic", build_id=f"b{i:03d}",
             build_index=i, build_started_at=_timestamp(i), test_id=test,
             failed=int((i + j) % (j + 3) == 0),
             execution_time_ms=int(10 + j * 9 + (i % 4)),
             changed_entity_count=int(i % 5))
        for i in range(1, 14) for j, test in enumerate(("alpha", "beta"))
    ])
    context = pd.DataFrame([
        dict(project_id="synthetic", build_id="b014", build_index=14,
             build_started_at=_timestamp(14), test_id=test,
             changed_entity_count=2)
        for test in ("alpha", "beta", "gamma")
    ])
    oracle = context.copy()
    oracle["failed"] = [1, 0, 1]
    oracle["execution_time_ms"] = [25, 35, 45]
    return history, context, oracle


def _ordered(frame):
    return frame.sort_values("test_id").reset_index(drop=True).loc[:, list(DECISION_COLUMNS)]


def test_matches_frozen_full_offline_features_even_with_real_target_truth():
    history, context, oracle = _fixture()
    got = materialize(history, context)
    offline = compute_history_features(pd.concat([history, oracle], ignore_index=True))
    expected = offline.loc[offline.build_index == 14, list(DECISION_COLUMNS)]
    pd.testing.assert_frame_equal(_ordered(got), _ordered(expected), check_dtype=False)
    assert len(got) == 3
    assert set(FEATURE_COLUMNS).issubset(got.columns)
    assert "build_index" in got.columns
    assert "failed" not in got.columns and "execution_time_ms" not in got.columns
    assert "changed_entity_count" in got.columns



def test_output_covers_frozen_primary_f1_f2_f3_configuration():
    history, context, _ = _fixture()
    frozen = yaml.safe_load(Path("configs/core_features_v1.yaml").read_text())
    primary = frozen["feature_sets"]["F1_F2_F3"]["features"]
    assert len(primary) == 11
    assert set(primary).issubset(materialize(history, context).columns)


def test_causal_history_mean_and_failure_rate():
    history, context, _ = _fixture()
    got = materialize(history, context).set_index("test_id")
    alpha = history.loc[history.test_id == "alpha"]
    assert got.loc["alpha", "prior_execution_count"] == 13
    assert got.loc["alpha", "failure_rate_history"] == pytest.approx(alpha.failed.mean())
    assert got.loc["alpha", "execution_time_mean_history_ms"] == pytest.approx(
        alpha.execution_time_ms.mean())
    assert got.loc["gamma", "prior_execution_count"] == 0
    assert pd.isna(got.loc["gamma", "failure_rate_history"])
    assert got.loc["gamma", "test_age_builds"] == 0
    assert got.loc["alpha", "changed_entity_count"] == 2


def test_input_row_order_does_not_change_features():
    history, context, _ = _fixture()
    baseline = materialize(history, context)
    other = materialize(history.sample(frac=1, random_state=2),
                        context.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(baseline, other)


def test_current_truth_not_in_api_and_mutations_do_not_change_output():
    history, context, oracle = _fixture()
    baseline = materialize(history, context)
    for failed, duration in [(0, 0), (1, 900_000)]:
        changed = oracle.copy()
        changed["failed"] = failed
        changed["execution_time_ms"] = duration
        offline = compute_history_features(pd.concat([history, changed], ignore_index=True))
        target = offline.loc[offline.build_index == 14, list(DECISION_COLUMNS)]
        pd.testing.assert_frame_equal(_ordered(baseline), _ordered(target), check_dtype=False)
    with pytest.raises(ValueError, match="unexpected fields"):
        materialize(history, oracle)


def test_prefix_equivalence_after_appending_full_information_feedback():
    history, context, oracle = _fixture()
    before = materialize(history, context)
    later = pd.DataFrame([
        dict(project_id="synthetic", build_id="b015", build_index=15,
             build_started_at=_timestamp(15), test_id=t, changed_entity_count=1)
        for t in ("alpha", "beta", "gamma")
    ])
    advanced_history = pd.concat([history, oracle], ignore_index=True)
    advanced = materialize(advanced_history, later)
    offline = compute_history_features(pd.concat([
        advanced_history,
        later.assign(failed=0, execution_time_ms=0)
    ], ignore_index=True))
    expected = offline.loc[offline.build_index == 15, list(DECISION_COLUMNS)]
    pd.testing.assert_frame_equal(_ordered(advanced), _ordered(expected), check_dtype=False)
    assert advanced.set_index("test_id").loc["gamma", "prior_execution_count"] == 1
    assert advanced.set_index("test_id").loc["gamma", "recent_failure"] == 1
    pd.testing.assert_frame_equal(before, materialize(history, context))


def test_earlier_history_mutation_changes_next_features():
    history, context, oracle = _fixture()
    later = context.copy()
    later["build_id"] = "b015"
    later["build_index"] = 15
    later["build_started_at"] = _timestamp(15)
    h1 = pd.concat([history, oracle], ignore_index=True)
    changed_oracle = oracle.copy()
    changed_oracle.loc[changed_oracle.test_id == "alpha", "failed"] = 0
    changed_oracle.loc[changed_oracle.test_id == "alpha", "execution_time_ms"] = 999
    h2 = pd.concat([history, changed_oracle], ignore_index=True)
    a = materialize(h1, later).set_index("test_id")
    b = materialize(h2, later).set_index("test_id")
    assert a.loc["alpha", "recent_failure"] != b.loc["alpha", "recent_failure"]
    assert a.loc["alpha", "execution_time_mean_history_ms"] != b.loc[
        "alpha", "execution_time_mean_history_ms"]
    # The former target's outcome cannot affect its OWN decision-time features.
    pd.testing.assert_frame_equal(materialize(history, context),
                                  materialize(history, context))


def test_initial_build_can_have_entirely_cold_start_features():
    history, context, _ = _fixture()
    empty = history.iloc[0:0].copy()
    first = context.copy()
    first["build_index"] = 1
    first["build_id"] = "b001"
    first["build_started_at"] = _timestamp(1)
    result = materialize(empty, first)
    assert len(result) == 3
    assert (result["prior_execution_count"] == 0).all()
    assert result["failure_rate_history"].isna().all()


@pytest.mark.parametrize("change, message", [
    ("target_outcome", "unexpected fields"),
    ("target_computed_feature", "unexpected fields"),
    ("duplicate_target", "duplicate"),
    ("two_target_builds", "one target build"),
    ("cross_project", "projects differ"),
    ("history_target_overlap", "strictly earlier"),
    ("history_future", "strictly earlier"),
    ("history_after_target_timestamp", "follows target"),
    ("target_invalid_build_index", "invalid build_index"),
    ("history_fractional_duration", "invalid realized durations"),
    ("history_invalid_label", "invalid failure labels"),
    ("negative_change_count", "invalid changed_entity_count"),
    ("missing_start_time", "invalid build_started_at"),
    ("inconsistent_build_map", "inconsistent build_id/build_index"),
])
def test_fail_closed_on_invalid_provenance(change, message):
    history, context, oracle = _fixture()
    if change == "target_outcome":
        context["failed"] = 0
    elif change == "target_computed_feature":
        context["failure_rate_history"] = 0.5
    elif change == "duplicate_target":
        context = pd.concat([context, context.iloc[[0]]], ignore_index=True)
    elif change == "two_target_builds":
        extra = context.iloc[[0]].copy()
        extra["build_index"] = 15
        extra["build_id"] = "b015"
        extra["build_started_at"] = _timestamp(15)
        context = pd.concat([context, extra], ignore_index=True)
    elif change == "cross_project":
        history["project_id"] = "different"
    elif change == "history_target_overlap":
        history.loc[history.build_index == 13, "build_index"] = 14
        history.loc[history.build_index == 14, "build_id"] = "b014"
        history.loc[history.build_index == 14, "build_started_at"] = _timestamp(14)
    elif change == "history_future":
        history.loc[history.build_index == 13, "build_index"] = 15
        history.loc[history.build_index == 15, "build_id"] = "b015"
        history.loc[history.build_index == 15, "build_started_at"] = _timestamp(15)
    elif change == "history_after_target_timestamp":
        history.loc[history.build_index == 13, "build_started_at"] = _timestamp(30)
    elif change == "target_invalid_build_index":
        context["build_index"] = context["build_index"].astype(float)
        context.loc[context.index[0], "build_index"] = 14.5
    elif change == "history_fractional_duration":
        history["execution_time_ms"] = history["execution_time_ms"].astype(float)
        history.loc[history.index[0], "execution_time_ms"] = 8.5
    elif change == "history_invalid_label":
        history.loc[history.index[0], "failed"] = 5
    elif change == "negative_change_count":
        context.loc[context.index[0], "changed_entity_count"] = -1
    elif change == "missing_start_time":
        context.loc[context.index[0], "build_started_at"] = None
    elif change == "inconsistent_build_map":
        history.loc[history.index[0], "build_id"] = "alternate"
    with pytest.raises(ValueError, match=message):
        materialize(history, context)
