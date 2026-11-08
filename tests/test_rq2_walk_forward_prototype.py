"""Synthetic, non-confirmatory checks for Option A chronology boundaries."""

import pandas as pd
import pytest

from src.tcp_ci.rq2_walk_forward_prototype import run_retrospective_prototype


def records():
    rows = []
    for b, failures, durations in [
        (1, (1, 0), (10, 20)),
        (2, (1, 0), (30, 40)),
        (3, (0, 1), (50, 60)),
    ]:
        for t, failed, duration in zip(("a", "b"), failures, durations):
            rows.append({
                "project_id": "p", "build_index": b, "test_id": t,
                "failed": failed, "execution_time_ms": duration,
            })
    return pd.DataFrame(rows)


def predict(history, decision):
    assert "failed" not in decision.columns
    assert "execution_time_ms" not in decision.columns
    assert (history["build_index"] < int(decision["build_index"].iloc[0])).all()
    return [0.8 if t == "a" else 0.2 for t in decision["test_id"]]


def test_prior_only_history_and_cost_means():
    seen = {}

    def checker(history, decision):
        idx = int(decision["build_index"].iloc[0])
        seen[idx] = (
            tuple(sorted(history["build_index"].unique())),
            dict(zip(decision["test_id"], decision["execution_time_mean_history_ms"])),
        )
        return predict(history, decision)

    steps = run_retrospective_prototype(records(), checker, 0.5)
    assert len(steps) == 3
    assert seen[1][0] == ()
    assert seen[2] == ((1,), {"a": 10.0, "b": 20.0})
    assert seen[3] == ((1, 2), {"a": 20.0, "b": 30.0})


def test_current_target_truth_mutation_cannot_change_selection():
    initial = records()
    altered = records()
    altered.loc[altered.build_index == 2, "failed"] = [0, 1]
    altered.loc[altered.build_index == 2, "execution_time_ms"] = [150, 15]
    first = run_retrospective_prototype(initial, predict, 0.5, start_build_index=2)[0]
    second = run_retrospective_prototype(altered, predict, 0.5, start_build_index=2)[0]
    assert first.selected_test_ids == second.selected_test_ids
    assert first.report["observed_selected_ms"] != second.report["observed_selected_ms"]


def test_future_truth_mutation_cannot_change_earlier_selection_or_score():
    initial = records()
    altered = records()
    altered.loc[altered.build_index == 3, "failed"] = [1, 0]
    altered.loc[altered.build_index == 3, "execution_time_ms"] = [500, 600]
    first = run_retrospective_prototype(initial, predict, 0.5, start_build_index=2)[0]
    second = run_retrospective_prototype(altered, predict, 0.5, start_build_index=2)[0]
    assert first.selected_test_ids == second.selected_test_ids
    assert first.report["observed_selected_ms"] == second.report["observed_selected_ms"]


def test_prefix_equals_full_history_decision():
    full = run_retrospective_prototype(records(), predict, 0.5, start_build_index=2)[0]
    prefix = run_retrospective_prototype(
        records().loc[lambda x: x.build_index <= 2], predict, 0.5,
        start_build_index=2,
    )[0]
    assert full.selected_test_ids == prefix.selected_test_ids
    assert full.report["observed_selected_ms"] == prefix.report["observed_selected_ms"]


def test_initial_no_positive_history_uses_explicit_existing_fallback():
    first = run_retrospective_prototype(records(), predict, 0.5)[0]
    assert first.selection_mode == "fallback_retest_all"
    assert set(first.selected_test_ids) == {"a", "b"}


def test_order_independent_decision():
    original = run_retrospective_prototype(records(), predict, 0.5)
    shuffled = run_retrospective_prototype(
        records().sample(frac=1, random_state=7), predict, 0.5
    )
    assert [s.selected_test_ids for s in original] == [
        s.selected_test_ids for s in shuffled
    ]


def test_cross_project_input_is_rejected_not_silently_pooled():
    mixed = records()
    mixed.loc[mixed.index[-1], "project_id"] = "q"
    with pytest.raises(ValueError, match="one project"):
        run_retrospective_prototype(mixed, predict, 0.5)


def test_duplicate_canonical_row_is_rejected():
    data = records()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate canonical"):
        run_retrospective_prototype(data, predict, 0.5)


def test_invalid_callback_probabilities_are_rejected():
    with pytest.raises(ValueError, match="Probabilities"):
        run_retrospective_prototype(records(), lambda h, d: [-0.1, 0.2], 0.5)


def test_m2_02_zero_risk_not_silently_treated_as_approved():
    with pytest.raises(ValueError):
        run_retrospective_prototype(records(), lambda h, d: [0.0, 0.0], 0.5)
