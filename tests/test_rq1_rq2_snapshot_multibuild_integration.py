"""WP5.17.16R2 synthetic-only tests of opt-in fitted-snapshot wiring.

No CORE data, no M2 policy approval; mocking isolates lifecycle/integration from
standalone real-estimator parity already covered by prior snapshot tests.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.tcp_ci import rq1_rq2_multibuild_prototype as multi
from src.tcp_ci import rq1_rq2_single_build_bridge as bridge

CONFIG = {
    "model_name": "logistic_regression",
    "probability_variant": "uncalibrated",
    "rho": 0.95,
    "rq1_config": {},
    "model_config": {},
    "calibration_config": {},
    "core_features_config": {},
}


def _frames(*, causal=False):
    development, target, oracle = [], [], []
    for b in (1, 2):
        for t in ("a", "b"):
            row = {
                "project_id": "synthetic-snapshot",
                "build_id": f"build-{b}",
                "build_index": b,
                "test_id": t,
                "failed": int((b == 2 and t == "a") or (b == 1 and t == "b")),
                "execution_time_ms": 10 if t == "a" else 20,
            }
            if causal:
                row.update(build_started_at=f"2024-01-0{b}T00:00:00Z", changed_entity_count=b)
            else:
                row.update(execution_time_mean_history_ms=10 if t == "a" else 20)
            development.append(row)
    for b in (3, 4):
        for t in ("a", "b"):
            ident = {
                "project_id": "synthetic-snapshot",
                "build_id": f"build-{b}",
                "build_index": b,
                "test_id": t,
            }
            context = dict(ident)
            if causal:
                context.update(build_started_at=f"2024-01-0{b}T00:00:00Z", changed_entity_count=b)
            else:
                context["execution_time_mean_history_ms"] = 10 if t == "a" else 20
            target.append(context)
            oracle.append(dict(ident, failed=int(t == "a" and b == 3),
                               execution_time_ms=10 if t == "a" else 20))
    return pd.DataFrame(development), pd.DataFrame(target), pd.DataFrame(oracle)


def _patch_fit_and_predict(monkeypatch):
    observed = {"fitted": [], "predicted": [], "snapshots": []}

    def fake_fit(development, **kwargs):
        observed["fitted"].append(development.copy(deep=True))
        assert set(pd.to_numeric(development["build_index"])) == {1, 2}
        assert set(development["project_id"]) == {"synthetic-snapshot"}
        return SimpleNamespace(
            model_name=kwargs["model_name"], project_id="synthetic-snapshot",
            feature_columns=("execution_time_mean_history_ms", "failure_rate_history", "changed_entity_count")
            if "failure_rate_history" in development.columns
            else ("execution_time_mean_history_ms",),
            last_development_build_index=2, model_fit_last_build_index=1,
            calibration_support_sufficient=True,
        )

    def fake_predict(snapshot, target_preexecution):
        observed["snapshots"].append(snapshot)
        observed["predicted"].append(target_preexecution.copy(deep=True))
        assert not {"failed", "execution_time_ms"}.intersection(target_preexecution.columns)
        assert set(target_preexecution.columns) == {
            "project_id", "build_id", "build_index", "test_id", *snapshot.feature_columns
        }
        out = target_preexecution.loc[:, [
            "project_id", "build_id", "build_index", "test_id"
        ]].copy(deep=True)
        out["p_uncalibrated"] = [0.99 if t == "a" else 0.01 for t in out["test_id"]]
        out["p_platt"] = out["p_uncalibrated"]
        out["p_isotonic"] = out["p_uncalibrated"]
        return out

    monkeypatch.setattr(multi, "fit_rq1_snapshot", fake_fit)
    monkeypatch.setattr(bridge, "predict_from_rq1_snapshot", fake_predict)
    return observed


def _run(frames, **options):
    return multi.run_synthetic_multibuild_rq1_rq2(
        *frames, **CONFIG, snapshot_mode="fixed_initial_synthetic", **options
    )


def test_fixed_initial_snapshot_fits_once_scores_two_builds(monkeypatch):
    frames = _frames()
    observed = _patch_fit_and_predict(monkeypatch)
    result = _run(frames)
    assert len(observed["fitted"]) == 1
    assert len(observed["predicted"]) == 2
    assert observed["snapshots"][0] is observed["snapshots"][1]
    assert [int(p["build_index"].iloc[0]) for p in observed["predicted"]] == [3, 4]
    assert len(result.steps) == 2
    assert result.development_rows_before == (4, 6)
    assert result.history_rows_before == (4, 6)
    assert [set(step.selected_test_ids) for step in result.steps] == [{"a"}, {"a"}]
    assert all(set(step.prediction_view.columns) >= {"p_uncalibrated"} for step in result.steps)
    assert all("failed" not in step.prediction_view for step in result.steps)


def test_oracle_truth_mutation_does_not_affect_current_selection(monkeypatch):
    observed = _patch_fit_and_predict(monkeypatch)
    frames = _frames()
    base = _run(frames)
    changed_oracle = frames[2].copy(deep=True)
    changed_oracle.loc[changed_oracle["build_index"] == 3, "failed"] = 0
    changed_oracle.loc[changed_oracle["build_index"] == 3, "execution_time_ms"] = 99
    changed = _run((frames[0], frames[1], changed_oracle))
    assert len(observed["fitted"]) == 2  # once in each independent run
    for one, two in zip(base.steps, changed.steps):
        assert one.selected_test_ids == two.selected_test_ids
        pd.testing.assert_frame_equal(one.prediction_view, two.prediction_view)
    assert base.steps[0].report != changed.steps[0].report


def test_causal_history_features_rematerialized_but_model_reused(monkeypatch):
    observed = _patch_fit_and_predict(monkeypatch)
    frames = _frames(causal=True)
    result = _run(frames, preexecution_mode="causal_from_history")
    assert len(result.steps) == 2
    assert len(observed["fitted"]) == 1
    assert len(observed["predicted"]) == 2
    assert "failure_rate_history" in observed["predicted"][1].columns
    assert "changed_entity_count" in observed["predicted"][1].columns

    mutated = frames[2].copy(deep=True)
    mutated.loc[(mutated["build_index"] == 3) & (mutated["test_id"] == "a"), "failed"] = 0
    _run((frames[0], frames[1], mutated), preexecution_mode="causal_from_history")
    assert len(observed["fitted"]) == 2
    assert len(observed["predicted"]) == 4
    pd.testing.assert_frame_equal(observed["predicted"][0], observed["predicted"][2])
    before = observed["predicted"][1].set_index("test_id").loc["a", "failure_rate_history"]
    after = observed["predicted"][3].set_index("test_id").loc["a", "failure_rate_history"]
    assert before != after  # feedback only to later target's features


def test_invalid_snapshot_mode_rejected(monkeypatch):
    frames = _frames()
    with pytest.raises(ValueError, match="snapshot_mode"):
        multi.run_synthetic_multibuild_rq1_rq2(*frames, **CONFIG, snapshot_mode="scheduled_final")


def test_legacy_default_still_uses_original_bridge(monkeypatch):
    frames = _frames()
    called = []

    def fake_bridge(**kwargs):
        called.append(kwargs)
        assert kwargs.get("rq1_snapshot") is None
        return SimpleNamespace(selected_test_ids=("a",), prediction_view=pd.DataFrame())

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("legacy default must not create a fitted snapshot")

    monkeypatch.setattr(multi, "fit_rq1_snapshot", unexpected_fit)
    monkeypatch.setattr(multi, "evaluate_one_build_with_rq1", fake_bridge)
    result = multi.run_synthetic_multibuild_rq1_rq2(*frames, **CONFIG)
    assert len(called) == 2
    assert len(result.steps) == 2
    assert result.development_rows_before == (4, 6)

@pytest.mark.parametrize("causal", [False, True], ids=["supplied_synthetic", "causal_from_history"])
def test_fixed_initial_snapshot_none_fails_closed(monkeypatch, causal):
    """Invalid fit output must never silently fall back to per-build RQ1 fitting."""
    frames = _frames(causal=causal)
    fitting_calls = []

    def unexpected_none(*args, **kwargs):
        fitting_calls.append("fit")
        return None

    def forbidden_refit(**kwargs):
        raise AssertionError("fixed_initial_synthetic silently refit RQ1")

    monkeypatch.setattr(multi, "fit_rq1_snapshot", unexpected_none)
    monkeypatch.setattr(multi, "evaluate_one_build_with_rq1", forbidden_refit)
    opts = {"preexecution_mode": "causal_from_history"} if causal else {}
    with pytest.raises(ValueError, match="fixed_initial_synthetic requires a fitted"):
        _run(frames, **opts)
    assert fitting_calls == ["fit"]
