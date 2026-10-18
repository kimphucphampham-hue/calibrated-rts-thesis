from pathlib import Path

import pandas as pd

from src.tcp_ci import (
    core_rq1_runner,
)

from src.tcp_ci.rq1_pipeline import (
    PREDICTION_ID_COLUMNS,
    RQ1_MODEL_NAMES,
)


def test_core_rq1_contract_resolves_ten_projects():
    contract = (
        core_rq1_runner
        .load_core_rq1_contract(
            core_rq1_runner
            .DEFAULT_CONFIG
        )
    )

    assert len(
        contract[
            "projects"
        ]
    ) == 10

    assert len(
        contract[
            "feature_columns"
        ]
    ) == 11

    assert (
        contract[
            "core"
        ][
            "primary_final_rq1_evidence"
        ]
        is False
    )


def test_safe_project_name_matches_core_paths():
    assert (
        core_rq1_runner
        .safe_project_name(
            "datasets/JMRI@JMRI"
        )
        ==
        "datasets__JMRI__JMRI"
    )


def test_required_input_columns_are_unique():
    contract = (
        core_rq1_runner
        .load_core_rq1_contract(
            core_rq1_runner
            .DEFAULT_CONFIG
        )
    )

    columns = (
        core_rq1_runner
        .required_input_columns(
            contract[
                "feature_columns"
            ]
        )
    )

    assert len(
        columns
    ) == len(
        set(
            columns
        )
    )

    for column in (
        PREDICTION_ID_COLUMNS
    ):
        assert (
            column
            in columns
        )


def test_run_project_reuses_frozen_rq1_components(
    tmp_path,
    monkeypatch,
):
    project_key = (
        "datasets/test@test-project"
    )

    safe = (
        core_rq1_runner
        .safe_project_name(
            project_key
        )
    )

    split_root = (
        tmp_path
        / "split"
    )

    project_root = (
        split_root
        / safe
    )

    project_root.mkdir(
        parents=True
    )

    feature_columns = [
        "feature_a",
    ]

    development = pd.DataFrame(
        {
            "project_id":
                [project_key] * 4,
            "build_id":
                ["1", "2", "3", "4"],
            "build_index":
                [1, 2, 3, 4],
            "test_id":
                ["t"] * 4,
            "failed":
                [0, 1, 0, 1],
            "feature_a":
                [0.0, 1.0, 2.0, 3.0],
        }
    )

    evaluation = pd.DataFrame(
        {
            "project_id":
                [project_key] * 2,
            "build_id":
                ["5", "6"],
            "build_index":
                [5, 6],
            "test_id":
                ["t", "t"],
            "failed":
                [0, 1],
            "feature_a":
                [4.0, 5.0],
        }
    )

    development.to_csv(
        project_root
        / "train.csv",
        index=False,
    )

    evaluation.to_csv(
        project_root
        / "test.csv",
        index=False,
    )

    model_calls = []
    evaluator_calls = []

    def fake_run_rq1_model(
        development,
        evaluation,
        model_name,
        rq1_config,
        model_config,
        calibration_config,
        core_features_config,
    ):
        model_calls.append(
            model_name
        )

        predictions = (
            evaluation[
                [
                    "project_id",
                    "build_id",
                    "build_index",
                    "test_id",
                    "failed",
                ]
            ]
            .copy()
        )

        predictions[
            "p_uncalibrated"
        ] = 0.1

        predictions[
            "p_platt"
        ] = 0.2

        predictions[
            "p_isotonic"
        ] = 0.3

        diagnostics = {
            "model":
                model_name,
            "status":
                "SUFFICIENT",
            "calibration_support_sufficient":
                True,
            "uncalibrated_available":
                True,
            "platt_available":
                True,
            "isotonic_available":
                True,
        }

        return (
            predictions,
            diagnostics,
        )

    def fake_evaluate(
        predictions,
        diagnostics,
        n_bins=10,
    ):
        evaluator_calls.append(
            diagnostics[
                "model"
            ]
        )

        rows = []

        for variant in [
            "uncalibrated",
            "platt",
            "isotonic",
        ]:
            rows.append(
                {
                    "project_id":
                        project_key,
                    "model":
                        diagnostics[
                            "model"
                        ],
                    "calibration_variant":
                        variant,
                    "condition_status":
                        "AVAILABLE",
                    "calibration_support_sufficient":
                        True,
                }
            )

        return (
            pd.DataFrame(
                rows
            ),
            pd.DataFrame(),
        )

    monkeypatch.setattr(
        core_rq1_runner,
        "run_rq1_model",
        fake_run_rq1_model,
    )

    monkeypatch.setattr(
        core_rq1_runner,
        "evaluate_rq1_predictions",
        fake_evaluate,
    )

    predictions_root = (
        tmp_path
        / "predictions"
    )

    (
        metrics,
        bins,
        diagnostics,
    ) = (
        core_rq1_runner
        .run_project(
            project_key=
                project_key,
            split_root=
                split_root,
            predictions_root=
                predictions_root,
            train_filename=
                "train.csv",
            test_filename=
                "test.csv",
            feature_columns=
                feature_columns,
            rq1_config={},
            model_config={},
            calibration_config={},
            core_features_config={},
            n_bins=10,
        )
    )

    assert (
        model_calls
        == list(
            RQ1_MODEL_NAMES
        )
    )

    assert (
        evaluator_calls
        == list(
            RQ1_MODEL_NAMES
        )
    )

    assert len(
        metrics
    ) == 9

    assert bins.empty

    assert len(
        diagnostics
    ) == 3

    for model_name in (
        RQ1_MODEL_NAMES
    ):
        assert (
            predictions_root
            / safe
            / f"{model_name}.csv"
        ).exists()


def test_project_identity_mismatch_is_rejected():
    data = pd.DataFrame(
        {
            "project_id":
                ["wrong-project"],
        }
    )

    try:
        (
            core_rq1_runner
            .validate_project_identity(
                data,
                "expected-project",
                "development",
            )
        )
    except ValueError as error:
        assert (
            "identity mismatch"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected project identity "
            "mismatch to be rejected."
        )
