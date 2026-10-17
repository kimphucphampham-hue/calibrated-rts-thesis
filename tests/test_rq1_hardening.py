from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.tcp_ci.rq1_pipeline as rq1_pipeline

from src.tcp_ci.rq1_pipeline import (
    load_yaml,
    run_rq1_model,
    run_sufficient_rq1_model,
)


CONFIG_DIR = Path("configs")


def load_configs():
    return (
        load_yaml(
            CONFIG_DIR
            / "rq1_v1.yaml"
        ),
        load_yaml(
            CONFIG_DIR
            / "rq1_models_v1.yaml"
        ),
        load_yaml(
            CONFIG_DIR
            / "calibration_v1.yaml"
        ),
        load_yaml(
            CONFIG_DIR
            / "core_features_v1.yaml"
        ),
    )


def make_partition(
    builds,
    failure_builds,
    project_id="project-a",
) -> pd.DataFrame:
    rows = []

    for build_index in builds:
        rows.append(
            {
                "project_id":
                    project_id,

                "build_id":
                    f"build-{build_index}",

                "build_index":
                    build_index,

                "test_id":
                    f"test-{build_index % 3}",

                "failed":
                    int(
                        build_index
                        in failure_builds
                    ),

                "failure_rate_history":
                    (
                        build_index % 10
                    ) / 10.0,

                "recent_failure":
                    int(
                        build_index % 7
                        == 0
                    ),

                "consecutive_failures":
                    build_index % 3,

                "consecutive_passes":
                    build_index % 5,

                "test_age_builds":
                    build_index - 1,

                "time_since_last_failure_builds":
                    float(
                        build_index % 11
                        + 1
                    ),

                "execution_time_mean_history_ms":
                    float(
                        100
                        + build_index
                    ),

                "changed_entity_count":
                    float(
                        build_index % 6
                    ),

                "recent_failure_density_10":
                    (
                        build_index % 5
                    ) / 10.0,

                "execution_time_trend_10":
                    float(
                        build_index % 9
                        - 4
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_sufficient_data(
    project_id="project-a",
):
    development = make_partition(
        builds=range(
            1,
            131,
        ),
        failure_builds={
            10,
            20,
            110,
            115,
            120,
            125,
            129,
        },
        project_id=project_id,
    )

    evaluation = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            132,
            134,
        },
        project_id=project_id,
    )

    return (
        development,
        evaluation,
    )


def test_evaluation_labels_do_not_change_probabilities():
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation_a,
    ) = make_sufficient_data()

    evaluation_b = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            131,
            133,
            135,
        },
    )

    predictions_a, _ = run_rq1_model(
        development=development,
        evaluation=evaluation_a,
        model_name=
            "logistic_regression",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    predictions_b, _ = run_rq1_model(
        development=development,
        evaluation=evaluation_b,
        model_name=
            "logistic_regression",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    assert not predictions_a[
        "failed"
    ].equals(
        predictions_b[
            "failed"
        ]
    )

    for column in [
        "p_uncalibrated",
        "p_platt",
        "p_isotonic",
    ]:
        np.testing.assert_allclose(
            predictions_a[
                column
            ].to_numpy(),
            predictions_b[
                column
            ].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )


def test_evaluation_features_do_not_change_fit_inputs(
    monkeypatch,
):
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation_a,
    ) = make_sufficient_data()

    evaluation_b = (
        evaluation_a.copy()
    )

    evaluation_b[
        "failure_rate_history"
    ] = (
        evaluation_b[
            "failure_rate_history"
        ]
        + 1000000.0
    )

    original_fit = (
        rq1_pipeline.fit_probability_model
    )

    original_platt_builder = (
        rq1_pipeline.build_platt_calibrator
    )

    original_isotonic_builder = (
        rq1_pipeline.build_isotonic_calibrator
    )

    model_fit_inputs = []

    calibrator_fit_inputs = {
        "platt": [],
        "isotonic": [],
    }

    def spy_fit(
        model,
        X_train,
        y_train,
    ):
        model_fit_inputs.append(
            (
                X_train.reset_index(
                    drop=True
                ).copy(),

                pd.Series(
                    y_train
                ).reset_index(
                    drop=True
                ).copy(),
            )
        )

        return original_fit(
            model,
            X_train,
            y_train,
        )

    class SpyCalibrator:
        def __init__(
            self,
            delegate,
            name,
        ):
            self.delegate = delegate
            self.name = name

        def fit(
            self,
            probability,
            target,
        ):
            calibrator_fit_inputs[
                self.name
            ].append(
                (
                    np.asarray(
                        probability,
                        dtype=float,
                    ).copy(),

                    np.asarray(
                        target,
                        dtype=int,
                    ).copy(),
                )
            )

            self.delegate.fit(
                probability,
                target,
            )

            return self

        def transform(
            self,
            probability,
        ):
            return self.delegate.transform(
                probability
            )

    def spy_platt_builder(
        config,
    ):
        return SpyCalibrator(
            original_platt_builder(
                config
            ),
            "platt",
        )

    def spy_isotonic_builder(
        config,
    ):
        return SpyCalibrator(
            original_isotonic_builder(
                config
            ),
            "isotonic",
        )

    monkeypatch.setattr(
        rq1_pipeline,
        "fit_probability_model",
        spy_fit,
    )

    monkeypatch.setattr(
        rq1_pipeline,
        "build_platt_calibrator",
        spy_platt_builder,
    )

    monkeypatch.setattr(
        rq1_pipeline,
        "build_isotonic_calibrator",
        spy_isotonic_builder,
    )

    run_rq1_model(
        development=development,
        evaluation=evaluation_a,
        model_name=
            "logistic_regression",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    run_rq1_model(
        development=development,
        evaluation=evaluation_b,
        model_name=
            "logistic_regression",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    assert (
        len(
            model_fit_inputs
        )
        == 2
    )

    pd.testing.assert_frame_equal(
        model_fit_inputs[
            0
        ][
            0
        ],
        model_fit_inputs[
            1
        ][
            0
        ],
    )

    pd.testing.assert_series_equal(
        model_fit_inputs[
            0
        ][
            1
        ],
        model_fit_inputs[
            1
        ][
            1
        ],
    )

    for name in [
        "platt",
        "isotonic",
    ]:
        assert (
            len(
                calibrator_fit_inputs[
                    name
                ]
            )
            == 2
        )

        np.testing.assert_allclose(
            calibrator_fit_inputs[
                name
            ][
                0
            ][
                0
            ],
            calibrator_fit_inputs[
                name
            ][
                1
            ][
                0
            ],
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_array_equal(
            calibrator_fit_inputs[
                name
            ][
                0
            ][
                1
            ],
            calibrator_fit_inputs[
                name
            ][
                1
            ][
                1
            ],
        )


def test_missing_calibration_build_is_rejected(
    monkeypatch,
):
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    original_materialize = (
        rq1_pipeline
        .materialize_calibration_partitions
    )

    def broken_materialize(
        data,
        split,
        build_column="build_index",
    ):
        (
            model_fit,
            calibration_data,
        ) = original_materialize(
            data,
            split,
            build_column=
                build_column,
        )

        calibration_data = (
            calibration_data
            .iloc[:-1]
            .copy()
        )

        return (
            model_fit,
            calibration_data,
        )

    monkeypatch.setattr(
        rq1_pipeline,
        "materialize_calibration_partitions",
        broken_materialize,
    )

    with pytest.raises(
        AssertionError,
        match=(
            "calibration builds do not match"
        ),
    ):
        run_sufficient_rq1_model(
            development=development,
            evaluation=evaluation,
            model_name=
                "logistic_regression",
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )


def test_cross_project_evaluation_is_rejected():
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        _,
    ) = make_sufficient_data(
        project_id=
            "project-a",
    )

    evaluation = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            132,
        },
        project_id=
            "project-b",
    )

    with pytest.raises(
        ValueError,
        match=(
            "same project"
        ),
    ):
        run_rq1_model(
            development=development,
            evaluation=evaluation,
            model_name=
                "logistic_regression",
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )


def test_reversed_outer_chronology_is_rejected():
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        _,
    ) = make_sufficient_data()

    evaluation = make_partition(
        builds=range(
            -5,
            0,
        ),
        failure_builds={
            -3,
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "strictly precede"
        ),
    ):
        run_rq1_model(
            development=development,
            evaluation=evaluation,
            model_name=
                "logistic_regression",
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )


def test_calibrators_are_not_shared_across_runs(
    monkeypatch,
):
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    original_platt_builder = (
        rq1_pipeline.build_platt_calibrator
    )

    original_isotonic_builder = (
        rq1_pipeline.build_isotonic_calibrator
    )

    platt_instances = []
    isotonic_instances = []

    def spy_platt_builder(
        config,
    ):
        calibrator = (
            original_platt_builder(
                config
            )
        )

        platt_instances.append(
            calibrator
        )

        return calibrator

    def spy_isotonic_builder(
        config,
    ):
        calibrator = (
            original_isotonic_builder(
                config
            )
        )

        isotonic_instances.append(
            calibrator
        )

        return calibrator

    monkeypatch.setattr(
        rq1_pipeline,
        "build_platt_calibrator",
        spy_platt_builder,
    )

    monkeypatch.setattr(
        rq1_pipeline,
        "build_isotonic_calibrator",
        spy_isotonic_builder,
    )

    development_a, evaluation_a = (
        make_sufficient_data(
            project_id=
                "project-a",
        )
    )

    run_rq1_model(
        development=development_a,
        evaluation=evaluation_a,
        model_name=
            "logistic_regression",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    run_rq1_model(
        development=development_a,
        evaluation=evaluation_a,
        model_name=
            "random_forest",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    development_b, evaluation_b = (
        make_sufficient_data(
            project_id=
                "project-b",
        )
    )

    run_rq1_model(
        development=development_b,
        evaluation=evaluation_b,
        model_name=
            "logistic_regression",
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    assert (
        len(
            platt_instances
        )
        == 3
    )

    assert (
        len(
            isotonic_instances
        )
        == 3
    )

    assert (
        platt_instances[0]
        is not
        platt_instances[1]
    )

    assert (
        platt_instances[0]
        is not
        platt_instances[2]
    )

    assert (
        isotonic_instances[0]
        is not
        isotonic_instances[1]
    )

    assert (
        isotonic_instances[0]
        is not
        isotonic_instances[2]
    )


def test_materialization_row_count_mismatch_is_rejected(
    monkeypatch,
):
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    (
        development,
        evaluation,
    ) = make_sufficient_data()

    original_materialize = (
        rq1_pipeline
        .materialize_calibration_partitions
    )

    def broken_materialize(
        data,
        split,
        build_column="build_index",
    ):
        (
            model_fit,
            calibration_data,
        ) = original_materialize(
            data,
            split,
            build_column=
                build_column,
        )

        duplicate = (
            calibration_data
            .iloc[[0]]
            .copy()
        )

        calibration_data = pd.concat(
            [
                calibration_data,
                duplicate,
            ],
            ignore_index=True,
        )

        return (
            model_fit,
            calibration_data,
        )

    monkeypatch.setattr(
        rq1_pipeline,
        "materialize_calibration_partitions",
        broken_materialize,
    )

    with pytest.raises(
        AssertionError,
        match=(
            "row coverage"
        ),
    ):
        run_sufficient_rq1_model(
            development=development,
            evaluation=evaluation,
            model_name=
                "logistic_regression",
            rq1_config=rq1,
            model_config=models,
            calibration_config=
                calibration,
            core_features_config=
                core,
        )
