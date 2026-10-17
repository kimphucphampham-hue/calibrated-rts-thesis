from pathlib import Path

import pandas as pd
import pytest

import src.tcp_ci.rq1_pipeline as rq1_pipeline

from src.tcp_ci.rq1_pipeline import (
    load_yaml,
    prediction_columns,
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
) -> pd.DataFrame:
    rows = []

    for build_index in builds:
        rows.append(
            {
                "project_id":
                    "project-a",

                "build_id":
                    f"build-{build_index}",

                "build_index":
                    build_index,

                "test_id":
                    (
                        f"test-"
                        f"{build_index % 3}"
                    ),

                "failed":
                    int(
                        build_index
                        in failure_builds
                    ),

                "failure_rate_history":
                    (
                        build_index
                        % 10
                    )
                    / 10.0,

                "recent_failure":
                    int(
                        build_index
                        % 7
                        == 0
                    ),

                "consecutive_failures":
                    (
                        build_index
                        % 3
                    ),

                "consecutive_passes":
                    (
                        build_index
                        % 5
                    ),

                "test_age_builds":
                    build_index
                    - 1,

                "time_since_last_failure_builds":
                    float(
                        (
                            build_index
                            % 11
                        )
                        + 1
                    ),

                "execution_time_mean_history_ms":
                    float(
                        100
                        + build_index
                    ),

                "changed_entity_count":
                    float(
                        build_index
                        % 6
                    ),

                "recent_failure_density_10":
                    (
                        build_index
                        % 5
                    )
                    / 10.0,

                "execution_time_trend_10":
                    float(
                        (
                            build_index
                            % 9
                        )
                        - 4
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_sufficient_data():
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
    )

    return (
        development,
        evaluation,
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ],
)
def test_sufficient_run_returns_all_probability_variants(
    model_name,
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

    (
        predictions,
        diagnostics,
    ) = run_sufficient_rq1_model(
        development=development,
        evaluation=evaluation,
        model_name=model_name,
        rq1_config=rq1,
        model_config=models,
        calibration_config=
            calibration,
        core_features_config=
            core,
    )

    assert list(
        predictions.columns
    ) == prediction_columns()

    assert (
        len(
            predictions
        )
        == len(
            evaluation
        )
    )

    for column in [
        "p_uncalibrated",
        "p_platt",
        "p_isotonic",
    ]:
        assert predictions[
            column
        ].notna().all()

        assert predictions[
            column
        ].between(
            0.0,
            1.0,
        ).all()

    assert (
        diagnostics[
            "status"
        ]
        == "SUFFICIENT"
    )

    assert (
        diagnostics[
            "model_fit_builds"
        ]
        == 104
    )

    assert (
        diagnostics[
            "calibration_builds"
        ]
        == 26
    )

    assert (
        diagnostics[
            "evaluation_builds"
        ]
        == 5
    )

    assert (
        diagnostics[
            "model_fit_build_max"
        ]
        <
        diagnostics[
            "calibration_build_min"
        ]
        <
        diagnostics[
            "evaluation_build_min"
        ]
    )


def test_outer_evaluation_is_never_used_for_fit(
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

    original_fit = (
        rq1_pipeline.fit_probability_model
    )

    original_probability = (
        rq1_pipeline.positive_probability
    )

    original_platt_builder = (
        rq1_pipeline.build_platt_calibrator
    )

    original_isotonic_builder = (
        rq1_pipeline.build_isotonic_calibrator
    )

    model_fit_builds = []

    probability_calls = []

    calibrator_fit_sizes = {
        "platt": [],
        "isotonic": [],
    }

    calibrator_transform_sizes = {
        "platt": [],
        "isotonic": [],
    }

    def spy_fit(
        model,
        X_train,
        y_train,
    ):
        model_fit_builds.extend(
            X_train[
                "build_index"
            ].tolist()
        )

        return original_fit(
            model,
            X_train,
            y_train,
        )

    def spy_probability(
        model,
        dataframe,
    ):
        probability_calls.append(
            tuple(
                dataframe[
                    "build_index"
                ].tolist()
            )
        )

        return original_probability(
            model,
            dataframe,
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
            calibrator_fit_sizes[
                self.name
            ].append(
                len(
                    probability
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
            calibrator_transform_sizes[
                self.name
            ].append(
                len(
                    probability
                )
            )

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
        "positive_probability",
        spy_probability,
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

    assert min(
        model_fit_builds
    ) == 1

    assert max(
        model_fit_builds
    ) == 104

    assert probability_calls == [
        tuple(
            range(
                105,
                131,
            )
        ),
        tuple(
            range(
                131,
                136,
            )
        ),
    ]

    assert (
        calibrator_fit_sizes[
            "platt"
        ]
        == [
            26,
        ]
    )

    assert (
        calibrator_fit_sizes[
            "isotonic"
        ]
        == [
            26,
        ]
    )

    assert (
        calibrator_transform_sizes[
            "platt"
        ]
        == [
            5,
        ]
    )

    assert (
        calibrator_transform_sizes[
            "isotonic"
        ]
        == [
            5,
        ]
    )


def test_overlapping_outer_build_is_rejected():
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
            130,
            136,
        ),
        failure_builds={
            132,
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "must be disjoint"
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


def test_insufficient_calibration_support_is_rejected():
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    development = make_partition(
        builds=range(
            1,
            131,
        ),
        failure_builds={
            10,
            20,
        },
    )

    evaluation = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            132,
        },
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "INSUFFICIENT_CALIBRATION_SUPPORT"
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


def test_general_run_uses_all_variants_when_support_is_sufficient():
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

    (
        predictions,
        diagnostics,
    ) = run_rq1_model(
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

    assert (
        diagnostics[
            "status"
        ]
        == "SUFFICIENT"
    )

    assert (
        diagnostics[
            "calibration_support_sufficient"
        ]
        is True
    )

    assert (
        diagnostics[
            "uncalibrated_available"
        ]
        is True
    )

    assert (
        diagnostics[
            "platt_available"
        ]
        is True
    )

    assert (
        diagnostics[
            "isotonic_available"
        ]
        is True
    )

    assert predictions[
        "p_uncalibrated"
    ].notna().all()

    assert predictions[
        "p_platt"
    ].notna().all()

    assert predictions[
        "p_isotonic"
    ].notna().all()


def test_general_run_keeps_raw_when_calibration_support_is_insufficient(
    monkeypatch,
):
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    development = make_partition(
        builds=range(
            1,
            131,
        ),
        failure_builds={
            10,
            20,
        },
    )

    evaluation = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            132,
        },
    )

    def fail_platt_builder(
        config,
    ):
        raise AssertionError(
            "Platt calibrator must not "
            "be built when support is "
            "insufficient."
        )

    def fail_isotonic_builder(
        config,
    ):
        raise AssertionError(
            "Isotonic calibrator must not "
            "be built when support is "
            "insufficient."
        )

    monkeypatch.setattr(
        rq1_pipeline,
        "build_platt_calibrator",
        fail_platt_builder,
    )

    monkeypatch.setattr(
        rq1_pipeline,
        "build_isotonic_calibrator",
        fail_isotonic_builder,
    )

    (
        predictions,
        diagnostics,
    ) = run_rq1_model(
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

    assert (
        diagnostics[
            "status"
        ]
        ==
        "INSUFFICIENT_CALIBRATION_SUPPORT"
    )

    assert (
        diagnostics[
            "calibration_support_sufficient"
        ]
        is False
    )

    assert (
        diagnostics[
            "uncalibrated_available"
        ]
        is True
    )

    assert (
        diagnostics[
            "platt_available"
        ]
        is False
    )

    assert (
        diagnostics[
            "isotonic_available"
        ]
        is False
    )

    assert predictions[
        "p_uncalibrated"
    ].notna().all()

    assert predictions[
        "p_uncalibrated"
    ].between(
        0.0,
        1.0,
    ).all()

    assert predictions[
        "p_platt"
    ].isna().all()

    assert predictions[
        "p_isotonic"
    ].isna().all()

    assert (
        diagnostics[
            "model_fit_builds"
        ]
        == 100
    )

    assert (
        diagnostics[
            "calibration_builds"
        ]
        == 30
    )

    assert (
        diagnostics[
            "expanded_by_builds"
        ]
        == 4
    )

    assert (
        diagnostics[
            "calibration_positive_rows"
        ]
        == 0
    )


def test_insufficient_general_run_fits_only_model_fit_partition(
    monkeypatch,
):
    (
        rq1,
        models,
        calibration,
        core,
    ) = load_configs()

    development = make_partition(
        builds=range(
            1,
            131,
        ),
        failure_builds={
            10,
            20,
        },
    )

    evaluation = make_partition(
        builds=range(
            131,
            136,
        ),
        failure_builds={
            132,
        },
    )

    original_fit = (
        rq1_pipeline.fit_probability_model
    )

    original_probability = (
        rq1_pipeline.positive_probability
    )

    fitted_builds = []
    probability_calls = []

    def spy_fit(
        model,
        X_train,
        y_train,
    ):
        fitted_builds.extend(
            X_train[
                "build_index"
            ].tolist()
        )

        return original_fit(
            model,
            X_train,
            y_train,
        )

    def spy_probability(
        model,
        dataframe,
    ):
        probability_calls.append(
            tuple(
                dataframe[
                    "build_index"
                ].tolist()
            )
        )

        return original_probability(
            model,
            dataframe,
        )

    monkeypatch.setattr(
        rq1_pipeline,
        "fit_probability_model",
        spy_fit,
    )

    monkeypatch.setattr(
        rq1_pipeline,
        "positive_probability",
        spy_probability,
    )

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

    assert fitted_builds == list(
        range(
            1,
            101,
        )
    )

    assert probability_calls == [
        tuple(
            range(
                131,
                136,
            )
        ),
    ]


def test_general_run_rejects_insufficient_policy_drift():
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

    changed = {
        **calibration,

        "rare_event_policy": {
            **calibration[
                "rare_event_policy"
            ],

            "insufficient_support_action": {
                **calibration[
                    "rare_event_policy"
                ][
                    "insufficient_support_action"
                ],

                "platt":
                    "run",
            },
        },
    }

    with pytest.raises(
        ValueError,
        match=(
            "frozen calibration policy"
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
                changed,
            core_features_config=
                core,
        )
