import numpy as np
import pandas as pd
import yaml

from src.tcp_ci.run_baseline import (
    build_logistic_pipeline,
    evaluate_probabilities,
    validate_feature_list,
)


FEATURES = [
    "failure_rate_history",
    "recent_failure",
    "consecutive_failures",
    "consecutive_passes",
    "test_age_builds",
    "time_since_last_failure_builds",
    "execution_time_mean_history_ms",
]


def load_config():
    with open(
        "configs/baseline_v1.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def make_training_data():
    return pd.DataFrame(
        {
            "failure_rate_history": [
                np.nan,
                0.0,
                0.5,
                1.0,
                0.25,
                0.75,
                0.1,
                0.9,
            ],
            "recent_failure": [
                np.nan,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
            ],
            "consecutive_failures": [
                0,
                0,
                1,
                2,
                0,
                1,
                0,
                3,
            ],
            "consecutive_passes": [
                0,
                2,
                0,
                0,
                3,
                0,
                2,
                0,
            ],
            "test_age_builds": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
            "time_since_last_failure_builds": [
                np.nan,
                np.nan,
                1,
                1,
                3,
                1,
                5,
                1,
            ],
            "execution_time_mean_history_ms": [
                np.nan,
                100,
                200,
                300,
                150,
                350,
                175,
                400,
            ],
        }
    )


def test_feature_allowlist_rejects_post_execution_fields():
    validate_feature_list(
        FEATURES
    )

    try:
        validate_feature_list(
            FEATURES
            + [
                "execution_time_ms",
            ]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "execution_time_ms should "
            "not be accepted as a feature."
        )


def test_logistic_pipeline_handles_missing_history():
    config = load_config()

    X = make_training_data()

    y = pd.Series(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    model = build_logistic_pipeline(
        FEATURES,
        config,
    )

    model.fit(
        X,
        y,
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    assert len(probabilities) == len(X)

    assert np.isfinite(
        probabilities
    ).all()

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()


def test_metrics_are_finite():
    y = pd.Series(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probability = np.array(
        [
            0.1,
            0.3,
            0.7,
            0.9,
        ]
    )

    metrics = evaluate_probabilities(
        y,
        probability,
    )

    for value in metrics.values():
        assert np.isfinite(value)


def test_config_does_not_enable_calibration_or_tuning():
    config = load_config()

    guards = config["guards"]

    assert (
        guards[
            "use_test_for_preprocessor_fit"
        ]
        is False
    )

    assert (
        guards[
            "use_test_for_model_fit"
        ]
        is False
    )

    assert (
        guards[
            "use_test_for_hyperparameter_tuning"
        ]
        is False
    )

    assert (
        guards[
            "calibration_enabled"
        ]
        is False
    )