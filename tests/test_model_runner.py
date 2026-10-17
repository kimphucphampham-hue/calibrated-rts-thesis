import numpy as np
import pandas as pd
import yaml

from src.tcp_ci.model_runner import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    fit_probability_model,
    positive_probability,
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


def load_rq1_config():
    with open(
        "configs/rq1_models_v1.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def make_data():
    X = pd.DataFrame(
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

    return X, y


def test_feature_validation_rejects_forbidden_field():
    try:
        validate_feature_list(
            FEATURES
            + ["execution_time_ms"]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Forbidden field was accepted."
        )


def test_feature_validation_rejects_duplicates():
    try:
        validate_feature_list(
            FEATURES
            + [FEATURES[0]]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Duplicate feature was accepted."
        )


def test_logistic_runner_returns_valid_probabilities():
    config = load_config()
    X, y = make_data()

    model = build_logistic_pipeline(
        FEATURES,
        config,
    )

    fitted = fit_probability_model(
        model,
        X,
        y,
    )

    probability = positive_probability(
        fitted,
        X,
    )

    assert len(probability) == len(X)

    assert np.isfinite(
        probability
    ).all()

    assert (
        probability >= 0.0
    ).all()

    assert (
        probability <= 1.0
    ).all()


def test_logistic_parameters_match_frozen_baseline():
    config = load_config()

    model = build_logistic_pipeline(
        FEATURES,
        config,
    )

    estimator = model["model"]

    assert estimator.solver == "lbfgs"
    assert estimator.C == 1.0
    assert estimator.max_iter == 2000
    assert estimator.class_weight is None


def test_prediction_does_not_refit_preprocessor():
    config = load_config()
    X, y = make_data()

    model = build_logistic_pipeline(
        FEATURES,
        config,
    )

    fit_probability_model(
        model,
        X,
        y,
    )

    imputer = (
        model[
            "preprocess"
        ]
        .named_transformers_[
            "numeric"
        ][
            "imputer"
        ]
    )

    before = (
        imputer.statistics_.copy()
    )

    X_eval = X.copy()

    X_eval[
        "execution_time_mean_history_ms"
    ] = 999999.0

    positive_probability(
        model,
        X_eval,
    )

    after = (
        imputer.statistics_.copy()
    )

    np.testing.assert_array_equal(
        before,
        after,
    )


def test_random_forest_runner_returns_valid_probabilities():
    config = load_rq1_config()
    X, y = make_data()

    model = build_random_forest_pipeline(
        FEATURES,
        config,
    )

    fitted = fit_probability_model(
        model,
        X,
        y,
    )

    probability = positive_probability(
        fitted,
        X,
    )

    assert len(probability) == len(X)

    assert np.isfinite(
        probability
    ).all()

    assert (
        probability >= 0.0
    ).all()

    assert (
        probability <= 1.0
    ).all()


def test_random_forest_parameters_match_frozen_config():
    config = load_rq1_config()

    model = build_random_forest_pipeline(
        FEATURES,
        config,
    )

    estimator = model["model"]

    assert estimator.n_estimators == 100
    assert estimator.criterion == "gini"
    assert estimator.max_depth is None
    assert estimator.min_samples_split == 2
    assert estimator.min_samples_leaf == 1
    assert estimator.max_features == "sqrt"
    assert estimator.bootstrap is True
    assert estimator.class_weight is None
    assert estimator.random_state == 42
    assert estimator.n_jobs == -1
