from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
import yaml

from src.tcp_ci.probability_calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    build_isotonic_calibrator,
    build_platt_calibrator,
    clipped_probability_logit,
    validate_probability,
)


CONFIG_DIR = Path("configs")


def load_calibration_config() -> dict:
    path = (
        CONFIG_DIR
        / "calibration_v1.yaml"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def make_calibration_data():
    probability = np.array(
        [
            0.01,
            0.03,
            0.08,
            0.12,
            0.20,
            0.30,
            0.45,
            0.55,
            0.70,
            0.80,
            0.90,
            0.97,
        ],
        dtype=float,
    )

    target = np.array(
        [
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=int,
    )

    return (
        probability,
        target,
    )


def test_logit_transform_matches_frozen_definition():
    epsilon = 0.000001

    probability = np.array(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        dtype=float,
    )

    result = (
        clipped_probability_logit(
            probability,
            epsilon=epsilon,
        )
    )

    clipped = np.clip(
        probability,
        epsilon,
        1.0 - epsilon,
    )

    expected = np.log(
        clipped
        /
        (
            1.0
            - clipped
        )
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=0.0,
        atol=0.0,
    )

    assert np.isfinite(
        result
    ).all()

    assert result[2] == 0.0


def test_platt_builder_uses_frozen_epsilon():
    config = (
        load_calibration_config()
    )

    calibrator = (
        build_platt_calibrator(
            config
        )
    )

    assert (
        calibrator.epsilon
        == pytest.approx(
            0.000001
        )
    )


def test_platt_calibrator_returns_valid_probabilities():
    config = (
        load_calibration_config()
    )

    probability, target = (
        make_calibration_data()
    )

    calibrator = (
        build_platt_calibrator(
            config
        )
    )

    calibrator.fit(
        probability,
        target,
    )

    calibrated = (
        calibrator.transform(
            probability
        )
    )

    assert (
        len(calibrated)
        == len(probability)
    )

    assert np.isfinite(
        calibrated
    ).all()

    assert (
        calibrated >= 0.0
    ).all()

    assert (
        calibrated <= 1.0
    ).all()


def test_platt_transform_matches_fitted_formula():
    config = (
        load_calibration_config()
    )

    probability, target = (
        make_calibration_data()
    )

    calibrator = (
        build_platt_calibrator(
            config
        )
    )

    calibrator.fit(
        probability,
        target,
    )

    transformed = (
        calibrator.transform(
            probability
        )
    )

    logit = (
        clipped_probability_logit(
            probability,
            epsilon=
                calibrator.epsilon,
        )
    )

    expected = expit(
        (
            calibrator.coefficient_
            * logit
        )
        + calibrator.intercept_
    )

    np.testing.assert_allclose(
        transformed,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_platt_calibrator_is_deterministic():
    probability, target = (
        make_calibration_data()
    )

    calibrator_1 = (
        PlattCalibrator(
            epsilon=0.000001
        )
    )

    calibrator_2 = (
        PlattCalibrator(
            epsilon=0.000001
        )
    )

    calibrator_1.fit(
        probability,
        target,
    )

    calibrator_2.fit(
        probability,
        target,
    )

    result_1 = (
        calibrator_1.transform(
            probability
        )
    )

    result_2 = (
        calibrator_2.transform(
            probability
        )
    )

    assert (
        calibrator_1.coefficient_
        ==
        calibrator_2.coefficient_
    )

    assert (
        calibrator_1.intercept_
        ==
        calibrator_2.intercept_
    )

    np.testing.assert_allclose(
        result_1,
        result_2,
        rtol=0.0,
        atol=0.0,
    )


def test_platt_requires_both_target_classes():
    probability = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        dtype=float,
    )

    target = np.zeros(
        4,
        dtype=int,
    )

    calibrator = (
        PlattCalibrator(
            epsilon=0.000001
        )
    )

    with pytest.raises(
        ValueError,
        match="both target classes",
    ):
        calibrator.fit(
            probability,
            target,
        )


def test_probability_validation_rejects_invalid_values():
    invalid_inputs = [
        np.array(
            [-0.01, 0.5]
        ),
        np.array(
            [0.5, 1.01]
        ),
        np.array(
            [0.2, np.nan]
        ),
        np.array(
            [0.2, np.inf]
        ),
    ]

    for probability in invalid_inputs:
        with pytest.raises(
            ValueError
        ):
            validate_probability(
                probability
            )


def test_isotonic_builder_uses_frozen_config():
    config = (
        load_calibration_config()
    )

    calibrator = (
        build_isotonic_calibrator(
            config
        )
    )

    assert (
        calibrator.increasing
        is True
    )

    assert (
        calibrator.out_of_bounds
        == "clip"
    )


def test_isotonic_calibrator_returns_valid_probabilities():
    config = (
        load_calibration_config()
    )

    probability, target = (
        make_calibration_data()
    )

    calibrator = (
        build_isotonic_calibrator(
            config
        )
    )

    calibrator.fit(
        probability,
        target,
    )

    calibrated = (
        calibrator.transform(
            probability
        )
    )

    assert (
        len(calibrated)
        == len(probability)
    )

    assert np.isfinite(
        calibrated
    ).all()

    assert (
        calibrated >= 0.0
    ).all()

    assert (
        calibrated <= 1.0
    ).all()


def test_isotonic_output_is_non_decreasing():
    config = (
        load_calibration_config()
    )

    probability, target = (
        make_calibration_data()
    )

    calibrator = (
        build_isotonic_calibrator(
            config
        )
    )

    calibrator.fit(
        probability,
        target,
    )

    query = np.linspace(
        0.0,
        1.0,
        101,
    )

    calibrated = (
        calibrator.transform(
            query
        )
    )

    assert (
        np.diff(
            calibrated
        )
        >= -1e-12
    ).all()


def test_isotonic_out_of_bounds_is_clipped():
    calibrator = (
        IsotonicCalibrator(
            increasing=True,
            out_of_bounds="clip",
        )
    )

    probability = np.array(
        [
            0.20,
            0.30,
            0.40,
            0.60,
            0.70,
            0.80,
        ],
        dtype=float,
    )

    target = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=int,
    )

    calibrator.fit(
        probability,
        target,
    )

    boundary = (
        calibrator.transform(
            np.array(
                [
                    0.20,
                    0.80,
                ],
                dtype=float,
            )
        )
    )

    outside = (
        calibrator.transform(
            np.array(
                [
                    0.0,
                    1.0,
                ],
                dtype=float,
            )
        )
    )

    np.testing.assert_allclose(
        outside,
        boundary,
        rtol=0.0,
        atol=0.0,
    )


def test_isotonic_requires_both_target_classes():
    probability = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        dtype=float,
    )

    target = np.zeros(
        4,
        dtype=int,
    )

    calibrator = (
        IsotonicCalibrator()
    )

    with pytest.raises(
        ValueError,
        match="both target classes",
    ):
        calibrator.fit(
            probability,
            target,
        )


def test_isotonic_calibrator_is_deterministic():
    probability, target = (
        make_calibration_data()
    )

    calibrator_1 = (
        IsotonicCalibrator()
    )

    calibrator_2 = (
        IsotonicCalibrator()
    )

    calibrator_1.fit(
        probability,
        target,
    )

    calibrator_2.fit(
        probability,
        target,
    )

    query = np.linspace(
        0.0,
        1.0,
        101,
    )

    result_1 = (
        calibrator_1.transform(
            query
        )
    )

    result_2 = (
        calibrator_2.transform(
            query
        )
    )

    np.testing.assert_allclose(
        result_1,
        result_2,
        rtol=0.0,
        atol=0.0,
    )
