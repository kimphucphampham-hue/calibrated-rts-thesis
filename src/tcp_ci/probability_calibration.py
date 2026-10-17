from typing import Optional

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def validate_probability(
    probability,
) -> np.ndarray:
    values = np.asarray(
        probability,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Probability input must be "
            "one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "Probability input must not "
            "be empty."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Probability input must contain "
            "only finite values."
        )

    if (
        (values < 0.0).any()
        or
        (values > 1.0).any()
    ):
        raise ValueError(
            "Probability values must lie "
            "within [0, 1]."
        )

    return values


def validate_binary_target(
    target,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    values = np.asarray(
        target
    )

    if values.ndim != 1:
        raise ValueError(
            "Calibration target must be "
            "one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "Calibration target must not "
            "be empty."
        )

    try:
        numeric = values.astype(
            float
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Calibration target must be "
            "binary 0/1."
        ) from exc

    if not np.isfinite(
        numeric
    ).all():
        raise ValueError(
            "Calibration target must contain "
            "only finite values."
        )

    unique = set(
        numeric.tolist()
    )

    if not unique.issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            "Calibration target must be "
            "binary 0/1."
        )

    if (
        expected_length is not None
        and len(numeric)
        != expected_length
    ):
        raise ValueError(
            "Probability and target lengths "
            "must match."
        )

    return numeric.astype(int)


def clipped_probability_logit(
    probability,
    epsilon: float,
) -> np.ndarray:
    values = validate_probability(
        probability
    )

    epsilon = float(
        epsilon
    )

    if not (
        0.0
        < epsilon
        < 0.5
    ):
        raise ValueError(
            "epsilon must satisfy "
            "0 < epsilon < 0.5."
        )

    clipped = np.clip(
        values,
        epsilon,
        1.0 - epsilon,
    )

    return np.log(
        clipped
        /
        (
            1.0
            - clipped
        )
    )


class PlattCalibrator:
    def __init__(
        self,
        epsilon: float,
    ) -> None:
        epsilon = float(
            epsilon
        )

        if not (
            0.0
            < epsilon
            < 0.5
        ):
            raise ValueError(
                "epsilon must satisfy "
                "0 < epsilon < 0.5."
            )

        self.epsilon = epsilon

        self._model = None

        self.coefficient_ = None
        self.intercept_ = None

    def fit(
        self,
        base_probability,
        target,
    ):
        probability = (
            validate_probability(
                base_probability
            )
        )

        y = validate_binary_target(
            target,
            expected_length=len(
                probability
            ),
        )

        if np.unique(
            y
        ).size < 2:
            raise ValueError(
                "Platt calibration requires "
                "both target classes."
            )

        logit = (
            clipped_probability_logit(
                probability,
                epsilon=self.epsilon,
            )
        )

        # No regularization:
        # fit a and b by minimizing binary
        # log loss on calibration data only.
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=2000,
            class_weight=None,
        )

        model.fit(
            logit.reshape(
                -1,
                1,
            ),
            y,
        )

        self._model = model

        self.coefficient_ = float(
            model.coef_[
                0,
                0,
            ]
        )

        self.intercept_ = float(
            model.intercept_[
                0
            ]
        )

        return self

    def transform(
        self,
        base_probability,
    ) -> np.ndarray:
        if (
            self.coefficient_
            is None
            or self.intercept_
            is None
        ):
            raise RuntimeError(
                "PlattCalibrator must be "
                "fitted before transform."
            )

        logit = (
            clipped_probability_logit(
                base_probability,
                epsilon=self.epsilon,
            )
        )

        calibrated = expit(
            (
                self.coefficient_
                * logit
            )
            + self.intercept_
        )

        calibrated = np.asarray(
            calibrated,
            dtype=float,
        )

        if not np.isfinite(
            calibrated
        ).all():
            raise RuntimeError(
                "Calibrated probabilities "
                "contain non-finite values."
            )

        if (
            (calibrated < 0.0).any()
            or
            (calibrated > 1.0).any()
        ):
            raise RuntimeError(
                "Calibrated probabilities "
                "must lie within [0, 1]."
            )

        return calibrated


def build_platt_calibrator(
    calibration_config: dict,
) -> PlattCalibrator:
    platt_config = (
        calibration_config[
            "platt"
        ]
    )

    return PlattCalibrator(
        epsilon=float(
            platt_config[
                "epsilon"
            ]
        )
    )

class IsotonicCalibrator:
    def __init__(
        self,
        increasing: bool = True,
        out_of_bounds: str = "clip",
    ) -> None:
        if increasing is not True:
            raise ValueError(
                "RQ1 v1 requires increasing "
                "isotonic calibration."
            )

        if out_of_bounds != "clip":
            raise ValueError(
                "RQ1 v1 requires "
                "out_of_bounds='clip'."
            )

        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

        self._model = None

    def fit(
        self,
        base_probability,
        target,
    ):
        probability = (
            validate_probability(
                base_probability
            )
        )

        y = validate_binary_target(
            target,
            expected_length=len(
                probability
            ),
        )

        if np.unique(
            y
        ).size < 2:
            raise ValueError(
                "Isotonic calibration requires "
                "both target classes."
            )

        model = IsotonicRegression(
            increasing=self.increasing,
            out_of_bounds=
                self.out_of_bounds,
        )

        model.fit(
            probability,
            y,
        )

        self._model = model

        return self

    def transform(
        self,
        base_probability,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "IsotonicCalibrator must be "
                "fitted before transform."
            )

        probability = (
            validate_probability(
                base_probability
            )
        )

        calibrated = (
            self._model.transform(
                probability
            )
        )

        calibrated = np.asarray(
            calibrated,
            dtype=float,
        )

        if not np.isfinite(
            calibrated
        ).all():
            raise RuntimeError(
                "Calibrated probabilities "
                "contain non-finite values."
            )

        if (
            (calibrated < 0.0).any()
            or
            (calibrated > 1.0).any()
        ):
            raise RuntimeError(
                "Calibrated probabilities "
                "must lie within [0, 1]."
            )

        return calibrated


def build_isotonic_calibrator(
    calibration_config: dict,
) -> IsotonicCalibrator:
    isotonic_config = (
        calibration_config[
            "isotonic"
        ]
    )

    direction = str(
        isotonic_config[
            "monotonic_direction"
        ]
    )

    if direction != "increasing":
        raise ValueError(
            "RQ1 v1 requires increasing "
            "isotonic calibration."
        )

    return IsotonicCalibrator(
        increasing=True,
        out_of_bounds=str(
            isotonic_config[
                "out_of_bounds"
            ]
        ),
    )
