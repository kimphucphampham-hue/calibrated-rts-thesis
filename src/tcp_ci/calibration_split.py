from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CalibrationSupport:
    positive_rows: int
    negative_rows: int
    failing_builds: int
    passing_builds: int


@dataclass(frozen=True)
class CalibrationSplit:
    model_fit_builds: tuple
    calibration_builds: tuple

    initial_model_fit_builds: int
    expanded_by_builds: int

    support: CalibrationSupport

    sufficient: bool
    status: str


def validate_development_data(
    data: pd.DataFrame,
    build_column: str = "build_index",
    target_column: str = "failed",
) -> None:
    required = {
        build_column,
        target_column,
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Development data is missing "
            f"required columns: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "Development data must not "
            "be empty."
        )

    if data[
        build_column
    ].isna().any():
        raise ValueError(
            f"{build_column} must not "
            "contain missing values."
        )

    target = pd.to_numeric(
        data[target_column],
        errors="raise",
    )

    values = set(
        target.unique()
    )

    if not values.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"{target_column} must be "
            "binary 0/1."
        )


def calibration_support(
    data: pd.DataFrame,
    build_column: str = "build_index",
    target_column: str = "failed",
) -> CalibrationSupport:
    validate_development_data(
        data,
        build_column=build_column,
        target_column=target_column,
    )

    target = pd.to_numeric(
        data[target_column],
        errors="raise",
    ).astype(int)

    positive_rows = int(
        target.sum()
    )

    negative_rows = int(
        len(data)
        - positive_rows
    )

    working = pd.DataFrame(
        {
            build_column:
                data[
                    build_column
                ].to_numpy(),

            target_column:
                target.to_numpy(),
        }
    )

    by_build = (
        working.groupby(
            build_column
        )[target_column]
        .max()
    )

    failing_builds = int(
        (by_build == 1).sum()
    )

    passing_builds = int(
        (by_build == 0).sum()
    )

    return CalibrationSupport(
        positive_rows=
            positive_rows,
        negative_rows=
            negative_rows,
        failing_builds=
            failing_builds,
        passing_builds=
            passing_builds,
    )


def support_is_sufficient(
    support: CalibrationSupport,
    minimum_support: dict,
) -> bool:
    return (
        support.positive_rows
        >= int(
            minimum_support[
                "positive_rows"
            ]
        )
        and
        support.negative_rows
        >= int(
            minimum_support[
                "negative_rows"
            ]
        )
        and
        support.failing_builds
        >= int(
            minimum_support[
                "failing_builds"
            ]
        )
        and
        support.passing_builds
        >= int(
            minimum_support[
                "passing_builds"
            ]
        )
    )


def split_development_for_calibration(
    data: pd.DataFrame,
    rq1_config: dict,
    calibration_config: dict,
    build_column: str = "build_index",
    target_column: str = "failed",
) -> CalibrationSplit:
    validate_development_data(
        data,
        build_column=build_column,
        target_column=target_column,
    )

    temporal = rq1_config[
        "temporal_design"
    ]

    inner = temporal[
        "inner_initial"
    ]

    expansion = temporal[
        "calibration_expansion"
    ]

    policy = calibration_config[
        "rare_event_policy"
    ]

    if (
        inner["rounding"]
        != "floor"
    ):
        raise ValueError(
            "Only floor rounding is "
            "supported."
        )

    if (
        expansion["enabled"]
        is not True
    ):
        raise ValueError(
            "Calibration expansion must "
            "be enabled for RQ1 v1."
        )

    if (
        expansion["direction"]
        != "backward_in_time"
    ):
        raise ValueError(
            "RQ1 v1 requires backward "
            "calibration expansion."
        )

    if (
        expansion["step_unit"]
        != "one_build"
    ):
        raise ValueError(
            "RQ1 v1 requires one-build "
            "expansion steps."
        )

    if (
        expansion[
            "evaluation_builds_may_be_used"
        ]
        is not False
    ):
        raise ValueError(
            "Evaluation builds must never "
            "be used for calibration."
        )

    model_fit_fraction = float(
        inner[
            "model_fit_fraction_of_development"
        ]
    )

    calibration_fraction = float(
        inner[
            "calibration_fraction_of_development"
        ]
    )

    if abs(
        (
            model_fit_fraction
            + calibration_fraction
        )
        - 1.0
    ) > 1e-12:
        raise ValueError(
            "Inner model-fit and "
            "calibration fractions must "
            "sum to 1."
        )

    minimum_support = expansion[
        "minimum_support"
    ]

    if (
        minimum_support
        != policy[
            "minimum_support"
        ]
    ):
        raise ValueError(
            "RQ1 and calibration "
            "minimum-support rules differ."
        )

    minimum_model_fit_builds = int(
        expansion[
            "minimum_model_fit_builds"
        ]
    )

    if (
        minimum_model_fit_builds
        != int(
            policy[
                "minimum_model_fit_builds"
            ]
        )
    ):
        raise ValueError(
            "RQ1 and calibration "
            "model-fit floors differ."
        )

    builds = tuple(
        sorted(
            data[
                build_column
            ]
            .unique()
        )
    )

    build_count = len(
        builds
    )

    initial_boundary = int(
        build_count
        * model_fit_fraction
    )

    if (
        initial_boundary
        < minimum_model_fit_builds
    ):
        raise ValueError(
            "Initial model-fit partition "
            "is below the configured "
            "minimum_model_fit_builds."
        )

    if (
        initial_boundary
        >= build_count
    ):
        raise ValueError(
            "Initial calibration "
            "partition is empty."
        )

    boundary = initial_boundary

    skip_status = str(
        policy["skip_status"]
    )

    while True:
        calibration_builds = (
            builds[boundary:]
        )

        calibration_data = data[
            data[
                build_column
            ].isin(
                calibration_builds
            )
        ]

        support = (
            calibration_support(
                calibration_data,
                build_column=
                    build_column,
                target_column=
                    target_column,
            )
        )

        if support_is_sufficient(
            support,
            minimum_support,
        ):
            sufficient = True
            status = "SUFFICIENT"
            break

        next_boundary = (
            boundary - 1
        )

        if (
            next_boundary
            < minimum_model_fit_builds
        ):
            sufficient = False
            status = skip_status
            break

        boundary = next_boundary

    model_fit_builds = (
        builds[:boundary]
    )

    calibration_builds = (
        builds[boundary:]
    )

    if not model_fit_builds:
        raise ValueError(
            "Model-fit partition is empty."
        )

    if not calibration_builds:
        raise ValueError(
            "Calibration partition "
            "is empty."
        )

    if not (
        max(model_fit_builds)
        <
        min(calibration_builds)
    ):
        raise ValueError(
            "Temporal ordering violation: "
            "model-fit builds must strictly "
            "precede calibration builds."
        )

    if not set(
        model_fit_builds
    ).isdisjoint(
        calibration_builds
    ):
        raise ValueError(
            "A build appears in both "
            "model-fit and calibration "
            "partitions."
        )

    return CalibrationSplit(
        model_fit_builds=
            model_fit_builds,
        calibration_builds=
            calibration_builds,

        initial_model_fit_builds=
            initial_boundary,

        expanded_by_builds=
            initial_boundary
            - boundary,

        support=support,

        sufficient=sufficient,
        status=status,
    )


def materialize_calibration_partitions(
    data: pd.DataFrame,
    split: CalibrationSplit,
    build_column: str = "build_index",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    model_fit = data[
        data[
            build_column
        ].isin(
            split.model_fit_builds
        )
    ].copy()

    calibration = data[
        data[
            build_column
        ].isin(
            split.calibration_builds
        )
    ].copy()

    return (
        model_fit,
        calibration,
    )
