from pathlib import Path

import pandas as pd
import yaml

from src.tcp_ci.calibration_split import (
    materialize_calibration_partitions,
    split_development_for_calibration,
)

from src.tcp_ci.model_runner import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    fit_probability_model,
    positive_probability,
    validate_feature_list,
)

from src.tcp_ci.probability_calibration import (
    build_isotonic_calibrator,
    build_platt_calibrator,
)


RQ1_MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)


RQ1_CALIBRATION_VARIANTS = (
    "uncalibrated",
    "platt",
    "isotonic",
)


PREDICTION_ID_COLUMNS = (
    "project_id",
    "build_id",
    "build_index",
    "test_id",
    "failed",
)


PROBABILITY_COLUMNS = {
    "uncalibrated":
        "p_uncalibrated",

    "platt":
        "p_platt",

    "isotonic":
        "p_isotonic",
}


MODEL_BUILDERS = {
    "logistic_regression":
        build_logistic_pipeline,

    "random_forest":
        build_random_forest_pipeline,

    "xgboost":
        build_xgboost_pipeline,
}


def load_yaml(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Configuration file must "
            "contain a YAML mapping."
        )

    return config


def rq1_feature_columns(
    rq1_config: dict,
    core_features_config: dict,
) -> list[str]:
    feature_set_name = (
        rq1_config[
            "feature_set"
        ][
            "name"
        ]
    )

    primary_feature_set = (
        core_features_config[
            "primary_feature_set"
        ]
    )

    if (
        feature_set_name
        != primary_feature_set
    ):
        raise ValueError(
            "RQ1 feature set does not "
            "match the frozen CORE "
            "primary feature set."
        )

    feature_sets = (
        core_features_config[
            "feature_sets"
        ]
    )

    if (
        feature_set_name
        not in feature_sets
    ):
        raise ValueError(
            "RQ1 feature set is not "
            "defined in CORE features."
        )

    features = list(
        feature_sets[
            feature_set_name
        ][
            "features"
        ]
    )

    validate_feature_list(
        features
    )

    return features


def validate_rq1_integration_contract(
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> list[str]:
    models = list(
        rq1_config[
            "models"
        ]
    )

    if models != list(
        RQ1_MODEL_NAMES
    ):
        raise ValueError(
            "RQ1 model list does not "
            "match the frozen integration "
            "contract."
        )

    configured_models = list(
        model_config[
            "models"
        ].keys()
    )

    if (
        configured_models
        != models
    ):
        raise ValueError(
            "RQ1 model configuration "
            "does not match the frozen "
            "model list."
        )

    variants = list(
        rq1_config[
            "calibration_variants"
        ]
    )

    if variants != list(
        RQ1_CALIBRATION_VARIANTS
    ):
        raise ValueError(
            "RQ1 calibration variants "
            "do not match the frozen "
            "integration contract."
        )

    if list(
        calibration_config[
            "methods"
        ]
    ) != variants:
        raise ValueError(
            "Calibration configuration "
            "does not match RQ1 variants."
        )

    feature_set_name = (
        rq1_config[
            "feature_set"
        ][
            "name"
        ]
    )

    if (
        model_config[
            "feature_config"
        ][
            "primary_feature_set"
        ]
        != feature_set_name
    ):
        raise ValueError(
            "Model configuration and "
            "RQ1 feature set differ."
        )

    if (
        rq1_config[
            "preprocessing"
        ][
            "fit_period"
        ]
        != "model_fit_only"
    ):
        raise ValueError(
            "RQ1 preprocessing must be "
            "fitted on model-fit only."
        )

    if (
        rq1_config[
            "preprocessing"
        ][
            "calibration_period_may_fit_preprocessing"
        ]
        is not False
    ):
        raise ValueError(
            "Calibration rows must not "
            "fit preprocessing."
        )

    if (
        rq1_config[
            "preprocessing"
        ][
            "evaluation_period_may_fit_preprocessing"
        ]
        is not False
    ):
        raise ValueError(
            "Evaluation rows must not "
            "fit preprocessing."
        )

    if (
        rq1_config[
            "pilot_retraining"
        ][
            "enabled"
        ]
        is not False
    ):
        raise ValueError(
            "RQ1 pilot retraining must "
            "remain disabled."
        )

    if (
        rq1_config[
            "threshold_dependent_classification_metrics"
        ][
            "enabled"
        ]
        is not False
    ):
        raise ValueError(
            "Threshold-dependent metrics "
            "must remain disabled for "
            "RQ1 v1."
        )

    expected_conditions = int(
        rq1_config[
            "reporting"
        ][
            "expected_conditions_per_project"
        ][
            "maximum_conditions"
        ]
    )

    actual_conditions = (
        len(models)
        * len(variants)
    )

    if (
        expected_conditions
        != actual_conditions
    ):
        raise ValueError(
            "RQ1 expected condition count "
            "does not match models x "
            "calibration variants."
        )

    if expected_conditions != 9:
        raise ValueError(
            "RQ1 v1 requires exactly "
            "nine conditions per project."
        )

    if (
        model_config[
            "target"
        ][
            "field"
        ]
        != "failed"
    ):
        raise ValueError(
            "RQ1 v1 target must be "
            "'failed'."
        )

    features = (
        rq1_feature_columns(
            rq1_config=
                rq1_config,
            core_features_config=
                core_features_config,
        )
    )

    if len(features) != 11:
        raise ValueError(
            "RQ1 v1 full CORE feature "
            "set must contain 11 "
            "features."
        )

    return features


def build_rq1_model(
    model_name: str,
    feature_columns: list[str],
    model_config: dict,
):
    if (
        model_name
        not in MODEL_BUILDERS
    ):
        raise ValueError(
            f"Unsupported RQ1 model: "
            f"{model_name}"
        )

    builder = MODEL_BUILDERS[
        model_name
    ]

    return builder(
        feature_columns,
        model_config,
    )


def prediction_columns() -> list[str]:
    return (
        list(
            PREDICTION_ID_COLUMNS
        )
        + [
            PROBABILITY_COLUMNS[
                variant
            ]
            for variant
            in RQ1_CALIBRATION_VARIANTS
        ]
    )


def validate_rq1_temporal_partitions(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "failed",
) -> None:
    required_columns = (
        set(
            PREDICTION_ID_COLUMNS
        )
        | set(
            feature_columns
        )
        | {
            target_column,
        }
    )

    for (
        partition_name,
        data,
    ) in [
        (
            "development",
            development,
        ),
        (
            "evaluation",
            evaluation,
        ),
    ]:
        missing = (
            required_columns
            - set(
                data.columns
            )
        )

        if missing:
            raise ValueError(
                f"{partition_name} data "
                "is missing required "
                f"columns: {sorted(missing)}"
            )

        if data.empty:
            raise ValueError(
                f"{partition_name} data "
                "must not be empty."
            )

        if data[
            "build_index"
        ].isna().any():
            raise ValueError(
                f"{partition_name} "
                "build_index must not "
                "contain missing values."
            )

        target = pd.to_numeric(
            data[
                target_column
            ],
            errors="raise",
        )

        target_values = set(
            target.unique()
        )

        if not target_values.issubset(
            {
                0,
                1,
            }
        ):
            raise ValueError(
                f"{partition_name} target "
                "must be binary 0/1."
            )

        project_count = int(
            data[
                "project_id"
            ].nunique(
                dropna=False
            )
        )

        if project_count != 1:
            raise ValueError(
                f"{partition_name} must "
                "contain exactly one "
                "project."
            )

    development_project = (
        development[
            "project_id"
        ].iloc[0]
    )

    evaluation_project = (
        evaluation[
            "project_id"
        ].iloc[0]
    )

    if (
        development_project
        != evaluation_project
    ):
        raise ValueError(
            "Development and evaluation "
            "must belong to the same "
            "project."
        )

    development_builds = set(
        development[
            "build_index"
        ].unique()
    )

    evaluation_builds = set(
        evaluation[
            "build_index"
        ].unique()
    )

    if not development_builds.isdisjoint(
        evaluation_builds
    ):
        raise ValueError(
            "Development and evaluation "
            "builds must be disjoint."
        )

    if not (
        max(
            development_builds
        )
        <
        min(
            evaluation_builds
        )
    ):
        raise ValueError(
            "All development builds must "
            "strictly precede all "
            "evaluation builds."
        )


def run_sufficient_rq1_model(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    model_name: str,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    feature_columns = (
        validate_rq1_integration_contract(
            rq1_config=rq1_config,
            model_config=model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )
    )

    target_column = (
        model_config[
            "target"
        ][
            "field"
        ]
    )

    validate_rq1_temporal_partitions(
        development=development,
        evaluation=evaluation,
        feature_columns=
            feature_columns,
        target_column=
            target_column,
    )

    split = (
        split_development_for_calibration(
            development,
            rq1_config=
                rq1_config,
            calibration_config=
                calibration_config,
            build_column=
                "build_index",
            target_column=
                target_column,
        )
    )

    if not split.sufficient:
        raise RuntimeError(
            "run_sufficient_rq1_model "
            "requires sufficient "
            "calibration support; "
            f"received status="
            f"{split.status}"
        )

    (
        model_fit,
        calibration,
    ) = (
        materialize_calibration_partitions(
            development,
            split,
            build_column=
                "build_index",
        )
    )

    validate_materialized_calibration_coverage(
        development=development,
        model_fit=model_fit,
        calibration=calibration,
        split=split,
        build_column="build_index",
    )

    model = build_rq1_model(
        model_name=model_name,
        feature_columns=
            feature_columns,
        model_config=
            model_config,
    )

    fitted_model = (
        fit_probability_model(
            model,
            model_fit[
                feature_columns
            ],
            model_fit[
                target_column
            ],
        )
    )

    calibration_probability = (
        positive_probability(
            fitted_model,
            calibration[
                feature_columns
            ],
        )
    )

    evaluation_probability = (
        positive_probability(
            fitted_model,
            evaluation[
                feature_columns
            ],
        )
    )

    calibration_target = (
        calibration[
            target_column
        ]
    )

    platt = (
        build_platt_calibrator(
            calibration_config
        )
    )

    platt.fit(
        calibration_probability,
        calibration_target,
    )

    isotonic = (
        build_isotonic_calibrator(
            calibration_config
        )
    )

    isotonic.fit(
        calibration_probability,
        calibration_target,
    )

    platt_probability = (
        platt.transform(
            evaluation_probability
        )
    )

    isotonic_probability = (
        isotonic.transform(
            evaluation_probability
        )
    )

    predictions = evaluation[
        list(
            PREDICTION_ID_COLUMNS
        )
    ].copy()

    predictions[
        PROBABILITY_COLUMNS[
            "uncalibrated"
        ]
    ] = evaluation_probability

    predictions[
        PROBABILITY_COLUMNS[
            "platt"
        ]
    ] = platt_probability

    predictions[
        PROBABILITY_COLUMNS[
            "isotonic"
        ]
    ] = isotonic_probability

    predictions = predictions[
        prediction_columns()
    ]

    for column in (
        PROBABILITY_COLUMNS.values()
    ):
        if not predictions[
            column
        ].between(
            0.0,
            1.0,
        ).all():
            raise AssertionError(
                f"Invalid probability "
                f"in {column}."
            )

    diagnostics = {
        "model":
            model_name,

        "status":
            split.status,

        "model_fit_rows":
            int(
                len(
                    model_fit
                )
            ),

        "calibration_rows":
            int(
                len(
                    calibration
                )
            ),

        "evaluation_rows":
            int(
                len(
                    evaluation
                )
            ),

        "model_fit_builds":
            int(
                len(
                    split.model_fit_builds
                )
            ),

        "calibration_builds":
            int(
                len(
                    split.calibration_builds
                )
            ),

        "evaluation_builds":
            int(
                evaluation[
                    "build_index"
                ].nunique()
            ),

        "model_fit_build_min":
            min(
                split.model_fit_builds
            ),

        "model_fit_build_max":
            max(
                split.model_fit_builds
            ),

        "calibration_build_min":
            min(
                split.calibration_builds
            ),

        "calibration_build_max":
            max(
                split.calibration_builds
            ),

        "evaluation_build_min":
            evaluation[
                "build_index"
            ].min(),

        "evaluation_build_max":
            evaluation[
                "build_index"
            ].max(),

        "initial_model_fit_builds":
            int(
                split.initial_model_fit_builds
            ),

        "expanded_by_builds":
            int(
                split.expanded_by_builds
            ),

        "calibration_positive_rows":
            int(
                split.support.positive_rows
            ),

        "calibration_negative_rows":
            int(
                split.support.negative_rows
            ),

        "calibration_failing_builds":
            int(
                split.support.failing_builds
            ),

        "calibration_passing_builds":
            int(
                split.support.passing_builds
            ),
    }

    return (
        predictions,
        diagnostics,
    )


def validate_insufficient_support_policy(
    calibration_config: dict,
) -> str:
    policy = (
        calibration_config[
            "rare_event_policy"
        ]
    )

    actions = (
        policy[
            "insufficient_support_action"
        ]
    )

    expected_actions = {
        "uncalibrated":
            "run",

        "platt":
            "skip_with_status",

        "isotonic":
            "skip_with_status",
    }

    if actions != expected_actions:
        raise ValueError(
            "RQ1 insufficient-support "
            "actions do not match the "
            "frozen calibration policy."
        )

    skip_status = str(
        policy[
            "skip_status"
        ]
    )

    if (
        skip_status
        !=
        "INSUFFICIENT_CALIBRATION_SUPPORT"
    ):
        raise ValueError(
            "RQ1 insufficient-support "
            "status does not match the "
            "frozen calibration policy."
        )

    return skip_status


def run_rq1_model(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    model_name: str,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    skip_status = (
        validate_insufficient_support_policy(
            calibration_config
        )
    )

    feature_columns = (
        validate_rq1_integration_contract(
            rq1_config=rq1_config,
            model_config=model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )
    )

    target_column = (
        model_config[
            "target"
        ][
            "field"
        ]
    )

    validate_rq1_temporal_partitions(
        development=development,
        evaluation=evaluation,
        feature_columns=
            feature_columns,
        target_column=
            target_column,
    )

    split = (
        split_development_for_calibration(
            development,
            rq1_config=
                rq1_config,
            calibration_config=
                calibration_config,
            build_column=
                "build_index",
            target_column=
                target_column,
        )
    )

    if split.sufficient:
        (
            predictions,
            diagnostics,
        ) = run_sufficient_rq1_model(
            development=development,
            evaluation=evaluation,
            model_name=model_name,
            rq1_config=rq1_config,
            model_config=model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )

        diagnostics = {
            **diagnostics,

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

    if split.status != skip_status:
        raise AssertionError(
            "Unexpected insufficient "
            "calibration status: "
            f"{split.status}"
        )

    (
        model_fit,
        calibration,
    ) = (
        materialize_calibration_partitions(
            development,
            split,
            build_column=
                "build_index",
        )
    )

    validate_materialized_calibration_coverage(
        development=development,
        model_fit=model_fit,
        calibration=calibration,
        split=split,
        build_column="build_index",
    )

    model = build_rq1_model(
        model_name=model_name,
        feature_columns=
            feature_columns,
        model_config=
            model_config,
    )

    fitted_model = (
        fit_probability_model(
            model,
            model_fit[
                feature_columns
            ],
            model_fit[
                target_column
            ],
        )
    )

    evaluation_probability = (
        positive_probability(
            fitted_model,
            evaluation[
                feature_columns
            ],
        )
    )

    predictions = evaluation[
        list(
            PREDICTION_ID_COLUMNS
        )
    ].copy()

    predictions[
        PROBABILITY_COLUMNS[
            "uncalibrated"
        ]
    ] = evaluation_probability

    predictions[
        PROBABILITY_COLUMNS[
            "platt"
        ]
    ] = float(
        "nan"
    )

    predictions[
        PROBABILITY_COLUMNS[
            "isotonic"
        ]
    ] = float(
        "nan"
    )

    predictions = predictions[
        prediction_columns()
    ]

    if not predictions[
        PROBABILITY_COLUMNS[
            "uncalibrated"
        ]
    ].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Invalid uncalibrated "
            "probability."
        )

    if not predictions[
        PROBABILITY_COLUMNS[
            "platt"
        ]
    ].isna().all():
        raise AssertionError(
            "Skipped Platt predictions "
            "must be missing."
        )

    if not predictions[
        PROBABILITY_COLUMNS[
            "isotonic"
        ]
    ].isna().all():
        raise AssertionError(
            "Skipped isotonic predictions "
            "must be missing."
        )

    diagnostics = {
        "model":
            model_name,

        "status":
            split.status,

        "calibration_support_sufficient":
            False,

        "uncalibrated_available":
            True,

        "platt_available":
            False,

        "isotonic_available":
            False,

        "model_fit_rows":
            int(
                len(
                    model_fit
                )
            ),

        "calibration_rows":
            int(
                len(
                    calibration
                )
            ),

        "evaluation_rows":
            int(
                len(
                    evaluation
                )
            ),

        "model_fit_builds":
            int(
                len(
                    split.model_fit_builds
                )
            ),

        "calibration_builds":
            int(
                len(
                    split.calibration_builds
                )
            ),

        "evaluation_builds":
            int(
                evaluation[
                    "build_index"
                ].nunique()
            ),

        "model_fit_build_min":
            min(
                split.model_fit_builds
            ),

        "model_fit_build_max":
            max(
                split.model_fit_builds
            ),

        "calibration_build_min":
            min(
                split.calibration_builds
            ),

        "calibration_build_max":
            max(
                split.calibration_builds
            ),

        "evaluation_build_min":
            evaluation[
                "build_index"
            ].min(),

        "evaluation_build_max":
            evaluation[
                "build_index"
            ].max(),

        "initial_model_fit_builds":
            int(
                split.initial_model_fit_builds
            ),

        "expanded_by_builds":
            int(
                split.expanded_by_builds
            ),

        "calibration_positive_rows":
            int(
                split.support.positive_rows
            ),

        "calibration_negative_rows":
            int(
                split.support.negative_rows
            ),

        "calibration_failing_builds":
            int(
                split.support.failing_builds
            ),

        "calibration_passing_builds":
            int(
                split.support.passing_builds
            ),
    }

    return (
        predictions,
        diagnostics,
    )


def validate_materialized_calibration_coverage(
    development: pd.DataFrame,
    model_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    split,
    build_column: str = "build_index",
) -> None:
    development_builds = set(
        development[
            build_column
        ].unique()
    )

    model_fit_builds = set(
        model_fit[
            build_column
        ].unique()
    )

    calibration_builds = set(
        calibration[
            build_column
        ].unique()
    )

    expected_model_fit_builds = set(
        split.model_fit_builds
    )

    expected_calibration_builds = set(
        split.calibration_builds
    )

    if (
        model_fit_builds
        != expected_model_fit_builds
    ):
        raise AssertionError(
            "Materialized model-fit "
            "builds do not match the "
            "calibration split."
        )

    if (
        calibration_builds
        != expected_calibration_builds
    ):
        raise AssertionError(
            "Materialized calibration "
            "builds do not match the "
            "calibration split."
        )

    if not model_fit_builds.isdisjoint(
        calibration_builds
    ):
        raise AssertionError(
            "Materialized model-fit and "
            "calibration builds overlap."
        )

    covered_builds = (
        model_fit_builds
        | calibration_builds
    )

    if (
        covered_builds
        != development_builds
    ):
        raise AssertionError(
            "Calibration partitions do "
            "not cover all development "
            "builds."
        )

    if (
        len(
            model_fit
        )
        + len(
            calibration
        )
        != len(
            development
        )
    ):
        raise AssertionError(
            "Calibration partition row "
            "coverage does not match "
            "development data."
        )
