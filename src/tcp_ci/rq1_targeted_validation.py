from __future__ import annotations

import pandas as pd

from src.tcp_ci.calibration_split import (
    materialize_calibration_partitions,
    split_development_for_calibration,
)
from src.tcp_ci.model_runner import (
    fit_probability_model,
    positive_probability,
    validate_feature_list,
)
from src.tcp_ci.probability_calibration import (
    build_isotonic_calibrator,
    build_platt_calibrator,
)
from src.tcp_ci.rq1_pipeline import (
    PREDICTION_ID_COLUMNS,
    PROBABILITY_COLUMNS,
    RQ1_CALIBRATION_VARIANTS,
    RQ1_MODEL_NAMES,
    build_rq1_model,
    prediction_columns,
    validate_insufficient_support_policy,
    validate_materialized_calibration_coverage,
    validate_rq1_integration_contract,
    validate_rq1_temporal_partitions,
)


TARGETED_FEATURE_SET_ORDER = (
    "F1",
    "F1_F2",
    "F1_F2_F3",
)


def validate_targeted_validation_contract(
    targeted_config: dict,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> dict[str, list[str]]:
    validate_rq1_integration_contract(
        rq1_config=rq1_config,
        model_config=model_config,
        calibration_config=
            calibration_config,
        core_features_config=
            core_features_config,
    )

    comparison_order = tuple(
        targeted_config[
            "feature_ablation"
        ][
            "comparison_order"
        ]
    )

    if comparison_order != TARGETED_FEATURE_SET_ORDER:
        raise ValueError(
            "Targeted feature-set order does not "
            "match the frozen ablation contract."
        )

    if (
        targeted_config[
            "validation"
        ][
            "primary_final_evidence"
        ]
        is not False
    ):
        raise ValueError(
            "Targeted validation must remain "
            "supplementary evidence."
        )

    if (
        targeted_config[
            "result_policy"
        ][
            "final_thesis_claims_allowed"
        ]
        is not False
    ):
        raise ValueError(
            "Targeted validation must not be "
            "used as primary thesis evidence."
        )

    if list(
        targeted_config[
            "models"
        ]
    ) != list(
        RQ1_MODEL_NAMES
    ):
        raise ValueError(
            "Targeted model list does not match "
            "the frozen RQ1 model list."
        )

    if list(
        rq1_config[
            "models"
        ]
    ) != list(
        RQ1_MODEL_NAMES
    ):
        raise ValueError(
            "RQ1 model list has drifted."
        )

    if list(
        model_config[
            "models"
        ].keys()
    ) != list(
        RQ1_MODEL_NAMES
    ):
        raise ValueError(
            "Model configuration has drifted."
        )

    if list(
        targeted_config[
            "calibration_variants"
        ]
    ) != list(
        RQ1_CALIBRATION_VARIANTS
    ):
        raise ValueError(
            "Targeted calibration variants have drifted."
        )

    if list(
        rq1_config[
            "calibration_variants"
        ]
    ) != list(
        RQ1_CALIBRATION_VARIANTS
    ):
        raise ValueError(
            "RQ1 calibration variants have drifted."
        )

    if list(
        calibration_config[
            "methods"
        ]
    ) != list(
        RQ1_CALIBRATION_VARIANTS
    ):
        raise ValueError(
            "Calibration configuration has drifted."
        )

    primary_feature_set = (
        core_features_config[
            "primary_feature_set"
        ]
    )

    if (
        primary_feature_set
        != "F1_F2_F3"
    ):
        raise ValueError(
            "Frozen CORE primary feature set has drifted."
        )

    if (
        targeted_config[
            "feature_ablation"
        ][
            "primary_feature_set_remains"
        ]
        != primary_feature_set
    ):
        raise ValueError(
            "Targeted validation must not change "
            "the primary CORE feature set."
        )

    if (
        rq1_config[
            "feature_set"
        ][
            "name"
        ]
        != primary_feature_set
    ):
        raise ValueError(
            "Frozen RQ1 primary feature set has drifted."
        )

    if (
        model_config[
            "feature_config"
        ][
            "primary_feature_set"
        ]
        != primary_feature_set
    ):
        raise ValueError(
            "Frozen model primary feature set has drifted."
        )

    feature_sets = (
        core_features_config[
            "feature_sets"
        ]
    )

    resolved = {}

    for feature_set_name in (
        TARGETED_FEATURE_SET_ORDER
    ):
        if feature_set_name not in feature_sets:
            raise ValueError(
                "Targeted feature set is not defined "
                f"in CORE features: {feature_set_name}"
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

        resolved[
            feature_set_name
        ] = features

    if not (
        set(
            resolved[
                "F1"
            ]
        )
        <
        set(
            resolved[
                "F1_F2"
            ]
        )
        <
        set(
            resolved[
                "F1_F2_F3"
            ]
        )
    ):
        raise ValueError(
            "Frozen feature sets must be strictly nested."
        )

    if [
        len(
            resolved[
                name
            ]
        )
        for name in TARGETED_FEATURE_SET_ORDER
    ] != [
        7,
        8,
        11,
    ]:
        raise ValueError(
            "Frozen targeted feature-set sizes "
            "must remain 7, 8, and 11."
        )

    temporal = targeted_config[
        "temporal_design"
    ]

    required_true = (
        "reuse_rq1_v1",
        "outer_evaluation_boundary_unchanged",
        "calibration_window_rule_unchanged",
        "minimum_support_rule_unchanged",
    )

    for field in required_true:
        if temporal[
            field
        ] is not True:
            raise ValueError(
                "Targeted temporal contract "
                f"has drifted: {field}"
            )

    required_false = (
        "evaluation_used_for_model_fit",
        "evaluation_used_for_calibration_fit",
        "evaluation_used_for_feature_selection",
    )

    for field in required_false:
        if temporal[
            field
        ] is not False:
            raise ValueError(
                "Targeted temporal isolation "
                f"has drifted: {field}"
            )

    if (
        targeted_config[
            "retraining"
        ][
            "enabled"
        ]
        is not False
    ):
        raise ValueError(
            "Targeted pilot retraining must remain disabled."
        )

    if (
        targeted_config[
            "feature_ablation"
        ][
            "feature_selection_from_results"
        ]
        is not False
    ):
        raise ValueError(
            "Feature selection from targeted "
            "results is forbidden."
        )

    reporting = targeted_config[
        "reporting"
    ]

    if (
        reporting[
            "primary_unit"
        ]
        != "project"
    ):
        raise ValueError(
            "Targeted reporting primary unit "
            "must remain project."
        )

    if (
        reporting[
            "pool_all_project_rows_as_primary_result"
        ]
        is not False
    ):
        raise ValueError(
            "Targeted validation must not pool "
            "all projects as the primary result."
        )

    if (
        reporting[
            "comparison_factor"
        ]
        != "feature_set"
    ):
        raise ValueError(
            "Targeted comparison factor must "
            "remain feature_set."
        )

    if (
        targeted_config[
            "freeze_policy"
        ][
            "result_driven_changes_allowed"
        ]
        is not False
    ):
        raise ValueError(
            "Result-driven targeted changes "
            "must remain disabled."
        )

    result_policy = targeted_config[
        "result_policy"
    ]

    for field in (
        "select_best_feature_set",
        "select_best_model",
        "select_best_calibration_method",
        "change_primary_feature_set_from_results",
    ):
        if result_policy[
            field
        ] is not False:
            raise ValueError(
                "Result-driven targeted-validation "
                f"selection is forbidden: {field}"
            )

    grid = targeted_config[
        "expected_grid"
    ]

    expected_grid = {
        "projects": 5,
        "feature_sets": 3,
        "models": 3,
        "calibration_variants": 3,
        "maximum_conditions": 135,
    }

    for (
        field,
        expected_value,
    ) in expected_grid.items():
        if int(
            grid[
                field
            ]
        ) != expected_value:
            raise ValueError(
                "Targeted-validation grid "
                f"has drifted: {field}"
            )

    condition_count = (
        int(
            grid[
                "projects"
            ]
        )
        * int(
            grid[
                "feature_sets"
            ]
        )
        * int(
            grid[
                "models"
            ]
        )
        * int(
            grid[
                "calibration_variants"
            ]
        )
    )

    if (
        condition_count
        != int(
            grid[
                "maximum_conditions"
            ]
        )
        or condition_count != 135
    ):
        raise ValueError(
            "Targeted-validation grid must remain "
            "exactly 135 conditions."
        )

    return resolved


def targeted_feature_columns(
    feature_set_name: str,
    targeted_config: dict,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> list[str]:
    resolved = (
        validate_targeted_validation_contract(
            targeted_config=
                targeted_config,
            rq1_config=
                rq1_config,
            model_config=
                model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )
    )

    if feature_set_name not in resolved:
        raise ValueError(
            "Unsupported targeted feature set: "
            f"{feature_set_name}"
        )

    return list(
        resolved[
            feature_set_name
        ]
    )


def _build_diagnostics(
    *,
    model_name: str,
    feature_set_name: str,
    split,
    model_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    support_sufficient: bool,
) -> dict:
    return {
        "model":
            model_name,

        "feature_set":
            feature_set_name,

        "status":
            split.status,

        "calibration_support_sufficient":
            bool(
                support_sufficient
            ),

        "uncalibrated_available":
            True,

        "platt_available":
            bool(
                support_sufficient
            ),

        "isotonic_available":
            bool(
                support_sufficient
            ),

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


def run_targeted_rq1_model(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    targeted_config: dict,
    rq1_config: dict,
    model_config: dict,
    calibration_config: dict,
    core_features_config: dict,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    feature_columns = (
        targeted_feature_columns(
            feature_set_name=
                feature_set_name,
            targeted_config=
                targeted_config,
            rq1_config=
                rq1_config,
            model_config=
                model_config,
            calibration_config=
                calibration_config,
            core_features_config=
                core_features_config,
        )
    )

    if model_name not in RQ1_MODEL_NAMES:
        raise ValueError(
            f"Unsupported RQ1 model: {model_name}"
        )

    skip_status = (
        validate_insufficient_support_policy(
            calibration_config
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

    if split.sufficient:
        calibration_probability = (
            positive_probability(
                fitted_model,
                calibration[
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

        predictions[
            PROBABILITY_COLUMNS[
                "platt"
            ]
        ] = platt.transform(
            evaluation_probability
        )

        predictions[
            PROBABILITY_COLUMNS[
                "isotonic"
            ]
        ] = isotonic.transform(
            evaluation_probability
        )

        support_sufficient = True

    else:
        if split.status != skip_status:
            raise AssertionError(
                "Unexpected insufficient "
                "calibration status: "
                f"{split.status}"
            )

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

        support_sufficient = False

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
            "Invalid uncalibrated probability."
        )

    if support_sufficient:
        for variant in (
            "platt",
            "isotonic",
        ):
            column = (
                PROBABILITY_COLUMNS[
                    variant
                ]
            )

            if not predictions[
                column
            ].between(
                0.0,
                1.0,
            ).all():
                raise AssertionError(
                    "Invalid calibrated "
                    f"probability in {column}."
                )

    else:
        for variant in (
            "platt",
            "isotonic",
        ):
            column = (
                PROBABILITY_COLUMNS[
                    variant
                ]
            )

            if not predictions[
                column
            ].isna().all():
                raise AssertionError(
                    "Skipped calibrated "
                    "predictions must be missing."
                )

    diagnostics = (
        _build_diagnostics(
            model_name=
                model_name,
            feature_set_name=
                feature_set_name,
            split=
                split,
            model_fit=
                model_fit,
            calibration=
                calibration,
            evaluation=
                evaluation,
            support_sufficient=
                support_sufficient,
        )
    )

    return (
        predictions,
        diagnostics,
    )
