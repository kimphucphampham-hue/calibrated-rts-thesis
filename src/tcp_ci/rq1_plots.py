from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RQ1_VARIANT_ORDER = (
    "uncalibrated",
    "platt",
    "isotonic",
)


RQ1_VARIANT_LABELS = {
    "uncalibrated":
        "Uncalibrated",

    "platt":
        "Platt",

    "isotonic":
        "Isotonic",
}


RELIABILITY_REQUIRED_COLUMNS = {
    "project_id",
    "model",
    "calibration_variant",
    "condition_status",
    "bin_id",
    "bin_lower",
    "bin_upper",
    "count",
    "weight",
    "mean_predicted_probability",
    "observed_failure_rate",
    "absolute_gap",
    "weighted_gap",
}


def validate_reliability_plot_contract(
    calibration_config: dict,
) -> int:
    reliability = (
        calibration_config[
            "reliability_diagram"
        ]
    )

    if (
        reliability[
            "enabled"
        ]
        is not True
    ):
        raise ValueError(
            "RQ1 reliability diagram "
            "must remain enabled."
        )

    binning = reliability[
        "binning"
    ]

    n_bins = int(
        binning[
            "n_bins"
        ]
    )

    if n_bins != 10:
        raise ValueError(
            "RQ1 reliability diagram "
            "requires 10 bins."
        )

    if (
        binning[
            "strategy"
        ]
        != "equal_width"
    ):
        raise ValueError(
            "RQ1 reliability diagram "
            "requires equal-width bins."
        )

    if not np.isclose(
        float(
            binning[
                "lower_bound"
            ]
        ),
        0.0,
    ):
        raise ValueError(
            "RQ1 reliability lower bound "
            "must be 0."
        )

    if not np.isclose(
        float(
            binning[
                "upper_bound"
            ]
        ),
        1.0,
    ):
        raise ValueError(
            "RQ1 reliability upper bound "
            "must be 1."
        )

    if (
        reliability[
            "x_axis"
        ]
        !=
        "mean_predicted_probability"
    ):
        raise ValueError(
            "RQ1 reliability x-axis "
            "contract has changed."
        )

    if (
        reliability[
            "y_axis"
        ]
        !=
        "observed_failure_frequency"
    ):
        raise ValueError(
            "RQ1 reliability y-axis "
            "contract has changed."
        )

    if (
        reliability[
            "reference_line"
        ]
        != "y=x"
    ):
        raise ValueError(
            "RQ1 reliability reference "
            "line must remain y=x."
        )

    if (
        reliability[
            "empty_bins"
        ]
        != "omit"
    ):
        raise ValueError(
            "RQ1 reliability empty-bin "
            "policy must remain omit."
        )

    generation_rule = str(
        reliability[
            "generation_rule"
        ]
    ).lower()

    if (
        "saved"
        not in generation_rule
        or
        "calibration-bin"
        not in generation_rule
    ):
        raise ValueError(
            "RQ1 reliability diagrams "
            "must use saved calibration "
            "bin outputs."
        )

    return n_bins


def select_reliability_bins(
    reliability_bins: pd.DataFrame,
    project_id: str,
    model_name: str,
    n_bins: int,
) -> pd.DataFrame:
    missing = (
        RELIABILITY_REQUIRED_COLUMNS
        - set(
            reliability_bins.columns
        )
    )

    if missing:
        raise ValueError(
            "Reliability-bin data are "
            "missing required columns: "
            f"{sorted(missing)}"
        )

    selected = reliability_bins[
        (
            reliability_bins[
                "project_id"
            ]
            == project_id
        )
        &
        (
            reliability_bins[
                "model"
            ]
            == model_name
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No saved reliability bins "
            "found for the requested "
            "project/model condition."
        )

    if not selected[
        "condition_status"
    ].eq(
        "AVAILABLE"
    ).all():
        raise ValueError(
            "Saved reliability-bin rows "
            "must represent AVAILABLE "
            "conditions only."
        )

    variants = set(
        selected[
            "calibration_variant"
        ].unique()
    )

    unknown_variants = (
        variants
        - set(
            RQ1_VARIANT_ORDER
        )
    )

    if unknown_variants:
        raise ValueError(
            "Unknown RQ1 calibration "
            f"variants: "
            f"{sorted(unknown_variants)}"
        )

    numeric_columns = [
        "bin_id",
        "bin_lower",
        "bin_upper",
        "count",
        "weight",
        "mean_predicted_probability",
        "observed_failure_rate",
        "absolute_gap",
        "weighted_gap",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            selected[
                column
            ],
            errors="raise",
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Reliability-bin data "
                f"contain non-finite "
                f"{column} values."
            )

        selected[
            column
        ] = values

    if not selected[
        "mean_predicted_probability"
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "Mean predicted probability "
            "must lie within [0, 1]."
        )

    if not selected[
        "observed_failure_rate"
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "Observed failure rate must "
            "lie within [0, 1]."
        )

    if not selected[
        "weight"
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "Reliability-bin weights must "
            "lie within [0, 1]."
        )

    if (
        selected[
            "count"
        ]
        <= 0
    ).any():
        raise ValueError(
            "Reliability-bin counts must "
            "be positive."
        )

    if not np.allclose(
        selected[
            "count"
        ],
        np.round(
            selected[
                "count"
            ]
        ),
    ):
        raise ValueError(
            "Reliability-bin counts must "
            "be integer-valued."
        )

    if (
        selected[
            "bin_id"
        ]
        < 0
    ).any() or (
        selected[
            "bin_id"
        ]
        >= n_bins
    ).any():
        raise ValueError(
            "Reliability bin_id lies "
            "outside the frozen bin range."
        )

    if not np.allclose(
        selected[
            "bin_id"
        ],
        np.round(
            selected[
                "bin_id"
            ]
        ),
    ):
        raise ValueError(
            "Reliability bin_id must be "
            "integer-valued."
        )

    selected[
        "bin_id"
    ] = selected[
        "bin_id"
    ].astype(
        int
    )

    duplicate_bins = (
        selected.duplicated(
            subset=[
                "calibration_variant",
                "bin_id",
            ]
        )
    )

    if duplicate_bins.any():
        raise ValueError(
            "Duplicate reliability bin "
            "found within a calibration "
            "variant."
        )

    expected_lower = (
        selected[
            "bin_id"
        ]
        / n_bins
    )

    expected_upper = (
        (
            selected[
                "bin_id"
            ]
            + 1
        )
        / n_bins
    )

    if not np.allclose(
        selected[
            "bin_lower"
        ],
        expected_lower,
    ):
        raise ValueError(
            "Saved bin lower bounds do "
            "not match the frozen "
            "equal-width definition."
        )

    if not np.allclose(
        selected[
            "bin_upper"
        ],
        expected_upper,
    ):
        raise ValueError(
            "Saved bin upper bounds do "
            "not match the frozen "
            "equal-width definition."
        )

    tolerance = 1e-12

    if (
        selected[
            "mean_predicted_probability"
        ]
        <
        (
            selected[
                "bin_lower"
            ]
            - tolerance
        )
    ).any():
        raise ValueError(
            "Mean predicted probability "
            "falls below its saved bin."
        )

    if (
        selected[
            "mean_predicted_probability"
        ]
        >
        (
            selected[
                "bin_upper"
            ]
            + tolerance
        )
    ).any():
        raise ValueError(
            "Mean predicted probability "
            "falls above its saved bin."
        )

    for variant in (
        RQ1_VARIANT_ORDER
    ):
        group = selected[
            selected[
                "calibration_variant"
            ]
            == variant
        ]

        if group.empty:
            continue

        if not np.isclose(
            group[
                "weight"
            ].sum(),
            1.0,
        ):
            raise ValueError(
                "Reliability-bin weights "
                f"for {variant} do not "
                "sum to 1."
            )

    variant_rank = {
        variant:
            index
        for (
            index,
            variant,
        ) in enumerate(
            RQ1_VARIANT_ORDER
        )
    }

    selected[
        "_variant_rank"
    ] = (
        selected[
            "calibration_variant"
        ].map(
            variant_rank
        )
    )

    selected = (
        selected.sort_values(
            [
                "_variant_rank",
                "bin_id",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_variant_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return selected


def plot_reliability_comparison(
    reliability_bins: pd.DataFrame,
    project_id: str,
    model_name: str,
    calibration_config: dict,
):
    n_bins = (
        validate_reliability_plot_contract(
            calibration_config
        )
    )

    selected = (
        select_reliability_bins(
            reliability_bins=
                reliability_bins,
            project_id=
                project_id,
            model_name=
                model_name,
            n_bins=
                n_bins,
        )
    )

    figure, axis = (
        plt.subplots(
            figsize=(
                6.0,
                6.0,
            )
        )
    )

    axis.plot(
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
        linestyle="--",
        label="Perfect calibration",
    )

    for variant in (
        RQ1_VARIANT_ORDER
    ):
        group = selected[
            selected[
                "calibration_variant"
            ]
            == variant
        ]

        if group.empty:
            continue

        axis.plot(
            group[
                "mean_predicted_probability"
            ],
            group[
                "observed_failure_rate"
            ],
            marker="o",
            label=
                RQ1_VARIANT_LABELS[
                    variant
                ],
        )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel(
        "Mean predicted probability"
    )

    axis.set_ylabel(
        "Observed failure frequency"
    )

    axis.set_title(
        "Reliability diagram\n"
        f"{project_id} — {model_name}"
    )

    axis.legend()

    figure.tight_layout()

    return (
        figure,
        axis,
    )


def save_reliability_figure(
    figure,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if (
        not output_path.exists()
        or
        output_path.stat().st_size
        <= 0
    ):
        raise RuntimeError(
            "Reliability figure was not "
            "written successfully."
        )

    return output_path


def safe_plot_name(
    value: str,
) -> str:
    result = []

    for character in str(
        value
    ):
        if (
            character.isalnum()
            or
            character
            in {
                "-",
                "_",
                ".",
            }
        ):
            result.append(
                character
            )

        else:
            result.append(
                "_"
            )

    name = "".join(
        result
    ).strip(
        "_"
    )

    if not name:
        raise ValueError(
            "Plot name must contain at "
            "least one usable character."
        )

    return name


def generate_reliability_figures(
    reliability_bins: pd.DataFrame,
    calibration_config: dict,
    output_dir: Path,
    dpi: int = 300,
) -> pd.DataFrame:
    n_bins = (
        validate_reliability_plot_contract(
            calibration_config
        )
    )

    missing = (
        RELIABILITY_REQUIRED_COLUMNS
        - set(
            reliability_bins.columns
        )
    )

    if missing:
        raise ValueError(
            "Reliability-bin data are "
            "missing required columns: "
            f"{sorted(missing)}"
        )

    combinations = (
        reliability_bins[
            [
                "project_id",
                "model",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "project_id",
                "model",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if combinations.empty:
        raise ValueError(
            "No project/model reliability "
            "conditions were found."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    for row in (
        combinations.itertuples(
            index=False
        )
    ):
        project_id = str(
            row.project_id
        )

        model_name = str(
            row.model
        )

        selected = (
            select_reliability_bins(
                reliability_bins=
                    reliability_bins,
                project_id=
                    project_id,
                model_name=
                    model_name,
                n_bins=
                    n_bins,
            )
        )

        available_variants = [
            variant
            for variant
            in RQ1_VARIANT_ORDER
            if not selected[
                selected[
                    "calibration_variant"
                ]
                == variant
            ].empty
        ]

        figure, _ = (
            plot_reliability_comparison(
                reliability_bins=
                    reliability_bins,
                project_id=
                    project_id,
                model_name=
                    model_name,
                calibration_config=
                    calibration_config,
            )
        )

        filename = (
            safe_plot_name(
                project_id
            )
            + "__"
            + safe_plot_name(
                model_name
            )
            + "__reliability.png"
        )

        output_path = (
            output_dir
            / filename
        )

        try:
            save_reliability_figure(
                figure,
                output_path,
                dpi=dpi,
            )

        finally:
            plt.close(
                figure
            )

        manifest_rows.append(
            {
                "project_id":
                    project_id,

                "model":
                    model_name,

                "variant_count":
                    len(
                        available_variants
                    ),

                "variants":
                    "|".join(
                        available_variants
                    ),

                "reliability_bin_rows":
                    len(
                        selected
                    ),

                "filename":
                    filename,

                "bytes":
                    output_path
                    .stat()
                    .st_size,
            }
        )

    return pd.DataFrame(
        manifest_rows
    )


CALIBRATION_SUMMARY_METRICS = {
    "brier_score":
        "Brier score",

    "ece":
        "Expected Calibration Error (ECE)",

    "absolute_calibration_bias":
        "Absolute calibration bias",
}


CALIBRATION_METRIC_REQUIRED_COLUMNS = {
    "project_id",
    "model",
    "calibration_variant",
    "condition_status",
    *CALIBRATION_SUMMARY_METRICS.keys(),
}


def project_plot_label(
    project_id: str,
) -> str:
    project_id = str(
        project_id
    )

    prefix = "datasets/"

    if project_id.startswith(
        prefix
    ):
        return project_id[
            len(
                prefix
            ):
        ]

    return project_id


def select_calibration_metric_rows(
    metrics: pd.DataFrame,
    model_name: str,
    metric_name: str,
) -> pd.DataFrame:
    if (
        metric_name
        not in CALIBRATION_SUMMARY_METRICS
    ):
        raise ValueError(
            "Unsupported calibration "
            f"summary metric: "
            f"{metric_name}"
        )

    missing = (
        CALIBRATION_METRIC_REQUIRED_COLUMNS
        - set(
            metrics.columns
        )
    )

    if missing:
        raise ValueError(
            "RQ1 metric data are missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    if metrics.empty:
        raise ValueError(
            "RQ1 metric data must not "
            "be empty."
        )

    duplicate_conditions = (
        metrics.duplicated(
            subset=[
                "project_id",
                "model",
                "calibration_variant",
            ]
        )
    )

    if duplicate_conditions.any():
        raise ValueError(
            "Duplicate RQ1 metric "
            "condition found."
        )

    variants = set(
        metrics[
            "calibration_variant"
        ].unique()
    )

    unknown_variants = (
        variants
        - set(
            RQ1_VARIANT_ORDER
        )
    )

    if unknown_variants:
        raise ValueError(
            "Unknown RQ1 calibration "
            f"variants: "
            f"{sorted(unknown_variants)}"
        )

    selected = metrics[
        metrics[
            "model"
        ]
        == model_name
    ].copy()

    if selected.empty:
        raise ValueError(
            "No RQ1 metric rows found "
            "for the requested model."
        )

    available = selected[
        selected[
            "condition_status"
        ]
        == "AVAILABLE"
    ].copy()

    if available.empty:
        raise ValueError(
            "No AVAILABLE RQ1 metric "
            "conditions found for the "
            "requested model."
        )

    values = pd.to_numeric(
        available[
            metric_name
        ],
        errors="raise",
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "AVAILABLE calibration metric "
            f"{metric_name} contains "
            "non-finite values."
        )

    if (
        values
        < 0.0
    ).any():
        raise ValueError(
            "Calibration summary metrics "
            "must be non-negative."
        )

    available[
        metric_name
    ] = values

    project_order = list(
        dict.fromkeys(
            selected[
                "project_id"
            ].tolist()
        )
    )

    project_rank = {
        project_id:
            index
        for (
            index,
            project_id,
        ) in enumerate(
            project_order
        )
    }

    variant_rank = {
        variant:
            index
        for (
            index,
            variant,
        ) in enumerate(
            RQ1_VARIANT_ORDER
        )
    }

    available[
        "_project_rank"
    ] = available[
        "project_id"
    ].map(
        project_rank
    )

    available[
        "_variant_rank"
    ] = available[
        "calibration_variant"
    ].map(
        variant_rank
    )

    available = (
        available.sort_values(
            [
                "_project_rank",
                "_variant_rank",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_project_rank",
                "_variant_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return available


def plot_calibration_metric_summary(
    metrics: pd.DataFrame,
    model_name: str,
    metric_name: str,
    y_upper: float | None = None,
):
    selected = (
        select_calibration_metric_rows(
            metrics=metrics,
            model_name=
                model_name,
            metric_name=
                metric_name,
        )
    )

    project_ids = list(
        dict.fromkeys(
            selected[
                "project_id"
            ].tolist()
        )
    )

    x = np.arange(
        len(
            project_ids
        ),
        dtype=float,
    )

    width = 0.24

    figure, axis = (
        plt.subplots(
            figsize=(
                9.0,
                5.5,
            )
        )
    )

    for (
        variant_index,
        variant,
    ) in enumerate(
        RQ1_VARIANT_ORDER
    ):
        values = []

        positions = []

        for (
            project_index,
            project_id,
        ) in enumerate(
            project_ids
        ):
            row = selected[
                (
                    selected[
                        "project_id"
                    ]
                    == project_id
                )
                &
                (
                    selected[
                        "calibration_variant"
                    ]
                    == variant
                )
            ]

            if row.empty:
                continue

            if len(
                row
            ) != 1:
                raise AssertionError(
                    "Expected exactly one "
                    "metric row per "
                    "project/model/variant."
                )

            offset = (
                variant_index
                - 1
            ) * width

            positions.append(
                x[
                    project_index
                ]
                + offset
            )

            values.append(
                float(
                    row[
                        metric_name
                    ].iloc[
                        0
                    ]
                )
            )

        if not values:
            continue

        axis.bar(
            positions,
            values,
            width=width,
            label=
                RQ1_VARIANT_LABELS[
                    variant
                ],
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            project_plot_label(
                project_id
            )
            for project_id
            in project_ids
        ],
        rotation=25,
        ha="right",
    )

    axis.set_xlabel(
        "Project"
    )

    axis.set_ylabel(
        CALIBRATION_SUMMARY_METRICS[
            metric_name
        ]
    )

    axis.set_title(
        (
            CALIBRATION_SUMMARY_METRICS[
                metric_name
            ]
            + "\n"
            + model_name
        )
    )

    if y_upper is None:
        axis.set_ylim(
            bottom=0.0
        )

    else:
        y_upper = float(
            y_upper
        )

        if (
            not np.isfinite(
                y_upper
            )
            or
            y_upper <= 0.0
        ):
            raise ValueError(
                "Summary y-axis upper "
                "limit must be finite "
                "and positive."
            )

        axis.set_ylim(
            0.0,
            y_upper,
        )

    axis.legend()

    figure.tight_layout()

    return (
        figure,
        axis,
    )


def generate_calibration_summary_figures(
    metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
) -> pd.DataFrame:
    missing = (
        CALIBRATION_METRIC_REQUIRED_COLUMNS
        - set(
            metrics.columns
        )
    )

    if missing:
        raise ValueError(
            "RQ1 metric data are missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    model_order = [
        model_name
        for model_name
        in (
            "logistic_regression",
            "random_forest",
            "xgboost",
        )
        if (
            metrics[
                "model"
            ]
            == model_name
        ).any()
    ]

    if not model_order:
        raise ValueError(
            "No supported RQ1 models "
            "were found in metric data."
        )

    shared_y_upper = {
        metric_name:
            calibration_metric_shared_y_upper(
                metrics=metrics,
                metric_name=
                    metric_name,
            )
        for metric_name
        in CALIBRATION_SUMMARY_METRICS
    }

    for model_name in (
        model_order
    ):
        for (
            metric_name,
            metric_label,
        ) in (
            CALIBRATION_SUMMARY_METRICS.items()
        ):
            selected = (
                select_calibration_metric_rows(
                    metrics=metrics,
                    model_name=
                        model_name,
                    metric_name=
                        metric_name,
                )
            )

            figure, _ = (
                plot_calibration_metric_summary(
                    metrics=metrics,
                    model_name=
                        model_name,
                    metric_name=
                        metric_name,
                    y_upper=
                        shared_y_upper[
                            metric_name
                        ],
                )
            )

            filename = (
                safe_plot_name(
                    model_name
                )
                + "__"
                + safe_plot_name(
                    metric_name
                )
                + "__summary.png"
            )

            output_path = (
                output_dir
                / filename
            )

            try:
                save_reliability_figure(
                    figure,
                    output_path,
                    dpi=dpi,
                )

            finally:
                plt.close(
                    figure
                )

            manifest_rows.append(
                {
                    "model":
                        model_name,

                    "metric":
                        metric_name,

                    "metric_label":
                        metric_label,

                    "project_count":
                        selected[
                            "project_id"
                        ].nunique(),

                    "available_conditions":
                        len(
                            selected
                        ),

                    "filename":
                        filename,

                    "bytes":
                        output_path
                        .stat()
                        .st_size,
                }
            )

    return pd.DataFrame(
        manifest_rows
    )


def calibration_metric_shared_y_upper(
    metrics: pd.DataFrame,
    metric_name: str,
    padding_fraction: float = 0.05,
) -> float:
    if (
        metric_name
        not in CALIBRATION_SUMMARY_METRICS
    ):
        raise ValueError(
            "Unsupported calibration "
            f"summary metric: "
            f"{metric_name}"
        )

    if (
        padding_fraction
        < 0.0
    ):
        raise ValueError(
            "Y-axis padding fraction "
            "must be non-negative."
        )

    required = {
        "condition_status",
        metric_name,
    }

    missing = (
        required
        - set(
            metrics.columns
        )
    )

    if missing:
        raise ValueError(
            "RQ1 metric data are missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    available = metrics[
        metrics[
            "condition_status"
        ]
        == "AVAILABLE"
    ]

    if available.empty:
        raise ValueError(
            "No AVAILABLE metric rows "
            "were found."
        )

    values = pd.to_numeric(
        available[
            metric_name
        ],
        errors="raise",
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "AVAILABLE calibration metric "
            f"{metric_name} contains "
            "non-finite values."
        )

    if (
        values < 0.0
    ).any():
        raise ValueError(
            "Calibration summary metrics "
            "must be non-negative."
        )

    maximum = float(
        values.max()
    )

    if maximum == 0.0:
        return 1.0

    return maximum * (
        1.0
        + padding_fraction
    )
