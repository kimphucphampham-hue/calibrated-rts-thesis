from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from src.tcp_ci.rq1_plots import (
    plot_reliability_comparison,
    save_reliability_figure,
    select_reliability_bins,
    validate_reliability_plot_contract,
)


CONFIG_PATH = Path(
    "configs/calibration_v1.yaml"
)


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def make_bins():
    rows = []

    variant_values = {
        "uncalibrated": [
            (
                0,
                0.05,
                0.02,
                3,
                0.75,
            ),
            (
                9,
                0.95,
                1.00,
                1,
                0.25,
            ),
        ],

        "platt": [
            (
                0,
                0.04,
                0.02,
                3,
                0.75,
            ),
            (
                9,
                0.92,
                1.00,
                1,
                0.25,
            ),
        ],

        "isotonic": [
            (
                0,
                0.02,
                0.02,
                3,
                0.75,
            ),
            (
                9,
                1.00,
                1.00,
                1,
                0.25,
            ),
        ],
    }

    for (
        variant,
        values,
    ) in variant_values.items():
        for (
            bin_id,
            mean_probability,
            observed_rate,
            count,
            weight,
        ) in values:
            gap = abs(
                mean_probability
                - observed_rate
            )

            rows.append(
                {
                    "project_id":
                        "project-a",

                    "model":
                        "logistic_regression",

                    "calibration_variant":
                        variant,

                    "condition_status":
                        "AVAILABLE",

                    "bin_id":
                        bin_id,

                    "bin_lower":
                        bin_id / 10,

                    "bin_upper":
                        (
                            bin_id + 1
                        ) / 10,

                    "count":
                        count,

                    "weight":
                        weight,

                    "mean_predicted_probability":
                        mean_probability,

                    "observed_failure_rate":
                        observed_rate,

                    "absolute_gap":
                        gap,

                    "weighted_gap":
                        weight
                        * gap,
                }
            )

    return pd.DataFrame(
        rows
    )


def test_frozen_reliability_contract_is_accepted():
    config = load_config()

    n_bins = (
        validate_reliability_plot_contract(
            config
        )
    )

    assert n_bins == 10


def test_reliability_contract_drift_is_rejected():
    config = deepcopy(
        load_config()
    )

    config[
        "reliability_diagram"
    ][
        "binning"
    ][
        "n_bins"
    ] = 20

    with pytest.raises(
        ValueError,
        match="10 bins",
    ):
        validate_reliability_plot_contract(
            config
        )


def test_saved_bins_are_selected_and_sorted():
    selected = (
        select_reliability_bins(
            reliability_bins=
                make_bins(),
            project_id=
                "project-a",
            model_name=
                "logistic_regression",
            n_bins=10,
        )
    )

    assert selected[
        "calibration_variant"
    ].tolist() == [
        "uncalibrated",
        "uncalibrated",
        "platt",
        "platt",
        "isotonic",
        "isotonic",
    ]

    for variant in [
        "uncalibrated",
        "platt",
        "isotonic",
    ]:
        group = selected[
            selected[
                "calibration_variant"
            ]
            == variant
        ]

        assert group[
            "bin_id"
        ].tolist() == [
            0,
            9,
        ]

        assert np.isclose(
            group[
                "weight"
            ].sum(),
            1.0,
        )


def test_duplicate_saved_bin_is_rejected():
    bins = make_bins()

    bins = pd.concat(
        [
            bins,
            bins.iloc[
                [
                    0
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate reliability bin",
    ):
        select_reliability_bins(
            reliability_bins=bins,
            project_id=
                "project-a",
            model_name=
                "logistic_regression",
            n_bins=10,
        )


def test_invalid_saved_bin_definition_is_rejected():
    bins = make_bins()

    bins.loc[
        bins.index[0],
        "bin_lower",
    ] = 0.2

    with pytest.raises(
        ValueError,
        match="lower bounds",
    ):
        select_reliability_bins(
            reliability_bins=bins,
            project_id=
                "project-a",
            model_name=
                "logistic_regression",
            n_bins=10,
        )


def test_plot_contains_reference_and_three_variants():
    figure, axis = (
        plot_reliability_comparison(
            reliability_bins=
                make_bins(),
            project_id=
                "project-a",
            model_name=
                "logistic_regression",
            calibration_config=
                load_config(),
        )
    )

    try:
        labels = [
            line.get_label()
            for line in axis.lines
        ]

        assert labels == [
            "Perfect calibration",
            "Uncalibrated",
            "Platt",
            "Isotonic",
        ]

        identity = (
            axis.lines[
                0
            ]
        )

        np.testing.assert_allclose(
            identity.get_xdata(),
            [
                0.0,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            identity.get_ydata(),
            [
                0.0,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            axis.get_xlim(),
            [
                0.0,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            axis.get_ylim(),
            [
                0.0,
                1.0,
            ],
        )

        assert (
            axis.get_xlabel()
            ==
            "Mean predicted probability"
        )

        assert (
            axis.get_ylabel()
            ==
            "Observed failure frequency"
        )

    finally:
        plt.close(
            figure
        )


def test_missing_calibration_variants_are_not_fabricated():
    bins = make_bins()

    bins = bins[
        bins[
            "calibration_variant"
        ]
        == "uncalibrated"
    ].copy()

    figure, axis = (
        plot_reliability_comparison(
            reliability_bins=bins,
            project_id=
                "project-a",
            model_name=
                "logistic_regression",
            calibration_config=
                load_config(),
        )
    )

    try:
        labels = [
            line.get_label()
            for line in axis.lines
        ]

        assert labels == [
            "Perfect calibration",
            "Uncalibrated",
        ]

    finally:
        plt.close(
            figure
        )


def test_reliability_figure_can_be_saved(
    tmp_path,
):
    figure, _ = (
        plot_reliability_comparison(
            reliability_bins=
                make_bins(),
            project_id=
                "project-a",
            model_name=
                "logistic_regression",
            calibration_config=
                load_config(),
        )
    )

    try:
        output = (
            tmp_path
            / "reliability.png"
        )

        result = (
            save_reliability_figure(
                figure,
                output,
            )
        )

        assert result == output

        assert output.exists()

        assert (
            output.stat().st_size
            > 0
        )

    finally:
        plt.close(
            figure
        )


def test_safe_plot_name_removes_path_separators():
    from src.tcp_ci.rq1_plots import (
        safe_plot_name,
    )

    result = safe_plot_name(
        "datasets/Angel-ML@angel"
    )

    assert "/" not in result
    assert "@" not in result

    assert result == (
        "datasets_Angel-ML_angel"
    )


def test_batch_generation_creates_one_figure_per_project_model(
    tmp_path,
):
    from src.tcp_ci.rq1_plots import (
        generate_reliability_figures,
    )

    bins_a = make_bins()

    bins_b = (
        make_bins()
        .copy()
    )

    bins_b[
        "project_id"
    ] = "project-b"

    bins_b[
        "model"
    ] = "random_forest"

    bins = pd.concat(
        [
            bins_a,
            bins_b,
        ],
        ignore_index=True,
    )

    manifest = (
        generate_reliability_figures(
            reliability_bins=bins,
            calibration_config=
                load_config(),
            output_dir=tmp_path,
        )
    )

    assert len(
        manifest
    ) == 2

    assert manifest[
        [
            "project_id",
            "model",
        ]
    ].drop_duplicates().shape[
        0
    ] == 2

    assert manifest[
        "variant_count"
    ].eq(
        3
    ).all()

    assert manifest[
        "bytes"
    ].gt(
        0
    ).all()

    for filename in manifest[
        "filename"
    ]:
        path = (
            tmp_path
            / filename
        )

        assert path.exists()

        assert (
            path.stat().st_size
            > 0
        )


def make_metric_summary():
    rows = []

    projects = [
        "datasets/project-a@repo-a",
        "datasets/project-b@repo-b",
    ]

    models = [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]

    variants = [
        "uncalibrated",
        "platt",
        "isotonic",
    ]

    for (
        project_index,
        project_id,
    ) in enumerate(
        projects
    ):
        for (
            model_index,
            model_name,
        ) in enumerate(
            models
        ):
            for (
                variant_index,
                variant,
            ) in enumerate(
                variants
            ):
                base = (
                    0.01
                    * (
                        1
                        + project_index
                        + model_index
                        + variant_index
                    )
                )

                rows.append(
                    {
                        "project_id":
                            project_id,

                        "model":
                            model_name,

                        "calibration_variant":
                            variant,

                        "condition_status":
                            "AVAILABLE",

                        "brier_score":
                            base,

                        "ece":
                            base / 2.0,

                        "absolute_calibration_bias":
                            base / 3.0,
                    }
                )

    return pd.DataFrame(
        rows
    )


def test_project_plot_label_removes_dataset_prefix():
    from src.tcp_ci.rq1_plots import (
        project_plot_label,
    )

    assert (
        project_plot_label(
            "datasets/Angel-ML@angel"
        )
        ==
        "Angel-ML@angel"
    )

    assert (
        project_plot_label(
            "custom-project"
        )
        ==
        "custom-project"
    )


def test_calibration_metric_rows_follow_fixed_variant_order():
    from src.tcp_ci.rq1_plots import (
        select_calibration_metric_rows,
    )

    metrics = (
        make_metric_summary()
    )

    selected = (
        select_calibration_metric_rows(
            metrics=metrics,
            model_name=
                "logistic_regression",
            metric_name=
                "brier_score",
        )
    )

    assert selected[
        "project_id"
    ].tolist() == [
        "datasets/project-a@repo-a",
        "datasets/project-a@repo-a",
        "datasets/project-a@repo-a",
        "datasets/project-b@repo-b",
        "datasets/project-b@repo-b",
        "datasets/project-b@repo-b",
    ]

    assert selected[
        "calibration_variant"
    ].tolist() == [
        "uncalibrated",
        "platt",
        "isotonic",
        "uncalibrated",
        "platt",
        "isotonic",
    ]


def test_duplicate_calibration_metric_condition_is_rejected():
    from src.tcp_ci.rq1_plots import (
        select_calibration_metric_rows,
    )

    metrics = (
        make_metric_summary()
    )

    metrics = pd.concat(
        [
            metrics,
            metrics.iloc[
                [
                    0
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate RQ1 metric",
    ):
        select_calibration_metric_rows(
            metrics=metrics,
            model_name=
                "logistic_regression",
            metric_name=
                "ece",
        )


def test_unavailable_metric_condition_is_not_fabricated():
    from src.tcp_ci.rq1_plots import (
        select_calibration_metric_rows,
    )

    metrics = (
        make_metric_summary()
    )

    mask = (
        (
            metrics[
                "project_id"
            ]
            ==
            "datasets/project-a@repo-a"
        )
        &
        (
            metrics[
                "model"
            ]
            ==
            "logistic_regression"
        )
        &
        (
            metrics[
                "calibration_variant"
            ]
            ==
            "isotonic"
        )
    )

    metrics.loc[
        mask,
        "condition_status",
    ] = (
        "INSUFFICIENT_CALIBRATION_SUPPORT"
    )

    metrics.loc[
        mask,
        "ece",
    ] = np.nan

    selected = (
        select_calibration_metric_rows(
            metrics=metrics,
            model_name=
                "logistic_regression",
            metric_name=
                "ece",
        )
    )

    assert not (
        (
            selected[
                "project_id"
            ]
            ==
            "datasets/project-a@repo-a"
        )
        &
        (
            selected[
                "calibration_variant"
            ]
            ==
            "isotonic"
        )
    ).any()


def test_calibration_metric_summary_plot_has_three_variant_groups():
    from src.tcp_ci.rq1_plots import (
        plot_calibration_metric_summary,
    )

    figure, axis = (
        plot_calibration_metric_summary(
            metrics=
                make_metric_summary(),
            model_name=
                "logistic_regression",
            metric_name=
                "brier_score",
        )
    )

    try:
        legend = (
            axis.get_legend()
        )

        labels = [
            text.get_text()
            for text in (
                legend.get_texts()
            )
        ]

        assert labels == [
            "Uncalibrated",
            "Platt",
            "Isotonic",
        ]

        assert (
            axis.get_xlabel()
            == "Project"
        )

        assert (
            axis.get_ylabel()
            == "Brier score"
        )

        assert (
            axis.get_ylim()[
                0
            ]
            == 0.0
        )

    finally:
        plt.close(
            figure
        )


def test_batch_calibration_summary_creates_nine_figures(
    tmp_path,
):
    from src.tcp_ci.rq1_plots import (
        generate_calibration_summary_figures,
    )

    manifest = (
        generate_calibration_summary_figures(
            metrics=
                make_metric_summary(),
            output_dir=
                tmp_path,
        )
    )

    assert len(
        manifest
    ) == 9

    assert manifest[
        "model"
    ].nunique() == 3

    assert manifest[
        "metric"
    ].nunique() == 3

    assert manifest[
        "project_count"
    ].eq(
        2
    ).all()

    assert manifest[
        "available_conditions"
    ].eq(
        6
    ).all()

    assert manifest[
        "bytes"
    ].gt(
        0
    ).all()

    for filename in manifest[
        "filename"
    ]:
        output = (
            tmp_path
            / filename
        )

        assert output.exists()

        assert (
            output.stat().st_size
            > 0
        )


def test_calibration_metric_shared_y_upper_uses_all_models():
    from src.tcp_ci.rq1_plots import (
        calibration_metric_shared_y_upper,
    )

    metrics = (
        make_metric_summary()
    )

    expected_max = metrics[
        "brier_score"
    ].max()

    result = (
        calibration_metric_shared_y_upper(
            metrics=metrics,
            metric_name=
                "brier_score",
        )
    )

    assert np.isclose(
        result,
        expected_max
        * 1.05,
    )


def test_summary_plots_can_share_identical_y_limits():
    from src.tcp_ci.rq1_plots import (
        calibration_metric_shared_y_upper,
        plot_calibration_metric_summary,
    )

    metrics = (
        make_metric_summary()
    )

    y_upper = (
        calibration_metric_shared_y_upper(
            metrics=metrics,
            metric_name=
                "ece",
        )
    )

    figures = []

    try:
        limits = []

        for model_name in [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ]:
            figure, axis = (
                plot_calibration_metric_summary(
                    metrics=metrics,
                    model_name=
                        model_name,
                    metric_name=
                        "ece",
                    y_upper=
                        y_upper,
                )
            )

            figures.append(
                figure
            )

            limits.append(
                axis.get_ylim()
            )

        assert (
            limits[
                0
            ]
            ==
            limits[
                1
            ]
            ==
            limits[
                2
            ]
        )

        assert np.isclose(
            limits[
                0
            ][
                0
            ],
            0.0,
        )

        assert np.isclose(
            limits[
                0
            ][
                1
            ],
            y_upper,
        )

    finally:
        for figure in figures:
            plt.close(
                figure
            )
