from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CONFIG = Path(
    "configs/baseline_v1.yaml"
)


IDENTIFIER_COLUMNS = [
    "project_id",
    "build_id",
    "build_index",
    "test_id",
]


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def safe_project_name(
    project_key: str,
) -> str:
    return (
        project_key
        .replace("/", "__")
        .replace("@", "__")
    )


def validate_feature_list(
    features: list[str],
) -> None:
    forbidden = {
        "failed",
        "execution_time_ms",
        "verdict_raw",
        "duration_raw",
        "split",
    }

    overlap = (
        set(features)
        & forbidden
    )

    if overlap:
        raise ValueError(
            "Forbidden post-execution/model-control "
            "fields found in feature list: "
            + ", ".join(
                sorted(overlap)
            )
        )

    if len(features) != len(
        set(features)
    ):
        raise ValueError(
            "Duplicate feature names found."
        )


def load_split(
    path: Path,
    feature_columns: list[str],
    target: str,
    cost_field: str,
) -> pd.DataFrame:
    required = (
        IDENTIFIER_COLUMNS
        + feature_columns
        + [
            target,
            cost_field,
        ]
    )

    dataframe = pd.read_csv(
        path,
        usecols=required,
        dtype={
            "project_id": "string",
            "build_id": "string",
            "test_id": "string",
        },
        low_memory=False,
    )

    missing = (
        set(required)
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{path}: missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    dataframe["build_index"] = (
        pd.to_numeric(
            dataframe["build_index"],
            errors="raise",
        )
        .astype("int64")
    )

    dataframe[target] = (
        pd.to_numeric(
            dataframe[target],
            errors="raise",
        )
        .astype("int8")
    )

    dataframe[cost_field] = (
        pd.to_numeric(
            dataframe[cost_field],
            errors="raise",
        )
        .astype("int64")
    )

    return dataframe


def validate_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> None:
    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if test.empty:
        raise ValueError(
            "Test split is empty."
        )

    train_projects = (
        train["project_id"]
        .dropna()
        .unique()
        .tolist()
    )

    test_projects = (
        test["project_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(train_projects) != 1:
        raise ValueError(
            "Expected exactly one project in train."
        )

    if train_projects != test_projects:
        raise ValueError(
            "Train/test project mismatch."
        )

    train_builds = set(
        train["build_id"]
        .astype(str)
        .unique()
    )

    test_builds = set(
        test["build_id"]
        .astype(str)
        .unique()
    )

    overlap = (
        train_builds
        & test_builds
    )

    if overlap:
        raise ValueError(
            "Train/test build overlap found."
        )

    if (
        train["build_index"].max()
        >= test["build_index"].min()
    ):
        raise ValueError(
            "Training period does not "
            "strictly precede test period."
        )

    train_classes = set(
        train[target].unique()
    )

    test_classes = set(
        test[target].unique()
    )

    if train_classes != {0, 1}:
        raise ValueError(
            "Training data must contain "
            "both target classes."
        )

    if test_classes != {0, 1}:
        raise ValueError(
            "Test data must contain "
            "both target classes."
        )


def build_logistic_pipeline(
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    preprocessing = config[
        "preprocessing"
    ]

    model_config = config[
        "models"
    ][
        "logistic_regression"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing[
                        "imputer"
                    ][
                        "strategy"
                    ],
                    add_indicator=bool(
                        preprocessing[
                            "imputer"
                        ][
                            "add_indicator"
                        ]
                    ),
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            (
                "history",
                numeric_pipeline,
                feature_columns,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        solver=model_config[
            "solver"
        ],
        C=float(
            model_config["C"]
        ),
        max_iter=int(
            model_config[
                "max_iter"
            ]
        ),
        class_weight=model_config[
            "class_weight"
        ],
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                transformer,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate_probabilities(
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict:
    probability = np.asarray(
        probability,
        dtype=float,
    )

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Non-finite probability found."
        )

    if not (
        (
            probability >= 0.0
        )
        &
        (
            probability <= 1.0
        )
    ).all():
        raise ValueError(
            "Probability outside [0, 1]."
        )

    y = np.asarray(
        y_true,
        dtype=int,
    )

    probability_matrix = (
        np.column_stack(
            [
                1.0 - probability,
                probability,
            ]
        )
    )

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    probability,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    y,
                    probability,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y,
                    probability,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y,
                    probability_matrix,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),
    }


def positive_probability(
    model,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    probabilities = (
        model.predict_proba(
            dataframe
        )
    )

    classes = list(
        model.classes_
        if hasattr(
            model,
            "classes_",
        )
        else model[
            "model"
        ].classes_
    )

    if 1 not in classes:
        raise ValueError(
            "Positive class 1 not found."
        )

    positive_index = (
        classes.index(1)
    )

    return probabilities[
        :,
        positive_index,
    ]


def run_project(
    project_key: str,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    features = config[
        "features"
    ]

    target = config[
        "target"
    ][
        "field"
    ]

    cost_field = config[
        "evaluation_cost"
    ][
        "field"
    ]

    validate_feature_list(
        features
    )

    split_root = Path(
        config[
            "inputs"
        ][
            "split_root"
        ]
    )

    project_dir = (
        split_root
        / safe_project_name(
            project_key
        )
    )

    train = load_split(
        project_dir / "train.csv",
        features,
        target,
        cost_field,
    )

    test = load_split(
        project_dir / "test.csv",
        features,
        target,
        cost_field,
    )

    validate_train_test(
        train,
        test,
        target,
    )

    X_train = train[
        features
    ]

    X_test = test[
        features
    ]

    y_train = train[
        target
    ]

    y_test = test[
        target
    ]

    # -------------------------------------------------
    # Dummy prior baseline
    # -------------------------------------------------

    dummy = DummyClassifier(
        strategy=config[
            "models"
        ][
            "dummy_prior"
        ][
            "strategy"
        ]
    )

    dummy.fit(
        np.zeros(
            (
                len(X_train),
                1,
            )
        ),
        y_train,
    )

    dummy_probability = (
        dummy.predict_proba(
            np.zeros(
                (
                    len(X_test),
                    1,
                )
            )
        )[
            :,
            list(
                dummy.classes_
            ).index(1),
        ]
    )

    # -------------------------------------------------
    # Logistic baseline
    #
    # Preprocessing is inside the sklearn Pipeline, so
    # imputer/scaler are fitted only during fit(X_train).
    # -------------------------------------------------

    logistic = (
        build_logistic_pipeline(
            features,
            config,
        )
    )

    logistic.fit(
        X_train,
        y_train,
    )

    logistic_probability = (
        logistic.predict_proba(
            X_test
        )[
            :,
            list(
                logistic[
                    "model"
                ].classes_
            ).index(1),
        ]
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    rows = []

    common = {
        "project_key":
            project_key,

        "train_rows":
            len(train),

        "test_rows":
            len(test),

        "train_builds":
            int(
                train[
                    "build_id"
                ].nunique()
            ),

        "test_builds":
            int(
                test[
                    "build_id"
                ].nunique()
            ),

        "train_failure_rows":
            int(
                y_train.sum()
            ),

        "test_failure_rows":
            int(
                y_test.sum()
            ),

        "train_failure_rate":
            float(
                y_train.mean()
            ),

        "test_failure_rate":
            float(
                y_test.mean()
            ),
    }

    for (
        model_name,
        probability,
    ) in [
        (
            "dummy_prior",
            dummy_probability,
        ),
        (
            "logistic_regression",
            logistic_probability,
        ),
    ]:
        metrics = (
            evaluate_probabilities(
                y_test,
                probability,
            )
        )

        rows.append(
            {
                **common,
                "model":
                    model_name,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        rows
    )

    predictions = test[
        [
            "project_id",
            "build_id",
            "build_index",
            "test_id",
            target,
            cost_field,
            "execution_time_mean_history_ms",
        ]
    ].copy()

    predictions[
        "p_dummy_prior"
    ] = (
        dummy_probability
    )

    predictions[
        "p_logistic_regression"
    ] = (
        logistic_probability
    )

    if not (
        predictions[
            [
                "p_dummy_prior",
                "p_logistic_regression",
            ]
        ]
        .apply(
            lambda column:
                column.between(
                    0.0,
                    1.0,
                ).all()
        )
        .all()
    ):
        raise AssertionError(
            "Invalid prediction probability."
        )

    return (
        metrics_df,
        predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    args = parser.parse_args()

    config = load_yaml(
        args.config
    )

    pilot_config = load_yaml(
        Path(
            config[
                "inputs"
            ][
                "pilot_projects_config"
            ]
        )
    )

    projects = [
        item["project_key"]
        for item in pilot_config[
            "projects"
        ]
    ]

    results_root = Path(
        config[
            "outputs"
        ][
            "results_root"
        ]
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_frames = []

    for project_key in projects:
        print(
            f"[INFO] Running baseline "
            f"for {project_key}"
        )

        (
            metrics,
            predictions,
        ) = run_project(
            project_key,
            config,
        )

        metric_frames.append(
            metrics
        )

        output_dir = (
            results_root
            / safe_project_name(
                project_key
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_csv(
            output_dir
            / "test_predictions.csv",
            index=False,
        )

    summary = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    summary_path = (
        results_root
        / "baseline_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    macro = (
        summary
        .groupby(
            "model",
            as_index=False,
        )[
            [
                "roc_auc",
                "average_precision",
                "brier_score",
                "log_loss",
            ]
        ]
        .mean()
    )

    macro_path = (
        results_root
        / "baseline_macro_summary.csv"
    )

    macro.to_csv(
        macro_path,
        index=False,
    )

    print()
    print(
        "[PASS] Baseline evaluation completed."
    )

    print()
    print(
        summary[
            [
                "project_key",
                "model",
                "train_rows",
                "test_rows",
                "train_failure_rate",
                "test_failure_rate",
                "roc_auc",
                "average_precision",
                "brier_score",
                "log_loss",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Macro mean across projects:"
    )

    print(
        macro.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()