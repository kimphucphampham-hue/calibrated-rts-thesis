"""Reproduce the legacy graduation-project experiment.

This script supports two temporal splitting modes:

1. original
   Reproduces the row-level TimeSeriesSplit used in the available
   legacy source code.

2. corrected
   Applies TimeSeriesSplit to unique Sprint_ID values, ensuring that
   one Sprint cannot appear in both training and testing data.

Run from the repository root:

    python legacy/reproduce_legacy.py

Optional:

    python legacy/reproduce_legacy.py --mode original
    python legacy/reproduce_legacy.py --mode corrected
    python legacy/reproduce_legacy.py --mode both

Outputs are written under:

    results/legacy/reproduction/
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# REPOSITORY PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "legacy.yaml"
)


# ============================================================
# EXPECTED DATASET SCHEMA
# ============================================================

IDENTIFIER_COLUMNS = [
    "Sprint_ID",
    "TestCase_ID",
]

FEATURE_COLUMNS = [
    "Code_Churn",
    "Cyclomatic_Complexity",
    "Dependencies",
    "Failure_History",
    "Execution_Time_ms",
    "Test_Age",
]

LABEL_COLUMN = "Label"

REQUIRED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + FEATURE_COLUMNS
    + [LABEL_COLUMN]
)


def resolve_repo_path(path_value: str) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Read and validate the YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the frozen synthetic dataset and validate its schema."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset was not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError("The synthetic dataset is empty.")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "The synthetic dataset contains missing values "
            "in required columns."
        )

    invalid_labels = set(
        df[LABEL_COLUMN].unique().tolist()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            "Label contains values other than 0 and 1: "
            f"{sorted(invalid_labels)}"
        )

    # Stable sorting preserves the existing order of rows
    # located within the same Sprint.
    df = (
        df.sort_values(
            by="Sprint_ID",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return df


def create_models(
    config: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Create the same three model families as the legacy source."""

    model_config = config.get("models", {})

    rf_config = model_config.get(
        "random_forest",
        {},
    )

    svm_config = model_config.get(
        "svm",
        {},
    )

    gb_config = model_config.get(
        "gradient_boosting",
        {},
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=int(
                rf_config.get(
                    "n_estimators",
                    100,
                )
            ),
            class_weight="balanced",
            random_state=random_seed,
        ),
        "SVM": SVC(
            kernel=str(
                svm_config.get(
                    "kernel",
                    "rbf",
                )
            ),
            class_weight="balanced",
            random_state=random_seed,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=int(
                gb_config.get(
                    "n_estimators",
                    100,
                )
            ),
            random_state=random_seed,
        ),
    }

    return models


def safe_filename(value: str) -> str:
    """Convert a model or mode name into a safe file-name fragment."""

    normalized = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        value.strip().lower(),
    )

    return normalized.strip("_")


def generate_original_splits(
    df: pd.DataFrame,
    n_splits: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Reproduce row-level TimeSeriesSplit from the legacy source."""

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
    )

    for fold, (
        train_positions,
        test_positions,
    ) in enumerate(
        splitter.split(df),
        start=1,
    ):
        yield (
            fold,
            train_positions,
            test_positions,
        )


def generate_corrected_splits(
    df: pd.DataFrame,
    n_splits: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Split unique Sprint_ID values instead of individual rows."""

    unique_sprints = (
        df["Sprint_ID"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    if len(unique_sprints) <= n_splits:
        raise ValueError(
            "There are not enough unique Sprints for "
            f"n_splits={n_splits}. "
            f"Found only {len(unique_sprints)} Sprints."
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
    )

    for fold, (
        train_sprint_positions,
        test_sprint_positions,
    ) in enumerate(
        splitter.split(unique_sprints),
        start=1,
    ):
        train_sprints = unique_sprints[
            train_sprint_positions
        ]

        test_sprints = unique_sprints[
            test_sprint_positions
        ]

        train_positions = np.flatnonzero(
            df["Sprint_ID"]
            .isin(train_sprints)
            .to_numpy()
        )

        test_positions = np.flatnonzero(
            df["Sprint_ID"]
            .isin(test_sprints)
            .to_numpy()
        )

        if not set(train_sprints).isdisjoint(
            set(test_sprints)
        ):
            raise RuntimeError(
                f"Fold {fold}: the same Sprint appears "
                "in both training and testing data."
            )

        if max(train_sprints) >= min(test_sprints):
            raise RuntimeError(
                f"Fold {fold}: training Sprints are not "
                "strictly earlier than testing Sprints."
            )

        yield (
            fold,
            train_positions,
            test_positions,
        )


def apply_smote(
    x_train_scaled: np.ndarray,
    y_train: pd.Series,
    config: dict[str, Any],
    random_seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply SMOTE only to training data."""

    smote_config = config.get(
        "smote",
        {},
    )

    enabled = bool(
        smote_config.get(
            "enabled",
            True,
        )
    )

    if not enabled:
        return (
            x_train_scaled,
            y_train.to_numpy(),
            "disabled",
        )

    class_counts = y_train.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            f"Fold {fold}: training data contains only "
            "one label class."
        )

    minority_count = int(
        class_counts.min()
    )

    if minority_count < 2:
        print(
            f"[WARNING] Fold {fold}: SMOTE was skipped "
            "because the minority class has fewer than "
            "two samples."
        )

        return (
            x_train_scaled,
            y_train.to_numpy(),
            "skipped_minority_too_small",
        )

    configured_k = int(
        smote_config.get(
            "k_neighbors",
            5,
        )
    )

    actual_k = min(
        configured_k,
        minority_count - 1,
    )

    smote = SMOTE(
        random_state=random_seed,
        k_neighbors=actual_k,
    )

    x_resampled, y_resampled = (
        smote.fit_resample(
            x_train_scaled,
            y_train,
        )
    )

    return (
        x_resampled,
        y_resampled,
        f"applied_k_{actual_k}",
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calculate classification metrics and confusion-matrix counts."""

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
        "false_negative": int(false_negative),
    }


def run_experiment_mode(
    df: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one splitting mode and save its outputs."""

    random_seed = int(
        config.get(
            "random_seed",
            42,
        )
    )

    n_splits = int(
        config.get(
            "n_splits",
            5,
        )
    )

    if mode == "original":
        split_generator = generate_original_splits(
            df,
            n_splits,
        )
    elif mode == "corrected":
        split_generator = generate_corrected_splits(
            df,
            n_splits,
        )
    else:
        raise ValueError(
            f"Unsupported experiment mode: {mode}"
        )

    metric_rows: list[dict[str, Any]] = []
    selected_test_frames: list[pd.DataFrame] = []

    print()
    print("=" * 72)
    print(
        f"RUNNING EXPERIMENT MODE: {mode.upper()}"
    )
    print("=" * 72)

    for (
        fold,
        train_positions,
        test_positions,
    ) in split_generator:
        train_df = df.iloc[
            train_positions
        ].copy()

        test_df = df.iloc[
            test_positions
        ].copy()

        train_sprints = sorted(
            train_df["Sprint_ID"]
            .unique()
            .tolist()
        )

        test_sprints = sorted(
            test_df["Sprint_ID"]
            .unique()
            .tolist()
        )

        overlapping_sprints = sorted(
            set(train_sprints).intersection(
                test_sprints
            )
        )

        print()
        print(
            f"[Fold {fold}/{n_splits}]"
        )

        print(
            "  Train Sprints: "
            f"{min(train_sprints)}"
            f" -> {max(train_sprints)}"
        )

        print(
            "  Test Sprints:  "
            f"{min(test_sprints)}"
            f" -> {max(test_sprints)}"
        )

        print(
            "  Overlapping Sprints: "
            f"{overlapping_sprints}"
        )

        x_train = train_df[
            FEATURE_COLUMNS
        ]

        y_train = train_df[
            LABEL_COLUMN
        ]

        x_test = test_df[
            FEATURE_COLUMNS
        ]

        y_test = test_df[
            LABEL_COLUMN
        ]

        # Fit the scaler only on training data.
        scaler = StandardScaler()

        x_train_scaled = scaler.fit_transform(
            x_train
        )

        x_test_scaled = scaler.transform(
            x_test
        )

        (
            x_train_resampled,
            y_train_resampled,
            smote_status,
        ) = apply_smote(
            x_train_scaled=x_train_scaled,
            y_train=y_train,
            config=config,
            random_seed=random_seed,
            fold=fold,
        )

        models = create_models(
            config=config,
            random_seed=random_seed,
        )

        for model_name, model in models.items():
            fit_started_at = time.perf_counter()

            model.fit(
                x_train_resampled,
                y_train_resampled,
            )

            fit_seconds = (
                time.perf_counter()
                - fit_started_at
            )

            predict_started_at = time.perf_counter()

            y_pred = model.predict(
                x_test_scaled
            )

            prediction_seconds = (
                time.perf_counter()
                - predict_started_at
            )

            metrics = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
            )

            selected_mask = y_pred == 1

            selected_df = test_df.loc[
                selected_mask,
                [
                    "Sprint_ID",
                    "TestCase_ID",
                    "Execution_Time_ms",
                    "Label",
                ],
            ].copy()

            selected_df.insert(
                0,
                "split_mode",
                mode,
            )

            selected_df.insert(
                1,
                "fold",
                fold,
            )

            selected_df.insert(
                2,
                "model",
                model_name,
            )

            selected_test_frames.append(
                selected_df
            )

            selected_count = int(
                selected_mask.sum()
            )

            selection_ratio = (
                selected_count
                / len(test_df)
            )

            total_runtime_ms = float(
                test_df[
                    "Execution_Time_ms"
                ].sum()
            )

            selected_runtime_ms = float(
                test_df.loc[
                    selected_mask,
                    "Execution_Time_ms",
                ].sum()
            )

            runtime_ratio = (
                selected_runtime_ms
                / total_runtime_ms
                if total_runtime_ms > 0
                else 0.0
            )

            metric_rows.append(
                {
                    "split_mode": mode,
                    "fold": fold,
                    "model": model_name,
                    "train_row_count": int(
                        len(train_df)
                    ),
                    "test_row_count": int(
                        len(test_df)
                    ),
                    "train_sprint_start": int(
                        min(train_sprints)
                    ),
                    "train_sprint_end": int(
                        max(train_sprints)
                    ),
                    "test_sprint_start": int(
                        min(test_sprints)
                    ),
                    "test_sprint_end": int(
                        max(test_sprints)
                    ),
                    "overlapping_sprints": (
                        ",".join(
                            str(sprint)
                            for sprint
                            in overlapping_sprints
                        )
                    ),
                    "train_positive_rate": float(
                        y_train.mean()
                    ),
                    "test_positive_rate": float(
                        y_test.mean()
                    ),
                    "smote_status": smote_status,
                    **metrics,
                    "selected_count": (
                        selected_count
                    ),
                    "selection_ratio": float(
                        selection_ratio
                    ),
                    "total_runtime_ms": (
                        total_runtime_ms
                    ),
                    "selected_runtime_ms": (
                        selected_runtime_ms
                    ),
                    "runtime_ratio": float(
                        runtime_ratio
                    ),
                    "fit_seconds": float(
                        fit_seconds
                    ),
                    "prediction_seconds": float(
                        prediction_seconds
                    ),
                }
            )

            print(
                f"  {model_name}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Selected={selected_count}/"
                f"{len(test_df)}"
            )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    summary_df = (
        metrics_df
        .groupby(
            [
                "split_mode",
                "model",
            ],
            as_index=False,
        )
        .agg(
            mean_precision=(
                "precision",
                "mean",
            ),
            mean_recall=(
                "recall",
                "mean",
            ),
            mean_f1=(
                "f1",
                "mean",
            ),
            std_f1=(
                "f1",
                "std",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            mean_selection_ratio=(
                "selection_ratio",
                "mean",
            ),
            mean_runtime_ratio=(
                "runtime_ratio",
                "mean",
            ),
            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
    )

    selected_tests_df = (
        pd.concat(
            selected_test_frames,
            ignore_index=True,
        )
        if selected_test_frames
        else pd.DataFrame()
    )

    metrics_output_path = (
        output_directory
        / f"fold_metrics_{mode}.csv"
    )

    summary_output_path = (
        output_directory
        / f"summary_metrics_{mode}.csv"
    )

    selected_output_path = (
        output_directory
        / f"selected_tests_{mode}.csv"
    )

    metrics_df.to_csv(
        metrics_output_path,
        index=False,
    )

    summary_df.to_csv(
        summary_output_path,
        index=False,
    )

    selected_tests_df.to_csv(
        selected_output_path,
        index=False,
    )

    print()
    print(
        f"[INFO] Fold metrics saved to: "
        f"{metrics_output_path}"
    )

    print(
        f"[INFO] Summary metrics saved to: "
        f"{summary_output_path}"
    )

    print(
        f"[INFO] Selected tests saved to: "
        f"{selected_output_path}"
    )

    return metrics_df, summary_df


def create_report_comparison(
    summary_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compare reproduced metrics with the report's reference values."""

    reference_metrics = config.get(
        "reference_metrics",
        {},
    )

    comparison_rows: list[dict[str, Any]] = []

    for _, summary_row in summary_df.iterrows():
        model_name = str(
            summary_row["model"]
        )

        reference = reference_metrics.get(
            model_name
        )

        if not isinstance(reference, dict):
            continue

        reproduced_precision = float(
            summary_row["mean_precision"]
        )

        reproduced_recall = float(
            summary_row["mean_recall"]
        )

        reproduced_f1 = float(
            summary_row["mean_f1"]
        )

        reference_precision = float(
            reference["precision"]
        )

        reference_recall = float(
            reference["recall"]
        )

        reference_f1 = float(
            reference["f1"]
        )

        comparison_rows.append(
            {
                "split_mode": summary_row[
                    "split_mode"
                ],
                "model": model_name,
                "report_precision": (
                    reference_precision
                ),
                "reproduced_precision": (
                    reproduced_precision
                ),
                "precision_difference_pp": (
                    reproduced_precision
                    - reference_precision
                )
                * 100,
                "report_recall": (
                    reference_recall
                ),
                "reproduced_recall": (
                    reproduced_recall
                ),
                "recall_difference_pp": (
                    reproduced_recall
                    - reference_recall
                )
                * 100,
                "report_f1": (
                    reference_f1
                ),
                "reproduced_f1": (
                    reproduced_f1
                ),
                "f1_difference_pp": (
                    reproduced_f1
                    - reference_f1
                )
                * 100,
            }
        )

    return pd.DataFrame(
        comparison_rows
    )


def save_run_metadata(
    output_directory: Path,
    config_path: Path,
    data_path: Path,
    mode: str,
) -> None:
    """Save environment and run information for reproducibility."""

    metadata = {
        "mode": mode,
        "repository_root": str(REPO_ROOT),
        "config_path": str(config_path),
        "data_path": str(data_path),
        "python_version": (
            platform.python_version()
        ),
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": (
                sklearn.__version__
            ),
            "imbalanced-learn": (
                imblearn.__version__
            ),
            "joblib": joblib.__version__,
            "pyyaml": yaml.__version__,
        },
    }

    metadata_path = (
        output_directory
        / "run_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the legacy regression-test "
            "selection experiment."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG_PATH
        ),
        help=(
            "Path to the legacy YAML configuration."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "original",
            "corrected",
            "both",
        ],
        default=None,
        help=(
            "Override the experiment mode "
            "defined in the YAML configuration."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the configured legacy reproduction experiment."""

    arguments = parse_arguments()

    config_path = Path(
        arguments.config
    )

    if not config_path.is_absolute():
        config_path = resolve_repo_path(
            arguments.config
        )

    try:
        config = load_yaml_config(
            config_path
        )

        configured_mode = str(
            config.get(
                "mode",
                "both",
            )
        )

        mode = (
            arguments.mode
            if arguments.mode is not None
            else configured_mode
        )

        if mode not in {
            "original",
            "corrected",
            "both",
        }:
            raise ValueError(
                "Mode must be original, corrected or both."
            )

        data_path = resolve_repo_path(
            config["data"]["path"]
        )

        output_directory = resolve_repo_path(
            config["output"]["directory"]
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[INFO] Repository root: {REPO_ROOT}"
        )

        print(
            f"[INFO] Configuration: {config_path}"
        )

        print(
            f"[INFO] Dataset: {data_path}"
        )

        print(
            f"[INFO] Output directory: "
            f"{output_directory}"
        )

        print(
            f"[INFO] Requested mode: {mode}"
        )

        df = load_dataset(
            data_path
        )

        print(
            f"[INFO] Loaded dataset: "
            f"{len(df)} rows x "
            f"{len(df.columns)} columns"
        )

        print(
            f"[INFO] Unique Sprints: "
            f"{df['Sprint_ID'].nunique()}"
        )

        print(
            f"[INFO] Unique Test Cases: "
            f"{df['TestCase_ID'].nunique()}"
        )

        print(
            f"[INFO] Positive-label rate: "
            f"{df[LABEL_COLUMN].mean():.2%}"
        )

        modes_to_run = (
            [
                "original",
                "corrected",
            ]
            if mode == "both"
            else [mode]
        )

        all_summary_frames: list[
            pd.DataFrame
        ] = []

        for experiment_mode in modes_to_run:
            _, summary_df = run_experiment_mode(
                df=df,
                config=config,
                mode=experiment_mode,
                output_directory=output_directory,
            )

            all_summary_frames.append(
                summary_df
            )

        combined_summary_df = pd.concat(
            all_summary_frames,
            ignore_index=True,
        )

        combined_summary_path = (
            output_directory
            / "summary_metrics_all_modes.csv"
        )

        combined_summary_df.to_csv(
            combined_summary_path,
            index=False,
        )

        comparison_df = create_report_comparison(
            summary_df=combined_summary_df,
            config=config,
        )

        comparison_path = (
            output_directory
            / "comparison_with_report.csv"
        )

        comparison_df.to_csv(
            comparison_path,
            index=False,
        )

        config_snapshot_path = (
            output_directory
            / "config_snapshot.yaml"
        )

        with config_snapshot_path.open(
            "w",
            encoding="utf-8",
        ) as snapshot_file:
            yaml.safe_dump(
                config,
                snapshot_file,
                sort_keys=False,
                allow_unicode=True,
            )

        save_run_metadata(
            output_directory=output_directory,
            config_path=config_path,
            data_path=data_path,
            mode=mode,
        )

        print()
        print("=" * 72)
        print("LEGACY REPRODUCTION COMPLETED")
        print("=" * 72)

        print()
        print(
            combined_summary_df.to_string(
                index=False
            )
        )

        print()
        print(
            "[INFO] Combined summary: "
            f"{combined_summary_path}"
        )

        print(
            "[INFO] Comparison with report: "
            f"{comparison_path}"
        )

        print(
            "[INFO] Run metadata: "
            f"{output_directory / 'run_metadata.json'}"
        )

        return 0

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        yaml.YAMLError,
    ) as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())