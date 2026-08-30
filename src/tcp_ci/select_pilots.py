from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG = Path(
    "configs/pilot_screening_v1.yaml"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/pilot/screening"
)


def load_config(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required inventory columns: "
            + ", ".join(missing)
        )


def extract_owner(
    project_key: str,
) -> str:
    project_name = (
        str(project_key)
        .rstrip("/")
        .split("/")[-1]
    )

    if "@" not in project_name:
        return project_name

    return project_name.split(
        "@",
        1,
    )[0]


def build_screening_table(
    inventory: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    required_columns = [
        "project_key",
        "pass_rows",
        "fail_rows",
        "other_result_rows",
        "failure_rate",
        "duration_missing",
        "unique_tests",
        "unique_builds_in_exe",
        "build_time_parse_rate",
    ]

    require_columns(
        inventory,
        required_columns,
    )

    df = inventory.copy()

    df["owner"] = (
        df["project_key"]
        .astype(str)
        .map(extract_owner)
    )

    df["execution_rows"] = (
        df["pass_rows"]
        + df["fail_rows"]
        + df["other_result_rows"]
    )

    criteria = config["eligibility"]

    df["pass_min_builds"] = (
        df["unique_builds_in_exe"]
        >= criteria["min_unique_builds"]
    )

    df["pass_min_tests"] = (
        df["unique_tests"]
        >= criteria["min_unique_tests"]
    )

    df["pass_min_failures"] = (
        df["fail_rows"]
        >= criteria["min_fail_rows"]
    )

    df["pass_min_failure_rate"] = (
        df["failure_rate"]
        >= criteria["min_failure_rate"]
    )

    df["pass_execution_row_cap"] = (
        df["execution_rows"]
        <= criteria["max_execution_rows"]
    )

    if criteria[
        "require_zero_other_result_rows"
    ]:
        df["pass_outcome_domain"] = (
            df["other_result_rows"] == 0
        )
    else:
        df["pass_outcome_domain"] = True

    if criteria[
        "require_zero_duration_missing"
    ]:
        df["pass_duration"] = (
            df["duration_missing"] == 0
        )
    else:
        df["pass_duration"] = True

    if criteria[
        "require_full_timestamp_parse"
    ]:
        df["pass_timestamp"] = (
            df["build_time_parse_rate"] == 1.0
        )
    else:
        df["pass_timestamp"] = True

    check_columns = [
        "pass_min_builds",
        "pass_min_tests",
        "pass_min_failures",
        "pass_min_failure_rate",
        "pass_execution_row_cap",
        "pass_outcome_domain",
        "pass_duration",
        "pass_timestamp",
    ]

    df["eligible"] = (
        df[check_columns]
        .all(axis=1)
    )

    def exclusion_reasons(
        row: pd.Series,
    ) -> str:
        reasons = []

        reason_map = {
            "pass_min_builds":
                "too_few_builds",
            "pass_min_tests":
                "too_few_tests",
            "pass_min_failures":
                "too_few_failures",
            "pass_min_failure_rate":
                "failure_rate_too_low",
            "pass_execution_row_cap":
                "too_many_execution_rows",
            "pass_outcome_domain":
                "unexpected_outcome_values",
            "pass_duration":
                "missing_duration",
            "pass_timestamp":
                "timestamp_not_fully_parseable",
        }

        for check, reason in (
            reason_map.items()
        ):
            if not bool(row[check]):
                reasons.append(reason)

        return ";".join(reasons)

    df["exclusion_reasons"] = df.apply(
        exclusion_reasons,
        axis=1,
    )

    return df


def select_projects(
    screening: pd.DataFrame,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pilot_count = int(
        config["selection"]["pilot_count"]
    )

    max_per_owner = int(
        config["selection"][
            "max_projects_per_owner"
        ]
    )

    eligible = screening[
        screening["eligible"]
    ].copy()

    eligible = eligible.sort_values(
        by=[
            "fail_rows",
            "unique_builds_in_exe",
            "project_key",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    )

    selected_indices = []
    owner_counts: dict[str, int] = {}

    for index, row in eligible.iterrows():
        owner = str(row["owner"])

        count = owner_counts.get(
            owner,
            0,
        )

        if count >= max_per_owner:
            continue

        selected_indices.append(index)

        owner_counts[owner] = (
            count + 1
        )

        if (
            len(selected_indices)
            >= pilot_count
        ):
            break

    if len(selected_indices) < pilot_count:
        raise RuntimeError(
            "Not enough eligible projects to "
            f"select {pilot_count} pilots "
            "under the current diversity rule."
        )

    screening = screening.copy()

    screening["selected"] = False
    screening["selected_rank"] = pd.NA

    for rank, index in enumerate(
        selected_indices,
        start=1,
    ):
        screening.loc[
            index,
            "selected",
        ] = True

        screening.loc[
            index,
            "selected_rank",
        ] = rank

    selected = (
        screening.loc[
            selected_indices
        ]
        .copy()
        .sort_values(
            "selected_rank"
        )
    )

    return screening, selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    inventory_path = Path(
        config["source_inventory"]["path"]
    )

    inventory = pd.read_csv(
        inventory_path
    )

    screening = build_screening_table(
        inventory,
        config,
    )

    screening, selected = (
        select_projects(
            screening,
            config,
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    screening_path = (
        args.output_dir
        / "project_screening.csv"
    )

    selected_path = (
        args.output_dir
        / "selected_projects.csv"
    )

    screening.to_csv(
        screening_path,
        index=False,
    )

    selected.to_csv(
        selected_path,
        index=False,
    )

    print(
        "[PASS] Pilot screening completed."
    )

    print(
        "Projects inventoried:",
        len(screening),
    )

    print(
        "Eligible projects:",
        int(screening["eligible"].sum()),
    )

    print(
        "Selected projects:",
        len(selected),
    )

    print()

    display_columns = [
        "selected_rank",
        "project_key",
        "owner",
        "unique_builds_in_exe",
        "unique_tests",
        "fail_rows",
        "failure_rate",
        "execution_rows",
    ]

    print(
        selected[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Screening table:",
        screening_path,
    )

    print(
        "Selected projects:",
        selected_path,
    )


if __name__ == "__main__":
    main()