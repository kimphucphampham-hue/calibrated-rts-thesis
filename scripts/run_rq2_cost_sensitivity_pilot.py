"""RQ2 M2-01 historical cost sensitivity pilot.

Phase 11 chronological holdout validation ONLY.
Not final longitudinal evidence.
Not an approved M2 method freeze.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.tcp_ci.rq2_cost_adapter import estimate_build_costs
from src.tcp_ci.rq2_selector import risk_density_select


DATA_ROOT = Path("data/processed/tcp_ci/core_split")
PRED_ROOT = Path("results/core/rq1/predictions")
OUTPUT = Path("results/core/rq2/cost_sensitivity_pilot.csv")

PROJECTS = (
    "datasets__facebook__buck",
    "datasets__Angel-ML__angel",
)

MODEL = "logistic_regression"
PROBABILITY = "p_platt"
RHO = 0.95
BUILDS_PER_PROJECT = 3

KEY = [
    "project_id",
    "build_id",
    "build_index",
    "test_id",
]

READ_DTYPES = {
    "project_id": "string",
    "build_id": "string",
    "test_id": "string",
}

DATA_COLUMNS = KEY + [
    "failed",
    "execution_time_ms",
    "execution_time_mean_history_ms",
]

PRED_COLUMNS = KEY + [
    "failed",
    PROBABILITY,
]

HISTORY_COLUMNS = [
    "project_id",
    "build_index",
    "execution_time_ms",
]


def load_csv(path, columns):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    result = pd.read_csv(
        path,
        usecols=columns,
        dtype=READ_DTYPES,
    )

    if result.empty:
        raise ValueError(f"Empty input: {path}")

    if result[KEY].isna().any().any():
        raise ValueError(f"Null join key: {path}")

    if result.duplicated(subset=KEY).any():
        raise ValueError(f"Duplicate join key: {path}")

    return result


def load_project(project):
    directory = DATA_ROOT / project

    train = load_csv(
        directory / "train.csv",
        DATA_COLUMNS,
    )

    test = load_csv(
        directory / "test.csv",
        DATA_COLUMNS,
    )

    predictions = load_csv(
        PRED_ROOT / project / f"{MODEL}.csv",
        PRED_COLUMNS,
    )

    if train["project_id"].nunique() != 1:
        raise ValueError("Invalid training project.")

    if test["project_id"].nunique() != 1:
        raise ValueError("Invalid test project.")

    if (
        train["project_id"].iloc[0]
        != test["project_id"].iloc[0]
    ):
        raise ValueError("Train/test project mismatch.")

    if (
        train["build_index"].max()
        >= test["build_index"].min()
    ):
        raise ValueError(
            "Train/test chronological boundary violation."
        )

    # Verify complete one-to-one prediction alignment.
    aligned = test.merge(
        predictions,
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_observed", "_prediction"),
    )

    if not aligned["_merge"].eq("both").all():
        counts = aligned["_merge"].value_counts().to_dict()
        raise ValueError(
            f"Prediction/test alignment mismatch: {counts}"
        )

    if aligned[
        ["failed_observed", "failed_prediction"]
    ].isna().any().any():
        raise ValueError("Missing evaluation labels.")

    if not aligned[
        "failed_observed"
    ].eq(
        aligned["failed_prediction"]
    ).all():
        raise ValueError(
            "Prediction and test failure labels disagree."
        )

    aligned = aligned.drop(
        columns=["_merge", "failed_prediction"]
    ).rename(
        columns={"failed_observed": "failed"}
    )

    positive_train = train.loc[
        train["execution_time_ms"] > 0,
        "execution_time_ms",
    ]

    if positive_train.empty:
        raise ValueError(
            f"No positive training durations: {project}"
        )

    floor_ms = float(positive_train.min())

    if not np.isfinite(floor_ms) or floor_ms <= 0:
        raise ValueError("Invalid training-derived floor.")

    aligned = aligned.sort_values(
        ["build_index", "test_id"],
        kind="stable",
    ).reset_index(drop=True)

    return train, test, aligned, floor_ms


def choose_build_indices(aligned, project):
    indices = sorted(
        aligned["build_index"].unique().tolist()
    )

    # Buck: inspect the first builds containing
    # historical zero costs, using decision-time inputs.
    if project == "datasets__facebook__buck":
        zero_rows = aligned.loc[
            aligned["execution_time_mean_history_ms"].eq(0)
        ]

        indices = sorted(
            zero_rows["build_index"].unique().tolist()
        )

    if not indices:
        raise ValueError(
            f"No eligible pilot builds for {project}"
        )

    return indices[:BUILDS_PER_PROJECT]


def evaluate_selection(
    rows,
    cost_frame,
    selection,
    project,
    build_index,
    policy,
    floor_ms,
):
    selected_ids = set(selection.selected_test_ids)

    available_ids = set(rows["test_id"])

    if not selected_ids.issubset(available_ids):
        raise ValueError("Selector returned unknown test IDs.")

    mask = rows["test_id"].isin(selected_ids)

    if int(mask.sum()) != len(selected_ids):
        raise ValueError("Selected test count mismatch.")

    observed = rows["execution_time_ms"].astype(float)
    failures = rows["failed"].astype(int)

    costs = cost_frame["estimated_cost_ms"].astype(float)
    probabilities = rows[PROBABILITY].astype(float)

    total_failures = int(failures.sum())
    captured_failures = int(failures.loc[mask].sum())

    total_runtime = float(observed.sum())
    selected_runtime = float(observed.loc[mask].sum())

    total_risk = float(probabilities.sum())
    selected_risk = float(probabilities.loc[mask].sum())

    if total_risk <= 0:
        raise ValueError(
            "Zero total predicted risk; M2-02 unresolved."
        )

    retained_risk_ratio = selected_risk / total_risk

    if retained_risk_ratio < RHO - 1e-9:
        raise ValueError(
            "Risk coverage constraint violated."
        )

    return {
        "evidence_scope": "phase11_holdout_pilot",
        "project": project,
        "build_index": int(build_index),
        "build_id": str(rows["build_id"].iloc[0]),
        "model": MODEL,
        "probability": PROBABILITY,
        "rho": RHO,
        "policy": policy,
        "floor_ms": floor_ms,
        "n_tests": len(rows),
        "n_zero_history": int(
            rows["execution_time_mean_history_ms"]
            .eq(0).sum()
        ),
        "n_selected": int(mask.sum()),
        "tsr": float(mask.mean()),
        "estimated_selected_ms": float(
            costs.loc[mask].sum()
        ),
        "observed_selected_ms": selected_runtime,
        "observed_full_ms": total_runtime,
        "rtr": (
            selected_runtime / total_runtime
            if total_runtime > 0
            else np.nan
        ),
        "total_failures": total_failures,
        "captured_failures": captured_failures,
        "missed_failures": (
            total_failures - captured_failures
        ),
        "failure_recall": (
            captured_failures / total_failures
            if total_failures > 0
            else np.nan
        ),
        "retained_risk_ratio": retained_risk_ratio,
        "selected_test_ids": "|".join(
            sorted(selected_ids)
        ),
    }


def run_project(project):
    train, test, aligned, floor_ms = load_project(project)

    build_indices = choose_build_indices(
        aligned, project
    )

    print(f"\nPROJECT: {project}")
    print(f"Training-derived floor: {floor_ms:g} ms")
    print(f"Pilot build count: {len(build_indices)}")

    outputs = []

    for build_index in build_indices:
        rows = aligned.loc[
            aligned["build_index"].eq(build_index)
        ].copy()

        rows = rows.sort_values(
            "test_id",
            kind="stable",
        )

        # Retrospective full-information history:
        # only rows from strictly earlier builds.
        earlier_test = test.loc[
            test["build_index"] < build_index
        ]

        history = pd.concat(
            [
                train[HISTORY_COLUMNS],
                earlier_test[HISTORY_COLUMNS],
            ],
            ignore_index=True,
        )

        if not (
            history["build_index"] < build_index
        ).all():
            raise ValueError(
                "Temporal history boundary violated."
            )

        # Selection-time inputs only.
        target = rows[
            KEY + ["execution_time_mean_history_ms"]
        ].copy()

        probabilities = rows[PROBABILITY].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                "Invalid prediction probabilities."
            )

        for policy, configured_floor in (
            ("positive_floor", floor_ms),
            ("historical_median", None),
        ):
            costs = estimate_build_costs(
                target=target,
                history=history,
                zero_cost_floor_ms=configured_floor,
            )

            selection = risk_density_select(
                test_ids=rows["test_id"].tolist(),
                probabilities=probabilities,
                costs_ms=costs[
                    "estimated_cost_ms"
                ].to_numpy(dtype=float),
                rho=RHO,
            )

            output = evaluate_selection(
                rows=rows,
                cost_frame=costs,
                selection=selection,
                project=project,
                build_index=build_index,
                policy=policy,
                floor_ms=floor_ms,
            )

            outputs.append(output)

        print(
            f"  Build {build_index}: "
            f"{len(rows)} tests, "
            f"{rows['execution_time_mean_history_ms'].eq(0).sum()} "
            f"historical zeros"
        )

    return outputs


def main():
    all_results = []

    for project in PROJECTS:
        all_results.extend(run_project(project))

    result = pd.DataFrame(all_results)

    # Every pilot build must have exactly two policies.
    pair_count = result.groupby(
        ["project", "build_index"]
    )["policy"].nunique()

    if not pair_count.eq(2).all():
        raise ValueError(
            "Incomplete paired-policy comparison."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print("\n=== WP5.5 SENSITIVITY PILOT ===")

    summary_columns = [
        "project",
        "build_index",
        "policy",
        "n_tests",
        "n_zero_history",
        "n_selected",
        "tsr",
        "rtr",
        "missed_failures",
        "failure_recall",
    ]

    print(
        result[summary_columns].to_string(
            index=False
        )
    )

    print(f"\nOutput: {OUTPUT}")
    print("Status: PILOT ONLY — NOT FINAL EVIDENCE")
    print("M2-01: NOT FROZEN")


if __name__ == "__main__":
    main()
