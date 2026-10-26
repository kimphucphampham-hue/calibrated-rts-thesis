"""Extended RQ2 historical-cost sensitivity pilot.

Phase 11 holdout diagnostic only.
Not final longitudinal evidence.
M2-01 remains NOT FROZEN.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.tcp_ci.rq2_cost_adapter import estimate_build_costs
from src.tcp_ci.rq2_selector import risk_density_select

from scripts.run_rq2_cost_sensitivity_pilot import (
    load_project,
    KEY,
    HISTORY_COLUMNS,
    PROBABILITY,
)


PROJECTS = (
    "datasets__facebook__buck",
    "datasets__Angel-ML__angel",
)

RHOS = (0.90, 0.95, 0.99)
MAX_BUILDS = 12

OUTPUT_ROOT = Path("results/core/rq2")

OUTPUT_DETAIL = (
    OUTPUT_ROOT / "cost_sensitivity_extended.csv"
)

OUTPUT_PAIRS = (
    OUTPUT_ROOT / "cost_sensitivity_extended_paired.csv"
)

OUTPUT_SKIPPED = (
    OUTPUT_ROOT / "cost_sensitivity_extended_skipped.csv"
)


def choose_builds(aligned):
    """Choose evenly spaced chronological build positions.

    Selection is independent of failures, costs, and
    observed sensitivity outcomes.
    """
    builds = sorted(
        aligned["build_index"].unique().tolist()
    )

    if not builds:
        raise ValueError("No evaluation builds.")

    count = min(MAX_BUILDS, len(builds))

    positions = np.rint(
        np.linspace(
            0,
            len(builds) - 1,
            num=count,
        )
    ).astype(int)

    return [builds[i] for i in positions]


def make_result(
    project,
    build_index,
    rows,
    costs,
    selection,
    rho,
    policy,
    floor_ms,
):
    selected_ids = {
        str(value)
        for value in selection.selected_test_ids
    }

    available_ids = set(
        rows["test_id"].astype(str)
    )

    if not selected_ids.issubset(available_ids):
        raise ValueError(
            "Selector returned unknown test IDs."
        )

    if not rows.index.equals(costs.index):
        raise ValueError(
            "Cost and prediction row alignment mismatch."
        )

    selected_mask = rows["test_id"].astype(str).isin(
        selected_ids
    )

    if int(selected_mask.sum()) != len(selected_ids):
        raise ValueError(
            "Selected test count mismatch."
        )

    probabilities = rows[PROBABILITY].astype(float)
    failures = rows["failed"].astype(int)
    durations = rows["execution_time_ms"].astype(float)

    estimated = costs["estimated_cost_ms"].astype(float)

    total_risk = float(probabilities.sum())
    retained_risk = float(
        probabilities.loc[selected_mask].sum()
    )

    if total_risk <= 0:
        raise ValueError(
            "Cannot evaluate zero predicted risk."
        )

    coverage = retained_risk / total_risk

    if coverage < rho - 1e-9:
        raise ValueError(
            "Risk coverage constraint violated."
        )

    total_failures = int(failures.sum())

    captured_failures = int(
        failures.loc[selected_mask].sum()
    )

    missed_failures = (
        total_failures - captured_failures
    )

    total_runtime = float(durations.sum())

    selected_runtime = float(
        durations.loc[selected_mask].sum()
    )

    historical_zero_mask = rows[
        "execution_time_mean_history_ms"
    ].eq(0)

    return {
        "evidence_scope": "phase11_extended_pilot",
        "project": project,
        "build_index": int(build_index),
        "build_id": str(rows["build_id"].iloc[0]),
        "model": "logistic_regression",
        "probability": PROBABILITY,
        "rho": float(rho),
        "policy": policy,
        "floor_ms": float(floor_ms),
        "n_tests": len(rows),
        "n_zero_history": int(
            historical_zero_mask.sum()
        ),
        "n_selected_zero_history": int(
            (
                selected_mask
                & historical_zero_mask
            ).sum()
        ),
        "n_selected": int(selected_mask.sum()),
        "tsr": float(selected_mask.mean()),
        "estimated_selected_ms": float(
            estimated.loc[selected_mask].sum()
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
        "missed_failures": missed_failures,
        "failure_recall": (
            captured_failures / total_failures
            if total_failures > 0
            else np.nan
        ),
        "missed_failure_rate": (
            missed_failures / total_failures
            if total_failures > 0
            else np.nan
        ),
        "retained_risk_ratio": coverage,
        "selected_test_ids": "|".join(
            sorted(selected_ids)
        ),
    }


def run_project(project):
    train, test, aligned, floor_ms = load_project(
        project
    )

    builds = choose_builds(aligned)

    print(f"\nPROJECT: {project}")
    print(f"Training-derived floor: {floor_ms:g} ms")
    print(f"Chronological sample: {len(builds)} builds")
    print(f"Build indices: {builds}")

    all_history = pd.concat(
        [
            train[HISTORY_COLUMNS],
            test[HISTORY_COLUMNS],
        ],
        ignore_index=True,
    )

    results = []
    skipped = []

    for build_index in builds:
        rows = aligned.loc[
            aligned["build_index"].eq(build_index)
        ].copy()

        rows = rows.sort_values(
            "test_id",
            kind="stable",
        )

        probabilities = rows[
            PROBABILITY
        ].to_numpy(dtype=float)

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                f"Invalid probabilities: {project}, "
                f"build {build_index}"
            )

        if float(probabilities.sum()) <= 0:
            skipped.append({
                "project": project,
                "build_index": int(build_index),
                "reason": "zero_total_predicted_risk",
            })

            print(
                f"  Build {build_index}: "
                "SKIPPED (M2-02 unresolved)"
            )
            continue

        history = all_history.loc[
            all_history["build_index"].lt(
                build_index
            )
        ]

        if not (
            history["build_index"] < build_index
        ).all():
            raise ValueError(
                "Temporal history violation."
            )

        target = rows[
            KEY + ["execution_time_mean_history_ms"]
        ].copy()

        history_input = history[HISTORY_COLUMNS].copy()

        # Cost estimation is performed once per policy
        # per build, not once per risk threshold.
        cost_frames = {
            "positive_floor": estimate_build_costs(
                target=target,
                history=history_input,
                zero_cost_floor_ms=floor_ms,
            ),
            "historical_median": estimate_build_costs(
                target=target,
                history=history_input,
            ),
        }

        for rho in RHOS:
            for policy, costs in cost_frames.items():
                selection = risk_density_select(
                    test_ids=rows["test_id"].tolist(),
                    probabilities=probabilities,
                    costs_ms=costs[
                        "estimated_cost_ms"
                    ].to_numpy(dtype=float),
                    rho=rho,
                )

                result = make_result(
                    project=project,
                    build_index=build_index,
                    rows=rows,
                    costs=costs,
                    selection=selection,
                    rho=rho,
                    policy=policy,
                    floor_ms=floor_ms,
                )

                results.append(result)

        print(
            f"  Build {build_index}: "
            f"{len(rows)} tests, "
            f"{target['execution_time_mean_history_ms'].eq(0).sum()} "
            "historical zeros"
        )

    return results, skipped


def build_pairs(detail):
    paired_rows = []

    for key, group in detail.groupby(
        ["project", "build_index", "rho"],
        sort=True,
    ):
        indexed = group.set_index("policy")

        if (
            len(group) != 2
            or set(indexed.index)
            != {"positive_floor", "historical_median"}
        ):
            raise ValueError(
                f"Incomplete paired comparison: {key}"
            )

        floor = indexed.loc["positive_floor"]
        median = indexed.loc["historical_median"]

        floor_ids = set(
            floor["selected_test_ids"].split("|")
        )

        median_ids = set(
            median["selected_test_ids"].split("|")
        )

        union = floor_ids | median_ids

        jaccard = (
            len(floor_ids & median_ids) / len(union)
            if union
            else 1.0
        )

        paired_rows.append({
            "project": key[0],
            "build_index": key[1],
            "rho": key[2],
            "n_zero_history": floor[
                "n_zero_history"
            ],
            "total_failures": floor[
                "total_failures"
            ],
            "same_selected_ids": (
                floor_ids == median_ids
            ),
            "jaccard": jaccard,
            "floor_n_selected": floor[
                "n_selected"
            ],
            "median_n_selected": median[
                "n_selected"
            ],
            "floor_tsr": floor["tsr"],
            "median_tsr": median["tsr"],
            "floor_rtr": floor["rtr"],
            "median_rtr": median["rtr"],
            "floor_missed_failures": floor[
                "missed_failures"
            ],
            "median_missed_failures": median[
                "missed_failures"
            ],
            "estimated_cost_delta_ms": (
                floor["estimated_selected_ms"]
                - median["estimated_selected_ms"]
            ),
        })

    return pd.DataFrame(paired_rows)


def main():
    results = []
    skipped = []

    for project in PROJECTS:
        project_results, project_skipped = (
            run_project(project)
        )

        results.extend(project_results)
        skipped.extend(project_skipped)

    if not results:
        raise ValueError(
            "No completed sensitivity comparisons."
        )

    detail = pd.DataFrame(results)
    pairs = build_pairs(detail)
    skipped_df = pd.DataFrame(
        skipped,
        columns=[
            "project",
            "build_index",
            "reason",
        ],
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        OUTPUT_DETAIL,
        index=False,
    )

    pairs.to_csv(
        OUTPUT_PAIRS,
        index=False,
    )

    skipped_df.to_csv(
        OUTPUT_SKIPPED,
        index=False,
    )

    print("\n=== WP5.5 EXTENDED SUMMARY ===")

    for (project, rho), group in pairs.groupby(
        ["project", "rho"],
        sort=True,
    ):
        changed_ids = int(
            (~group["same_selected_ids"]).sum()
        )

        changed_estimated = int(
            group["estimated_cost_delta_ms"]
            .abs()
            .gt(1e-8)
            .sum()
        )

        with_failures = int(
            group["total_failures"].gt(0).sum()
        )

        print(
            f"{project} | rho={rho:.2f} | "
            f"pairs={len(group)} | "
            f"changed_ids={changed_ids} | "
            f"changed_est_cost={changed_estimated} | "
            f"failure_builds={with_failures}"
        )

    print("\nCompleted paired comparisons:", len(pairs))
    print("Skipped builds:", len(skipped_df))

    print("\nOutputs:")
    print(OUTPUT_DETAIL)
    print(OUTPUT_PAIRS)
    print(OUTPUT_SKIPPED)

    print("\nPILOT ONLY — NOT FINAL EVIDENCE")
    print("M2-01: NOT FROZEN")


if __name__ == "__main__":
    main()
