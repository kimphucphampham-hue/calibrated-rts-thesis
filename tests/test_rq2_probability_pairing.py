
"""RQ2 calibrated vs uncalibrated paired-selection checks.

Implementation validation only — not final longitudinal evidence.
"""

import math

import pytest

from src.tcp_ci.rq2_selector import risk_density_select


TEST_IDS = ("a", "b", "c")
COSTS = (1.0, 2.0, 3.0)
P_PLATT = (0.7, 0.2, 0.1)
P_UNCALIBRATED = (0.1, 0.8, 0.1)
RHO = 0.8


def select(probabilities):
    return risk_density_select(
        test_ids=TEST_IDS,
        probabilities=probabilities,
        costs_ms=COSTS,
        rho=RHO,
    )


def test_probability_source_changes_selection():
    calibrated = select(P_PLATT)
    uncalibrated = select(P_UNCALIBRATED)

    assert calibrated.selected_test_ids == ("a", "b")
    assert uncalibrated.selected_test_ids == ("b",)


@pytest.mark.parametrize(
    "probabilities",
    [P_PLATT, P_UNCALIBRATED],
)
def test_each_condition_meets_its_own_risk_bound(
    probabilities,
):
    result = select(probabilities)

    assert result.retained_risk >= (
        RHO * result.total_risk
        - 1e-12 * result.total_risk
    )


@pytest.mark.parametrize(
    "probabilities",
    [P_PLATT, P_UNCALIBRATED],
)
def test_estimated_cost_matches_selected_ids(
    probabilities,
):
    result = select(probabilities)
    costs_by_id = dict(zip(TEST_IDS, COSTS))

    expected = math.fsum(
        costs_by_id[test_id]
        for test_id in result.selected_test_ids
    )

    assert result.estimated_cost_ms == pytest.approx(
        expected
    )


def test_paired_selection_is_deterministic():
    assert select(P_PLATT) == select(P_PLATT)
    assert select(P_UNCALIBRATED) == select(
        P_UNCALIBRATED
    )


def test_paired_selection_independent_of_row_order():
    original = select(P_PLATT)

    reordered = risk_density_select(
        test_ids=("c", "a", "b"),
        probabilities=(0.1, 0.7, 0.2),
        costs_ms=(3.0, 1.0, 2.0),
        rho=RHO,
    )

    assert original == reordered


@pytest.mark.parametrize(
    "bad_probability",
    [float("nan"), float("inf"), -0.1, 1.1],
)
def test_invalid_probabilities_are_rejected(
    bad_probability,
):
    with pytest.raises(ValueError):
        select((0.7, bad_probability, 0.1))
