"""M2-01 candidate zero-cost floor tests.

Prototype validation — not an approved M2 policy.
"""

import pandas as pd
import pytest

from src.tcp_ci.rq2_cost_adapter import estimate_build_costs


def make_target():
    return pd.DataFrame({
        "project_id": ["project_a"] * 3,
        "build_id": ["b5"] * 3,
        "build_index": [5] * 3,
        "test_id": ["positive", "missing", "zero"],
        "execution_time_mean_history_ms": [
            15.0,
            float("nan"),
            0.0,
        ],
    })


def make_history():
    return pd.DataFrame({
        "project_id": ["project_a"] * 3,
        "build_id": ["b1", "b2", "b6"],
        "build_index": [1, 2, 6],
        "test_id": ["x", "y", "z"],
        "execution_time_ms": [10.0, 30.0, 100000.0],
    })


def test_zero_floor_preserves_positive_and_missing_policy():
    result = estimate_build_costs(
        target=make_target(),
        history=make_history(),
        zero_cost_floor_ms=1.0,
    )

    by_id = result.set_index("test_id")[
        "estimated_cost_ms"
    ]

    assert by_id["positive"] == pytest.approx(15.0)
    assert by_id["missing"] == pytest.approx(20.0)
    assert by_id["zero"] == pytest.approx(1.0)


def test_zero_floor_is_configurable():
    result = estimate_build_costs(
        target=make_target(),
        history=make_history(),
        zero_cost_floor_ms=1000.0,
    )

    by_id = result.set_index("test_id")[
        "estimated_cost_ms"
    ]

    assert by_id["zero"] == pytest.approx(1000.0)
    assert by_id["missing"] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "invalid_floor",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_invalid_zero_floor_is_rejected(invalid_floor):
    with pytest.raises(ValueError):
        estimate_build_costs(
            target=make_target(),
            history=make_history(),
            zero_cost_floor_ms=invalid_floor,
        )

def test_zero_floor_does_not_require_positive_history():
    target = make_target()
    target = target.loc[
        target["test_id"].eq("zero")
    ].copy()

    history = make_history().iloc[0:0].copy()

    result = estimate_build_costs(
        target=target,
        history=history,
        zero_cost_floor_ms=1000.0,
    )

    assert result["estimated_cost_ms"].iloc[0] == pytest.approx(
        1000.0
    )