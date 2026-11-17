"""Pre-freeze M2-04/M2-05 one-build Random baseline pairing (NO ORACLE INPUT).

This is a selection-time integration candidate, not a final RQ2 evaluation
runner. Callers must verify the temporal provenance of supplied cost estimates.
The comparison pairs Random with the reference selector's selected-test COUNT,
not its estimated or realized cost. Budget-capped Random is separate sensitivity.
Neither real CI observability nor historical-rate baselines are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import fsum, isfinite
from numbers import Integral
from typing import Optional, Sequence, Tuple

from src.tcp_ci.rq2_baselines import (
    BaselineResult,
    random_matched_budget,
    random_matched_count,
    retest_all,
)

SEED_SCHEME_VERSION = "rq2-m2-05-seed-v1"
WORKING_BASE_SEED = 20261113
WORKING_REPETITIONS = 30
_CONDITION_PREFIXES = (
    "model=", "probability=", "rho=", "cost_policy=", "observation="
)


@dataclass(frozen=True)
class RandomPairedReplicate:
    repetition: int
    matched_count_seed: int
    matched_count: BaselineResult
    budget_seed: Optional[int]
    budget_capped: Optional[BaselineResult]


@dataclass(frozen=True)
class RandomPairedBuild:
    project_id: str
    build_index: int
    condition_key: Tuple[str, ...]
    reference_selected_test_ids: Tuple[str, ...]
    matched_count_k: int
    reference_estimated_cost_ms: float
    full_retest: BaselineResult
    random_replicates: Tuple[RandomPairedReplicate, ...]


def _canonical_identity(project_id, build_index, condition_key, base_seed):
    if not isinstance(project_id, str) or not project_id.strip() or project_id != project_id.strip():
        raise ValueError("project_id must be a nonempty canonical string.")
    if isinstance(build_index, bool) or not isinstance(build_index, Integral) or build_index < 1:
        raise ValueError("build_index must be a positive integer.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    if not isinstance(condition_key, tuple) or len(condition_key) != len(_CONDITION_PREFIXES):
        raise ValueError("condition_key must be a five-field canonical tuple.")
    for value, prefix in zip(condition_key, _CONDITION_PREFIXES):
        if (not isinstance(value, str) or not value.startswith(prefix)
                or len(value) <= len(prefix) or value != value.strip()
                or not value[len(prefix):].strip()):
            raise ValueError("Invalid canonical condition_key field: " + prefix)
    return str(project_id), int(build_index), condition_key, int(base_seed)


def derive_random_seed(
    *, project_id: str, build_index: int, condition_key: Tuple[str, ...],
    repetition: int, comparator: str, base_seed: int = WORKING_BASE_SEED,
) -> int:
    """Versioned, delimiter-safe SHA-256 first-8-byte unsigned big-endian seed.

    Canonical UTF-8 JSON array order:
      [version, base_seed, project, build_index, model, probability, rho,
       cost_policy, observation, comparator, repetition].
    Python's randomized hash() is never used.
    """
    project, build, condition, seed = _canonical_identity(
        project_id, build_index, condition_key, base_seed
    )
    if isinstance(repetition, bool) or not isinstance(repetition, Integral) or not 0 <= repetition < WORKING_REPETITIONS:
        raise ValueError("repetition must be an integer in [0, 29].")
    if comparator not in ("random_matched_count", "random_matched_budget"):
        raise ValueError("Unsupported random baseline comparator.")
    payload = [
        SEED_SCHEME_VERSION, str(seed), project, str(build),
        *condition, comparator, str(int(repetition)),
    ]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:8], "big")


def pair_random_baselines_for_build(
    *, project_id: str, build_index: int, condition_key: Tuple[str, ...],
    test_ids: Sequence[str], estimated_costs_ms: Sequence[float],
    reference_selected_test_ids: Sequence[str],
    include_budget_sensitivity: bool = False,
    base_seed: int = WORKING_BASE_SEED,
) -> RandomPairedBuild:
    """Pair thirty seeded Random selections with one *already fixed* selection.

    No target outcomes, target oracle, realized target durations, probabilities,
    model fitting or historical-rate estimates are accepted by this function.
    Costs MUST have been computed causally before calling it. This function
    does not prove that upstream provenance. Cold-start without positive cost
    estimates must be reported separately, not assigned artificial costs.
    """
    project, build, condition, seed = _canonical_identity(
        project_id, build_index, condition_key, base_seed
    )
    if not isinstance(include_budget_sensitivity, bool):
        raise ValueError("include_budget_sensitivity must be boolean.")

    # The existing baseline module validates uniqueness, cost positivity,
    # canonical selection order and stable arithmetic. Reuse it unchanged.
    ids = tuple(test_ids)
    costs = tuple(estimated_costs_ms)
    full = retest_all(ids, costs)
    if not isinstance(reference_selected_test_ids, (list, tuple)):
        raise ValueError("reference_selected_test_ids must be a sequence.")
    reference = tuple(reference_selected_test_ids)
    if (not reference or any(not isinstance(x, str) for x in reference)
            or len(set(reference)) != len(reference)
            or not set(reference).issubset(set(ids))):
        raise ValueError("Reference selection must be a nonempty unique candidate subset.")
    # Treat reference order as irrelevant to comparison identity.
    reference = tuple(sorted(reference))
    lookup = dict(zip(ids, costs))
    budget = float(fsum(float(lookup[test_id]) for test_id in reference))
    if not isfinite(budget) or budget <= 0:
        raise ValueError("Reference estimated cost must be finite and positive.")
    k = len(reference)

    repeats = []
    for rep in range(WORKING_REPETITIONS):
        count_seed = derive_random_seed(
            project_id=project, build_index=build, condition_key=condition,
            repetition=rep, comparator="random_matched_count", base_seed=seed,
        )
        matched = random_matched_count(ids, costs, k=k, seed=count_seed)
        if len(matched.selected_test_ids) != k:
            raise RuntimeError("Random baseline violated matched-count pairing.")
        budget_seed = None
        budget_baseline = None
        if include_budget_sensitivity:
            budget_seed = derive_random_seed(
                project_id=project, build_index=build, condition_key=condition,
                repetition=rep, comparator="random_matched_budget", base_seed=seed,
            )
            budget_baseline = random_matched_budget(
                ids, costs, budget_ms=budget, seed=budget_seed
            )
            if budget_baseline.estimated_cost_ms > budget:
                raise RuntimeError("Random sensitivity exceeded estimated budget.")
        repeats.append(RandomPairedReplicate(
            repetition=rep, matched_count_seed=count_seed, matched_count=matched,
            budget_seed=budget_seed, budget_capped=budget_baseline,
        ))

    return RandomPairedBuild(
        project_id=project, build_index=build, condition_key=condition,
        reference_selected_test_ids=reference, matched_count_k=k,
        reference_estimated_cost_ms=budget, full_retest=full,
        random_replicates=tuple(repeats),
    )
