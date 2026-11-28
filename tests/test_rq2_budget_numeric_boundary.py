"""Synthetic numeric boundary regressions for the proposed strict budget fix."""
from math import fsum
import numpy as np
from src.tcp_ci.rq2_baselines import random_matched_budget
from src.tcp_ci.rq2_h2_budget_pairing import _pack_ranked_ids, pair_h2_budget_comparators_for_build

IDS=('alpha','beta','delta','gamma')
COSTS=(12.00632911392405,19.00632911392405,33.00632911392405,26.00632911392405)
CAP=45.0126582278481
SEED=8834034738617794647


def test_r4_budget_rounding_reproducer():
    assert fsum([COSTS[1],COSTS[3]])==CAP
    assert fsum([COSTS[0],COSTS[2]])>CAP
    assert COSTS[2]<=CAP-COSTS[0]  # old subtraction guard accepts an unaffordable candidate
    r=random_matched_budget(IDS,COSTS,CAP,SEED)
    assert r.selected_test_ids==('alpha','gamma')
    assert r.estimated_cost_ms==fsum(COSTS[IDS.index(t)] for t in r.selected_test_ids)<=CAP


def test_r4_historical_rounding_reproducer():
    r=_pack_ranked_ids(('alpha','delta','gamma','beta'),dict(zip(IDS,COSTS)),CAP,'test')
    assert r.selected_test_ids==('alpha','gamma')
    assert r.estimated_cost_ms<=CAP
    # Equality is valid: same fsum-derived cap, no artificial underspend.
    r=_pack_ranked_ids(('beta','gamma','alpha','delta'),dict(zip(IDS,COSTS)),CAP,'test')
    assert r.selected_test_ids==('beta','gamma') and r.estimated_cost_ms==CAP


def test_r4_greedy_parity_and_row_order():
    # Independent reference uses full fsum for each tentative addition.
    rng=np.random.default_rng(739)
    for _ in range(40):
        ids=tuple(f't{i:02d}' for i in range(12)); costs=tuple(float(v) for v in rng.uniform(.001,500,12))
        cap=fsum(costs[i] for i in rng.choice(len(ids),size=4,replace=False))
        for seed in (0,1,7,SEED):
            chosen=[]
            for i in np.random.default_rng(seed).permutation(len(ids)):
                if fsum([costs[j] for j in chosen]+[costs[i]])<=cap: chosen.append(i)
            expected=tuple(ids[j] for j in sorted(chosen))
            r=random_matched_budget(ids,costs,cap,seed)
            assert r.selected_test_ids==expected and r.estimated_cost_ms<=cap
            assert r==random_matched_budget(tuple(reversed(ids)),tuple(reversed(costs)),cap,seed)


def test_r4_h2_full_pairing_strict_cap():
    key=('model=logistic_regression','probability=platt','rho=0.95','cost_policy=historical_median','observation=retrospective_full_information')
    p=pair_h2_budget_comparators_for_build(project_id='synthetic-numeric',build_index=159,condition_key=key,test_ids=IDS,estimated_costs_ms=COSTS,historical_failure_rates=(1,.2,.9,.5),reference_selected_test_ids=('beta','gamma'))
    assert p.random_pair.reference_estimated_cost_ms==CAP
    assert len(p.random_pair.random_replicates)==30
    for r in [p.failure_history_budget,p.history_duration_budget]+[r.budget_capped for r in p.random_pair.random_replicates]:
        assert r.estimated_cost_ms==fsum(COSTS[IDS.index(t)] for t in r.selected_test_ids)<=CAP
