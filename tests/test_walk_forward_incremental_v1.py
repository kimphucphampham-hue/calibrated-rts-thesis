"""Deterministic synthetic-only component parity/measurement; no real loader."""
from time import perf_counter
import json
import platform
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.tcp_ci import walk_forward_pre_freeze_v1 as v1
from src.tcp_ci.rq2_causal_feature_materializer import materialize_target_preexecution_features
from src.tcp_ci.rq2_cost_adapter import estimate_build_costs
from src.tcp_ci.walk_forward_causal_state_v1 import CausalState

D = ['project_id','build_index','test_id','execution_time_mean_history_ms']

def state_fixture(n, width, seed):
    rng = np.random.default_rng(seed)
    # Rotating tests, absent/reappearing tests, first appearances, zero costs,
    # equal adjacent timestamps, irregular outcomes, shuffled rows.
    for i in range(1,n+1):
        ids = [f't{x:05d}' for x in range(width) if rng.random()>.15]
        ids += [f'new{i:04d}']
        if not ids: raise AssertionError('empty fixture')
        k=len(ids)
        raw=pd.DataFrame(dict(project_id=v1.PROJECT,build_id=f'b{i:05d}',build_index=i,
            test_id=ids,build_started_at=pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=(i+1)//2),
            failed=rng.integers(0,2,k),execution_time_ms=(np.zeros(k,dtype=int) if i<=2 else rng.integers(0,3000,k))))
        yield raw.sample(frac=1,random_state=i).reset_index(drop=True)

def cost_result(fn):
    try: return fn(),None
    except ValueError as e: return None,str(e)

def expect_error(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Expected ValueError')

def parity(seed):
    s=CausalState(); hist=pd.DataFrame(columns=v1.RAW); rows=0
    for raw in state_fixture(24,25,seed):
        ctx=raw[v1.CTX]; oracle=raw[v1.ORACLE]
        expected=materialize_target_preexecution_features(hist,ctx,feature_scope='F1_ONLY_V1')
        actual=s.prepare(ctx)
        assert_frame_equal(actual,expected,check_exact=True)
        ac,ae=cost_result(lambda:s.estimated_costs(actual))
        rc,re=cost_result(lambda:estimate_build_costs(expected[D],hist,zero_cost_floor_ms=None))
        assert ae==re,(ae,re)
        tampered=actual.copy(); tampered["execution_time_mean_history_ms"]=123456.0
        expect_error(lambda:s.estimated_costs(tampered))
        if ac is not None: assert_frame_equal(ac,rc,check_exact=True)
        # A future context cannot advance state; bad oracle is rejected atomically.
        before=(s.last_index,dict(s.tests),dict(s.durations))
        expect_error(lambda:s.prepare(ctx))
        bad=oracle.copy(); bad.iloc[0,bad.columns.get_loc('failed')]=2
        expect_error(lambda:s.advance(bad))
        assert before==(s.last_index,s.tests,s.durations)
        hist=pd.concat([hist, s.advance(oracle.sample(frac=1,random_state=seed))],ignore_index=True)
        expect_error(lambda:s.advance(oracle))
        rows+=len(actual)
    return rows

def mutation():
    a,b=CausalState(),CausalState()
    blocks=list(state_fixture(5,8,42))
    for raw in blocks[:3]:
        for s in (a,b): s.prepare(raw[v1.CTX]); s.advance(raw[v1.ORACLE])
    raw=blocks[3]; ctx=raw[v1.CTX]; truth=raw[v1.ORACLE]
    changed=truth.copy(); changed['failed']=1-changed.failed; changed['execution_time_ms']+=100000
    pa,pb=a.prepare(ctx),b.prepare(ctx)
    assert_frame_equal(pa,pb,check_exact=True)
    assert_frame_equal(a.estimated_costs(pa),b.estimated_costs(pb),check_exact=True)
    a.advance(truth); b.advance(changed)
    # mutation is allowed to affect the next target only
    na,nb=a.prepare(blocks[4][v1.CTX]),b.prepare(blocks[4][v1.CTX])
    assert not na.equals(nb)
    # bad context must not install pending state
    fresh=CausalState(); bad=blocks[0][v1.CTX].copy(); bad['failed']=0
    expect_error(lambda:fresh.prepare(bad))
    expect_error(lambda:fresh.prepare(blocks[1][v1.CTX]))
    fresh.prepare(blocks[0][v1.CTX]); fresh.advance(blocks[0][v1.ORACLE])
    bad=blocks[1][v1.CTX].copy(); bad['build_started_at']=pd.Timestamp('2019-01-01',tz='UTC')
    expect_error(lambda:fresh.prepare(bad))
    fresh.prepare(blocks[1][v1.CTX])

def integer_boundary():
    s=CausalState(); hist=None
    for i,raw in enumerate(state_fixture(5,1,42),1):
        # One repeated identity, exact integer inputs, cumulative sum > 2**53.
        raw=raw.iloc[:1].copy(); raw['test_id']='large-duration'
        raw['execution_time_ms']=2**53-1-i
        ctx=raw[v1.CTX]; oracle=raw[v1.ORACLE]
        expected=materialize_target_preexecution_features(hist if hist is not None else raw.iloc[:0],ctx,feature_scope='F1_ONLY_V1')
        actual=s.prepare(ctx); assert_frame_equal(actual,expected,check_exact=True)
        if i>1: assert_frame_equal(s.estimated_costs(actual),estimate_build_costs(expected[D],hist),check_exact=True)
        revealed=s.advance(oracle); hist=revealed if hist is None else pd.concat([hist,revealed],ignore_index=True)
    assert s.tests['large-duration'].duration_sum>2**53



def test_incremental_state_exact_parity():
    assert sum(parity(seed) for seed in (7,19,29)) == 1611

def test_incremental_state_mutation_and_fail_closed():
    mutation()

def test_incremental_state_integer_boundary():
    integer_boundary()

"""Read-only parity of complete synthetic decisions and cached fit inputs.
No CORE loader. Same R5 config, same frozen implementations. No performance claim.
"""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from unittest.mock import patch
import importlib.util
import json
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.tcp_ci import walk_forward_pre_freeze_v1 as v1
from src.tcp_ci.rq2_cold_start_orchestrator import select_with_cold_start_fallback, NO_POSITIVE_HISTORY_ERROR
from src.tcp_ci import walk_forward_incremental_v1 as candidate

COLS=v1.KEY+list(v1.F1_ONLY_FEATURES)+['failed']

def fixture():
    spec=importlib.util.spec_from_file_location('r5_fixture',Path('tests/test_walk_forward_pre_freeze_v1.py'))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod._fixture()

def exact_records(a,b):
    # repr is intentional: dataclasses and dictionaries include NaN, whose == is false.
    # Compare the entire structured repr (all nested fields, order, seeds and costs).
    assert type(a) is type(b)
    assert repr(a)==repr(b), 'Structured record mismatch'

def validate_lazy_cost():
    dev,ctx,truth=fixture()
    calls=[]
    original=candidate.CausalState.estimated_costs
    def counted(state,pre):
        calls.append(int(pre.build_index.iloc[0])); return original(state,pre)
    def fake_fit(frame,**kw):
        return SimpleNamespace(feature_columns=v1.F1_ONLY_FEATURES,last_development_build_index=int(frame.build_index.max()),calibration_support_sufficient=False)
    def fake_predict(s,pre):
        result=pre[v1.KEY].copy(); i=int(pre.build_index.iloc[0]); result['p_uncalibrated']=0. if i%2 else .25
        result['p_platt']=np.nan; result['p_isotonic']=np.nan; return result
    # All durations zero: positive-risk branches need the cached no-history error,
    # zero-risk branches never need costs. Exactly one call per even target.
    for mode in ('no_history','normal'):
        calls.clear()
        d=dev.assign(execution_time_ms=0) if mode=='no_history' else dev
        t={i:g.assign(execution_time_ms=0) for i,g in truth.items()} if mode=='no_history' else truth
        with patch.object(candidate.CausalState,'estimated_costs',counted),patch.object(candidate,'predict_from_rq1_snapshot',fake_predict):
            _,choices,_=candidate._run_incremental(d,ctx,lambda i:t[i],h2_enabled=True,snapshot_fitter=fake_fit)
        assert calls==[i for i in sorted(t) if i%2==0],calls
        if mode=='normal': assert any(c.h2_status=='computed' for cs in choices.values() for c in cs)
        print(f'LAZY_COST_AT_MOST_ONCE_PER_TARGET=PASS; MODE={mode}; CALLS=16; TARGETS=32',flush=True)



def test_incremental_lazy_cost():
    validate_lazy_cost()

from functools import partial
def reference_matrix(fits,choices,rows,truth):
    matrix=v1._report_matrix(choices,rows,truth)
    summary,actual=v1._rq2_summary(matrix)
    rq3,blocks=v1._rq3_common_targets(matrix['status'],actual,sorted(truth))
    return dict(**matrix,rq2_summary=summary,rq3=rq3,rq3_blocks=blocks)


def compare_partitioned(expected,spool):
    pieces={name:[] for name in expected}; keys=[]
    for key,tables in spool.iter_reports():
        keys.append(key)
        for name,frame in tables.items():
            if not frame.empty: pieces[name].append(frame)
    for name,ref in expected.items():
        if ref.empty:
            assert not pieces[name],name
            continue
        actual=pd.concat(pieces[name],ignore_index=True)
        assert set(actual.columns)==set(ref.columns), (name,list(actual),list(ref))
        sort=[k for k in ('project','arm','model','probability','rho','cost_policy','observation_regime','baseline','comparison_role','repetition','build_index','block_start','bin','bin_index','bin_lower','bin_upper') if k in ref.columns]
        sort+= [k for k in ref.columns if k not in sort]
        try:
            assert_frame_equal(ref.sort_values(sort).reset_index(drop=True),actual[ref.columns].sort_values(sort).reset_index(drop=True),check_exact=True)
        except AssertionError as e:
            raise AssertionError(f'{name}: {e}') from e
    return len(keys), {k:len(v) for k,v in expected.items()}


def validate(stub=False):
    dev,ctx,truth=fixture()
    if stub:
        dev=dev.assign(execution_time_ms=0)
        truth={i:g.assign(execution_time_ms=0) for i,g in truth.items()}
    def fake_fit(frame,**kw):
        return SimpleNamespace(feature_columns=v1.F1_ONLY_FEATURES,last_development_build_index=int(frame.build_index.max()),calibration_support_sufficient=False)
    def fake_predict(s,pre):
        result=pre[v1.KEY].copy(); i=int(pre.build_index.iloc[0]); result['p_uncalibrated']=0. if i%2 else .25
        result['p_platt']=np.nan; result['p_isotonic']=np.nan; return result
    events=[]; order=[]; captured=[]; fitted=[]
    def reference_fit(frame, **kw):
        captured.append((kw["model_name"],frame[COLS].copy(deep=True).reset_index(drop=True)))
        return fake_fit(frame,**kw) if stub else v1.fit_rq1_snapshot(frame,**kw)
    def cached_fit(frame,**kw):
        name,expected=captured[len(fitted)]; fitted.append(name)
        assert name==kw["model_name"]
        assert_frame_equal(frame[COLS].reset_index(drop=True),expected,check_exact=True)
        return fake_fit(frame,**kw) if stub else v1.fit_rq1_snapshot(frame,timing_sink=events.append,**kw)
    def fixed(i,cs): order.append(('fixed',i))
    def reveal(i):
        assert order[-1]==('fixed',i); order.append(('reveal',i)); return truth[i].copy()
    def ref_run():
        return v1._run(dev,ctx,lambda i:truth[i],h2_enabled=True,snapshot_fitter=reference_fit)
    with candidate._ScoredSpool() as spool:
        def sink(i,cs,rs,oracle):
            assert order[-1]==('reveal',i); order.append(('scored',i))
            exact_records(a[1][i],cs)
            exact_records(tuple(r for r in a[2] if r["build_index"]==i),rs)
            spool.append(i,cs,rs,oracle)
        def candidate_run():
            return candidate._run_incremental(dev,ctx,reveal,on_fixed=fixed,h2_enabled=True,scored_sink=sink,snapshot_fitter=cached_fit)
        if stub:
            with patch.object(candidate,'predict_from_rq1_snapshot',fake_predict), patch.object(v1,'predict_from_rq1_snapshot',fake_predict):
                a=ref_run(); b=candidate_run()
        else:
            a=ref_run(); b=candidate_run()
        assert len(fitted)==len(captured)==33
        assert a[0]==b[0] and len(b[0])==33 and b[1]=={} and b[2]==[]
        assert order==[(tag,i) for i in sorted(truth) for tag in ('fixed','reveal','scored')]
        try: next(spool.iter_reports())
        except ValueError: pass
        else: raise AssertionError('Unsealed report accepted')
        spool.seal(sorted(truth))
        if not stub:
            assert len(events)==33
            e,t=v1._timing_report(b[0],events)
            assert len(e)==33 and len(t)==12
            assert (e.model_fit_ns>0).all() and (e.other_snapshot_ns>=0).all()
            assert e.calibration_status.eq('SUFFICIENT').all()
            assert (e[['calibration_prediction_ns','platt_fit_ns','isotonic_fit_ns']]>0).all().all()
            print('DIRECT_TIMING_EVENTS=PASS; PHYSICAL_FITS=33; LOGICAL_ARMS=12',flush=True)
        expected=reference_matrix(*a,truth)
        n,counts=compare_partitioned(expected,spool)
        assert n==27
        print(f'PARTITIONED_REPORT_EXACT_PARITY=PASS; STUB={stub}; STRATA={n}; COUNTS={counts}; SPOOL_BYTES={spool.size_bytes()}',flush=True)
        p=spool._path
    assert not p.exists()
    print('SPOOL_CLEANUP=PASS',flush=True)


def validate_insufficient_timing():
    dev,ctx,truth=fixture()
    # Force calibration support below minimum while retaining both base-fit classes.
    dev=dev.assign(failed=np.where(dev.build_index<=90,dev.failed,0))
    from src.tcp_ci.build_history_features import compute_history_features
    features=compute_history_features(dev.assign(changed_entity_count=np.nan))
    events=[]; cfg=v1._configs(); name='logistic_regression'
    ref=v1.fit_rq1_snapshot(features,model_name=name,**cfg)
    actual=v1.fit_rq1_snapshot(features,model_name=name,timing_sink=events.append,**cfg)
    assert not ref.calibration_support_sufficient and not actual.calibration_support_sufficient
    e=events[0]
    assert e['model_fit_ns']>0 and e['other_snapshot_ns']>=0
    assert all(e[k]==0 for k in ('calibration_prediction_ns','platt_fit_ns','isotonic_fit_ns'))
    pre=v1.materialize_target_preexecution_features(dev,ctx[ctx.build_index==129],feature_scope='F1_ONLY_V1')
    pre=pre[v1.KEY+list(v1.F1_ONLY_FEATURES)]
    assert_frame_equal(v1.predict_from_rq1_snapshot(ref,pre),v1.predict_from_rq1_snapshot(actual,pre),check_exact=True)
    print('INSUFFICIENT_CALIBRATION_TIMING_AND_PREDICTION_PARITY=PASS',flush=True)




def test_incremental_partitioned_fallback_and_skip():
    validate(stub=True)

def test_incremental_partitioned_three_models():
    validate(stub=False)

def test_incremental_insufficient_calibration_timing():
    validate_insufficient_timing()

def test_incremental_public_api_and_lock():
    import pytest
    from dataclasses import replace
    dev,ctx,truth=fixture()
    calls=[]; keys=[]
    config=v1.WorkingConfig(models=('logistic_regression',))
    def reveal(i):
        calls.append(i); return truth[i].copy()
    with pytest.raises(ValueError):
        candidate.run_partitioned_synthetic(dev,ctx,reveal,report_sink=lambda k,t:None,config=replace(config,execution_authorized=True))
    assert calls==[]
    fits,physical,logical,size=candidate.run_partitioned_synthetic(dev,ctx,reveal,report_sink=lambda k,t:keys.append(k),config=config)
    assert calls==sorted(truth) and len(keys)==9 and len(set(keys))==9
    assert len(fits)==len(physical)==11 and len(logical)==4 and size>0
