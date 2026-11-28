"""Synthetic-only integration tests migrated from WP7.4-R4."""
from types import SimpleNamespace
from time import perf_counter_ns
from unittest.mock import patch
import math
import numpy as np
import pandas as pd
import pytest
from src.tcp_ci import rq1_fitted_snapshot_prototype as snapshot_api
from src.tcp_ci.probability_calibration import PlattCalibrator, IsotonicCalibrator
from src.tcp_ci import walk_forward_pre_freeze_v1 as runner
from src.tcp_ci.walk_forward_pre_freeze_v1 import (
    WorkingConfig,
    Choice,
    PROJECT,
    KEY,
    CTX,
    ORACLE,
    RAW,
    MODELS,
    ARMS,
    RHOS,
    REGIME,
    PROBABILITY_COLUMNS,
    F1_ONLY_FEATURES,
    MODEL_BUILDERS,
    _configs,
    _identity,
    _truth,
    _run,
    _budget_inputs,
    _h2_pair,
    _report_matrix,
    _timing_report,
    _rq2_summary,
    _rq3_common_targets,
    _validate_synthetic_file,
    fit_rq1_snapshot,
    predict_from_rq1_snapshot,
    compute_history_features,
    materialize_target_preexecution_features,
    evaluate_post_selection,
    evaluate_rq1_probability
)

def _fixture():
    # Deterministic, generated in memory. 160 builds -> 128 development / 32 targets.
    rows=[]
    for i in range(1,161):
        for j,t in enumerate(('alpha','beta','gamma','delta')):
            rows.append(dict(project_id=PROJECT,build_id=f'b{i:04d}',build_index=i,
                             test_id=t,build_started_at=(pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=i)).isoformat(),
                             failed=int((i+j)%(4+j)==0),execution_time_ms=11+7*j+i%3))
    data=pd.DataFrame(rows)
    dev=data.loc[data.build_index<=128,RAW].copy()
    contexts=data.loc[data.build_index>128,CTX].copy()
    # Oracle closure is not accessed by the orchestrator except reveal(index).
    truth={int(i):g[ORACLE].copy() for i,g in data.loc[data.build_index>128].groupby('build_index')}
    return dev,contexts,truth


def test_r3_real_lr_h2_before_reveal_and_reporting():
    dev,ctx,truth=_fixture(); seen={}; revealed={}
    def fixed(i,cs):
        assert len(cs)==36
        for c in cs:
            if c.h2_status=='computed':
                p=c.h2_pairing
                assert len(p.random_pair.random_replicates)==30
                assert all(0<=v<=1+1e-12 for v in p.random_budget_utilizations)
                assert set(p.random_pair.reference_selected_test_ids)==set(c.selection.selected_test_ids)
                assert all(x.budget_capped.estimated_cost_ms<=p.random_pair.reference_estimated_cost_ms for x in p.random_pair.random_replicates)
            else: assert c.h2_status.startswith('not_applicable_')
        seen[i]=cs
    def reveal(i):
        assert i in seen and i not in revealed
        revealed[i]=truth[i].sample(frac=1,random_state=i).copy(); return revealed[i]
    fits,cs,rows=_run(dev,ctx,reveal,config=WorkingConfig(models=('logistic_regression',)),on_fixed=fixed,h2_enabled=True)
    result=_report_matrix(cs,rows,revealed)
    assert len(fits)==11 and len(rows)==1152 and len(revealed)==32
    assert len(result['rq1'])==12 and result['rq1'].n_available_builds.eq(32).all()
    assert result['rq1'].rows.eq(128).all()
    # Frozen evaluator omits empty reliability bins; assert coverage, not 10 emitted rows.
    bins=result['bins']; groups=bins.groupby(['arm','model','probability'])
    assert len(groups)==12 and groups['count'].sum().eq(128).all()
    assert groups.size().between(1,10).all() and bins['count'].gt(0).all()
    assert len(result['status'])==1152 and len(result['rq2'])==1152
    h=result['h2_build']; assert not h.empty and set(h.arm)==set(ARMS)
    assert not h.duplicated(['project','arm','model','probability','rho','baseline','comparison_role','repetition','build_index']).any()
    assert (h.loc[h.comparison_role.eq('budget_comparator'),'budget_utilization']<=1+1e-12).all()
    assert result['random_project'].n_repeats.eq(30).all()
    a,b=[next(c for c in cs[129] if c.arm==arm and c.variant=='platt' and c.rho==.95) for arm in ('static','h5')]
    assert a.h2_pairing==b.h2_pairing
    assert result['rq1'].groupby(['arm','model','probability']).size().eq(1).all()


def test_r3_fallback_and_undefined_reporting():
    import pytest
    for reason in ('zero_total_predicted_risk','no_positive_prior_cost_history'):
        sel=SimpleNamespace(selection_mode='fallback_retest_all',fallback_reason=reason)
        assert _h2_pair(sel,None,'logistic_regression','platt',.95,129)==('not_applicable_'+reason,None)
    with pytest.raises(ValueError,match='Unknown fallback'):
        _h2_pair(SimpleNamespace(selection_mode='fallback_retest_all',fallback_reason='bad'),None,'logistic_regression','platt',.95,129)
    metrics,_=evaluate_rq1_probability([0,0],[.2,.3],n_bins=10)
    assert math.isnan(metrics['roc_auc']) and math.isnan(metrics['average_precision'])


def test_r3_skip_status():
    dev,ctx,truth=_fixture(); i=129
    c=Choice('static','logistic_regression','isotonic',.90,128,'INSUFFICIENT_CALIBRATION_SUPPORT',None,(),tuple(truth[i].test_id),'INSUFFICIENT_CALIBRATION_SUPPORT',None)
    r=dict(build_index=i,arm=c.arm,model=c.model,variant=c.variant,rho=c.rho,status=c.status)
    out=_report_matrix({i:(c,)},[r],{i:truth[i]})
    assert out['status'].h2_status.tolist()==['INSUFFICIENT_CALIBRATION_SUPPORT']
    assert out['rq1'].n_skipped_builds.tolist()==[1] and out['rq1'].status.tolist()==['NO_AVAILABLE_PREDICTIONS']
    assert out['h2_build'].empty and out['random_project'].empty and out['rq2'].empty


def test_shared_schedules_and_reveal(monkeypatch):
    from types import SimpleNamespace
    dev,ctx,truth=_fixture(); fitted=[]; predicted=[]; events=[]
    def fake_fit(prefix,*,model_name,**kwargs):
        cutoff=int(prefix.build_index.max()); fitted.append((model_name,cutoff))
        return SimpleNamespace(feature_columns=F1_ONLY_FEATURES,last_development_build_index=cutoff,calibration_support_sufficient=True)
    def fake_predict(s,pre):
        assert set(pre)==set(KEY+list(F1_ONLY_FEATURES)); predicted.append((id(s),int(pre.build_index.iloc[0])))
        result=pre[KEY].copy()
        for col in PROBABILITY_COLUMNS.values(): result[col]=[.1,.2,.3,.4]
        return result
    monkeypatch.setattr(runner,'fit_rq1_snapshot',fake_fit)
    monkeypatch.setattr(runner,'predict_from_rq1_snapshot',fake_predict)
    def fixed(i,c):
        assert len(c)==108; events.append(('fixed',i))
    def reveal(i):
        assert events[-1]==('fixed',i); events.append(('reveal',i)); return truth[i]
    fits,choices,reports=_run(dev,ctx,reveal,on_fixed=fixed)
    assert len(fits)==33 and len(fitted)==33
    assert len(choices)==32 and len(reports)==3456
    assert len(events)==64 and len({(a,b) for a,b in predicted})==len(predicted)
    for m in MODELS:
        assert [i for i,model,a,_ in fits if model==m and a=='initial']==[129]
        for arm,h in ARMS.items():
            assert [i for i,model,a,_ in fits if model==m and a==arm]==([] if h is None else list(range(129+h,161,h)))
    assert all(c.cutoff==128 for c in choices[129])
    assert all(c.cutoff==128 for v in choices.values() for c in v if c.arm=='static')


def test_real_lr_all_arms_and_oracle_mutation():
    dev,ctx,truth=_fixture(); cfg=WorkingConfig(models=('logistic_regression',))
    fits,a,reports=_run(dev,ctx,lambda i:truth[i],config=cfg)
    assert len(fits)==11 and len(reports)==1152
    assert all(r['status']=='computed' for r in reports)
    # Change current/future truth only at final build; every decision must stay fixed.
    def mutated(i):
        v=truth[i].copy()
        if i==160: v['failed']=1-v.failed; v['execution_time_ms']=999
        return v
    fits2,b,_=_run(dev,ctx,mutated,config=cfg)
    assert fits==fits2
    def signature(choices):
        return [(i,c.arm,c.model,c.variant,c.rho,c.cutoff,c.status,c.selection.selected_test_ids if c.selection else (),c.probabilities) for i,cs in choices.items() for c in cs]
    assert signature(a)==signature(b)


def test_unsupported_calibration_and_invalid_future_truth(monkeypatch):
    from types import SimpleNamespace
    import pytest
    dev,ctx,truth=_fixture(); seen=[]
    def fake_fit(prefix,**kw):
        return SimpleNamespace(feature_columns=F1_ONLY_FEATURES,last_development_build_index=int(prefix.build_index.max()),calibration_support_sufficient=False)
    def fake_predict(s,pre):
        result=pre[KEY].copy(); result['p_uncalibrated']=.25; result['p_platt']=np.nan; result['p_isotonic']=np.nan; return result
    monkeypatch.setattr(runner,'fit_rq1_snapshot',fake_fit)
    monkeypatch.setattr(runner,'predict_from_rq1_snapshot',fake_predict)
    cfg=WorkingConfig(models=('logistic_regression',))
    def fixed(i,c):
        assert len(c)==36 and sum(v.selection is None for v in c)==24; seen.append(('fixed',i))
    def reveal(i):
        seen.append(('reveal',i)); v=truth[i].copy()
        if i==131: v['failed']=2
        return v
    with pytest.raises(ValueError,match='Invalid revealed'):
        _run(dev,ctx,reveal,config=cfg,on_fixed=fixed)
    assert seen==[(s,i) for i in (129,130,131) for s in ('fixed','reveal')]


def test_context_and_config_fail_closed():
    import pytest
    dev,ctx,truth=_fixture()
    def never(i): raise AssertionError('Unexpected oracle access')
    for invalid in [ctx.assign(failed=0),ctx.assign(project_id='real'),ctx.assign(build_index=128)]:
        with pytest.raises(ValueError): _run(dev,invalid,never)
    with pytest.raises(ValueError): _run(dev,ctx,never,config=WorkingConfig(execution_authorized=True))


def test_model_dispatch_real_rf_xgb():
    # Actual snapshot/prediction smoke, not a full multi-arm RF/XGB integration run.
    dev,ctx,_=_fixture(); cfg=_configs()
    features=compute_history_features(dev.assign(changed_entity_count=np.nan))
    pre=materialize_target_preexecution_features(dev,ctx.loc[ctx.build_index==129],feature_scope='F1_ONLY_V1')
    assert tuple(MODEL_BUILDERS)==MODELS
    for model in ('random_forest','xgboost'):
        s=fit_rq1_snapshot(features,model_name=model,**cfg)
        assert s.calibration_support_sufficient
        p=predict_from_rq1_snapshot(s,pre[KEY+list(s.feature_columns)])
        assert all(p[col].between(0,1).all() for col in PROBABILITY_COLUMNS.values())


def _timed_fitter(events):
    """Sequential test-only instrumentation, calls exact committed fit helpers.
    Not thread-safe or a production instrumentation interface.
    """
    def fit(prefix, *, model_name, **cfg):
        elapsed={}; splits=[]
        def timed(name,fn):
            def call(*a,**kw):
                t=perf_counter_ns()
                try: return fn(*a,**kw)
                finally: elapsed[name]=elapsed.get(name,0)+perf_counter_ns()-t
            return call
        original_split=snapshot_api.split_development_for_calibration
        def split(*a,**kw):
            result=original_split(*a,**kw); splits.append(result); return result
        t=perf_counter_ns()
        with patch.object(snapshot_api,'fit_probability_model',timed('model_fit_ns',snapshot_api.fit_probability_model)), \
             patch.object(snapshot_api,'positive_probability',timed('calibration_prediction_ns',snapshot_api.positive_probability)), \
             patch.object(snapshot_api,'split_development_for_calibration',split), \
             patch.object(PlattCalibrator,'fit',timed('platt_fit_ns',PlattCalibrator.fit)), \
             patch.object(IsotonicCalibrator,'fit',timed('isotonic_fit_ns',IsotonicCalibrator.fit)):
            result=fit_rq1_snapshot(prefix,model_name=model_name,**cfg)
        total=perf_counter_ns()-t
        assert len(splits)==1 and 'model_fit_ns' in elapsed
        s=splits[0]
        assert bool(s.sufficient)==result.calibration_support_sufficient
        assert all((k in elapsed)==s.sufficient for k in ('calibration_prediction_ns','platt_fit_ns','isotonic_fit_ns'))
        events.append(dict(model=model_name,cutoff=result.last_development_build_index,
            snapshot_total_ns=total,model_fit_ns=elapsed['model_fit_ns'],
            calibration_prediction_ns=elapsed.get('calibration_prediction_ns',0),
            platt_fit_ns=elapsed.get('platt_fit_ns',0),isotonic_fit_ns=elapsed.get('isotonic_fit_ns',0),
            other_snapshot_ns=total-sum(elapsed.values()),
            model_fit_builds=len(s.model_fit_builds),calibration_builds=len(s.calibration_builds),
            expanded_by_builds=s.expanded_by_builds,calibration_status=s.status,
            positive_rows=s.support.positive_rows,negative_rows=s.support.negative_rows,
            failing_builds=s.support.failing_builds,passing_builds=s.support.passing_builds))
        return result
    return fit


def test_r4_full_three_model_matrix_and_timing():
    dev,ctx,truth=_fixture(); events=[]; fixed={}; revealed={}
    def on_fixed(i,cs):
        assert len(cs)==108 and i not in revealed
        for c in cs:
            assert c.cutoff<i
            if c.h2_pairing:
                pair=c.h2_pairing; assert len(pair.random_pair.random_replicates)==30
                assert all(0<=x<=1+1e-12 for x in pair.random_budget_utilizations)
        fixed[i]=cs
    def reveal(i):
        assert i in fixed and i not in revealed
        revealed[i]=truth[i].sample(frac=1,random_state=i).copy(); return revealed[i]
    fits,choices,rows=_run(dev,ctx,reveal,on_fixed=on_fixed,h2_enabled=True,snapshot_fitter=_timed_fitter(events))
    assert len(fits)==33 and len(rows)==3456 and len(revealed)==32 and len(events)==33
    assert all(e['other_snapshot_ns']>=0 and e['model_fit_ns']>0 and e['calibration_status']=='SUFFICIENT' and e['platt_fit_ns']>0 and e['isotonic_fit_ns']>0 for e in events)
    assert all(e['model_fit_builds']>=100 and e['calibration_builds']>=20 for e in events)
    timing,overhead=_timing_report(fits,events)
    assert len(timing)==33 and len(overhead)==12
    assert overhead.groupby('arm').n_refresh.first().to_dict()==dict(static=0,h5=6,h10=3,h20=1)
    matrix=_report_matrix(choices,rows,revealed)
    assert len(matrix['rq1'])==36 and matrix['rq1'].n_available_builds.eq(32).all()
    summary,actual=_rq2_summary(matrix)
    assert len(summary)==108 and summary.n_scheduled_builds.eq(32).all() and summary.n_computed_builds.eq(32).all()
    assert summary.n_tests.eq(128).all() and summary.n_skipped_builds.eq(0).all()
    comp,blocks=_rq3_common_targets(matrix['status'],actual,sorted(choices))
    assert len(comp)==81 and len(blocks)==162 and comp.n_common.eq(32).all()
    assert comp.fr_block_sample_variance_static.isna().all() and comp.fr_block_sample_variance_periodic.isna().all()
    assert blocks.groupby(['project','model','probability','rho','arm']).block_scheduled.sum().eq(32).all()
    assert matrix['random_project'].n_repeats.eq(30).all()
    for model in MODELS:
        c=[x for x in choices[129] if x.model==model and x.variant=='platt' and x.rho==.95]
        assert len(c)==4 and all(x.probabilities==c[0].probabilities and x.h2_pairing==c[0].h2_pairing for x in c)


def test_r4_skip_fallback_common_coverage():
    from src.tcp_ci.rq2_cold_start_orchestrator import ColdStartSelectionResult
    dev,ctx,truth=_fixture(); i=129; ids=tuple(truth[i].test_id)
    realized=_truth(ctx[ctx.build_index==i],truth[i]); choices=[]; reports=[]
    for arm in ARMS:
        skip=arm=='h5'; reason='zero_total_predicted_risk' if arm=='static' else 'no_positive_prior_cost_history'
        sel=None if skip else ColdStartSelectionResult(PROJECT,str(i),ids,'fallback_retest_all',reason,len(ids),None,np.nan if arm=='static' else 1.)
        c=Choice(arm,'logistic_regression','platt',.95,128,'INSUFFICIENT_CALIBRATION_SUPPORT' if skip else 'computed',sel,() if skip else (0.,)*4,ids,'INSUFFICIENT_CALIBRATION_SUPPORT' if skip else 'not_applicable_'+reason,None)
        choices.append(c); row=dict(build_index=i,arm=arm,model=c.model,variant=c.variant,rho=c.rho,status=c.status)
        if sel is not None: row['metrics']=evaluate_post_selection(realized,sel,rho=c.rho,model=c.model,probability=PROBABILITY_COLUMNS[c.variant],cost_policy='historical_median',baseline='risk_density',observation_regime=REGIME)
        reports.append(row)
    matrix=_report_matrix({i:tuple(choices)},reports,{i:truth[i]})
    summary,actual=_rq2_summary(matrix); s=summary.set_index('arm')
    assert s.loc['static','n_zero_risk_fallback']==1 and s.loc['h10','n_no_prior_cost_fallback']==1
    assert s.loc['h5','n_skipped_builds']==1 and np.isnan(s.loc['h5','tsr_micro'])
    assert s.loc['static','tsr_micro']==1 and s.n_h2_paired_builds.eq(0).all()
    comp,_=_rq3_common_targets(matrix['status'],actual,[i]); z=comp.set_index('arm')
    assert z.loc['h5','n_common']==0 and np.isnan(z.loc['h5','fr_delta'])
    assert z.loc['h10','n_common']==1 and z.loc['h10','tsr_delta']==0


def test_r4_fixed_blocks_and_undefined_fr():
    status=[]; actual=[]; schedule=list(range(129,171))
    for j,i in enumerate(schedule):
        for arm in ARMS:
            meta=dict(project=PROJECT,model='logistic_regression',probability='platt',rho=.95,arm=arm,build_index=i)
            skip=arm=='h5' and j==0
            status.append(dict(**meta,status='INSUFFICIENT_CALIBRATION_SUPPORT' if skip else 'computed'))
            if not skip:
                failures=int(j<40); captured=int(failures and (arm!='static' or j<20))
                actual.append(dict(**meta,total_failures=failures,captured_failures=captured,observed_full_ms=100.,observed_selected_ms=60.,n_tests=4,n_selected=2))
    comp,blocks=_rq3_common_targets(pd.DataFrame(status),pd.DataFrame(actual),schedule)
    a=comp.set_index('arm'); assert a.loc['h5','n_common']==41
    assert a.loc['h10','fr_decay_static']<0 and a.loc['h10','fr_decay_periodic']==0
    assert a.loc['h10','fr_block_sample_variance_static']==.5 and a.loc['h10','fr_block_sample_variance_periodic']==0
    assert a.n_paired_defined_complete_blocks.eq(2).all()
    tail=blocks[~blocks.complete_block]; assert len(tail)==3 and tail.fr_static.isna().all() and not tail.variance_included.any()


def test_r5_versioned_config_locked(monkeypatch):
    observed = _validate_synthetic_file()
    assert observed['execution_authorized'] is False
    for patch_fields in ({'execution_authorized': True}, {'execution_authorized': 0}, {'scope': 'CORE'}, {'rho': [.95]}, {'method_freeze': 'APPROVED'}):
        monkeypatch.setattr(runner, 'load_yaml', lambda path: dict(observed, **patch_fields))
        with pytest.raises(ValueError, match='Unsupported versioned synthetic config'):
            _validate_synthetic_file()


def test_r5_no_oracle_on_unapproved_file(monkeypatch):
    dev, ctx, _ = _fixture()
    observed = _validate_synthetic_file()
    monkeypatch.setattr(runner, 'load_yaml', lambda path: dict(observed, execution_authorized=True))
    def never(index):
        raise AssertionError('Oracle must not be accessed')
    with pytest.raises(ValueError, match='Unsupported versioned synthetic config'):
        _run(dev, ctx, never)
