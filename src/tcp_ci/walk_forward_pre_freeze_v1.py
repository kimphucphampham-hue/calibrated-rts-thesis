"""Version 1 pre-freeze synthetic integration only.
Extracted from the reviewed WP7.4-R4 candidate. No real dataset loader or
approved experimental entrypoint. Research configuration remains locked.
"""
from pathlib import Path


from dataclasses import dataclass


from types import SimpleNamespace


from src.tcp_ci.rq2_cost_adapter import estimate_build_costs


from src.tcp_ci.rq2_h2_budget_pairing import pair_h2_budget_comparators_for_build


from src.tcp_ci.rq2_h2_postselection_reporting import report_h2_prototype_steps


from src.tcp_ci.rq1_evaluator import evaluate_rq1_probability


import math


import numpy as np


import pandas as pd


from src.tcp_ci.rq1_pipeline import load_yaml, PROBABILITY_COLUMNS, F1_ONLY_FEATURES, MODEL_BUILDERS


from src.tcp_ci.rq1_fitted_snapshot_prototype import fit_rq1_snapshot, predict_from_rq1_snapshot


from src.tcp_ci.rq2_causal_feature_materializer import materialize_target_preexecution_features


from src.tcp_ci.build_history_features import compute_history_features


from src.tcp_ci.rq2_cold_start_orchestrator import select_with_cold_start_fallback


from src.tcp_ci.rq2_cold_start_reporting import evaluate_post_selection


from src.tcp_ci.rq1_rq2_multibuild_prototype import _check_frame


from src.tcp_ci.rq2_cold_start_reporting import summarize_selection_reports


PROJECT = 'synthetic-wp7-4-r2'


KEY = ['project_id','build_id','build_index','test_id']


CTX = KEY + ['build_started_at']


ORACLE = KEY + ['failed','execution_time_ms']


RAW = CTX + ['failed','execution_time_ms']


MODELS = ('logistic_regression','random_forest','xgboost')


ARMS = {'static':None,'h5':5,'h10':10,'h20':20}


RHOS = (.90,.95,.99)


REGIME = 'retrospective_full_information'


@dataclass(frozen=True)
class WorkingConfig:
    models: tuple = MODELS
    arms: tuple = tuple(ARMS)
    rho: tuple = RHOS
    feature_scope: str = 'F1_ONLY_V1'
    execution_authorized: bool = False  # never authorizes real data


@dataclass(frozen=True)
class Choice:
    arm: str
    model: str
    variant: str
    rho: float
    cutoff: int
    status: str
    selection: object
    probabilities: tuple
    candidate_ids: tuple = ()
    h2_status: str = "not_requested"
    h2_pairing: object = None


def _configs():
    _validate_synthetic_file()
    base = Path('configs')
    cfg = dict(rq1_config=load_yaml(base/'rq1_v1.yaml'),
               model_config=load_yaml(base/'rq1_models_v1.yaml'),
               calibration_config=load_yaml(base/'calibration_v1.yaml'),
               core_features_config=load_yaml(base/'core_features_v1.yaml'))
    cfg['rq1_config']['feature_set']['name']='F1'
    cfg['rq1_config']['pre_freeze_feature_scope']='F1_ONLY_V1'
    cfg['model_config']['feature_config']['primary_feature_set']='F1'
    return cfg


def _identity(frame, columns, label):
    if not isinstance(frame,pd.DataFrame) or set(frame.columns)!=set(columns):
        raise ValueError(f'{label}: unexpected schema')
    project,indices=_check_frame(frame,label,set(columns))
    if project!=PROJECT:
        raise ValueError('Only generated synthetic project accepted')
    if not pd.api.types.is_integer_dtype(frame.build_index):
        raise ValueError('Canonical integer build_index required')
    if any(not isinstance(v,str) for k in ['project_id','build_id','test_id'] for v in frame[k]):
        raise ValueError('Canonical string identities required')
    return indices


def _truth(context, oracle):
    ids=_identity(oracle,ORACLE,'oracle')
    if ids!={int(context.build_index.iloc[0])}:
        raise ValueError('Wrong target oracle')
    joined=context.merge(oracle,on=KEY,how='outer',validate='one_to_one',indicator=True)
    if len(joined)!=len(context) or not joined['_merge'].eq('both').all():
        raise ValueError('Oracle identity mismatch')
    y=pd.to_numeric(joined.failed,errors='raise')
    t=pd.to_numeric(joined.execution_time_ms,errors='raise')
    if not y.isin([0,1]).all() or not np.isfinite(t).all() or (t<0).any() or (t!=np.floor(t)).any():
        raise ValueError('Invalid revealed outcomes/durations')
    return joined.drop(columns='_merge').loc[:,RAW]


def _run(dev, contexts, reveal, *, config=WorkingConfig(), on_fixed=lambda i, choices:None, h2_enabled=False, snapshot_fitter=None):
    """Synthetic engineering only. No file loader. Caller reveal is invoked once after all choices."""
    if (config.execution_authorized is not False or config.feature_scope!='F1_ONLY_V1'
            or config.arms!=tuple(ARMS) or config.rho!=RHOS
            or not config.models or len(set(config.models))!=len(config.models)
            or not set(config.models)<=set(MODELS)):
        raise ValueError('Unsupported synthetic config')
    prior=_identity(dev,RAW,'development'); future=_identity(contexts,CTX,'contexts')
    ordered=sorted(future)
    if max(prior)>=min(future) or len(prior)!=math.floor(.8*(len(prior)+len(future))):
        raise ValueError('Invalid outer chronological split')
    if prior|future!=set(range(1,len(prior)+len(future)+1)):
        raise ValueError('Synthetic fixture requires contiguous build indices')
    if set(dev.build_id)&set(contexts.build_id):
        raise ValueError('Reused build identity')
    cfg=_configs()
    history=dev.copy(deep=True)
    snapshots={}; fit_events=[]; reports=[]; all_choices={}
    for j,index in enumerate(ordered):
        current=contexts.loc[contexts.build_index==index,CTX].copy()
        if int(history.build_index.max())>=index: raise AssertionError('Future history')
        pre=materialize_target_preexecution_features(history,current,feature_scope=config.feature_scope)
        decision=pre.loc[:,['project_id','build_index','test_id','execution_time_mean_history_ms']].copy()
        fit_arms=[a for a,h in ARMS.items() if j>0 and h is not None and j%h==0]
        if j==0 or fit_arms:
            features=compute_history_features(history.assign(changed_entity_count=np.nan))
            for model in config.models:
                # Share initial fit across all four arms; later refreshes are independent per arm.
                for arm in (['initial'] if j==0 else fit_arms):
                    snapshot=(snapshot_fitter or fit_rq1_snapshot)(features,model_name=model,**cfg)
                    if tuple(snapshot.feature_columns)!=tuple(F1_ONLY_FEATURES) or snapshot.last_development_build_index>=index:
                        raise AssertionError('Invalid snapshot')
                    fit_events.append((index,model,arm,snapshot.last_development_build_index))
                    for a in (ARMS if arm=='initial' else [arm]): snapshots[model,a]=snapshot
        choices=[]; prediction_cache={}; budget_inputs=None
        for model in config.models:
            for arm in config.arms:
                s=snapshots[model,arm]
                if s.last_development_build_index>=index: raise AssertionError('Snapshot overlaps target')
                if id(s) not in prediction_cache:
                    prediction_cache[id(s)]=predict_from_rq1_snapshot(s,pre.loc[:,KEY+list(s.feature_columns)].copy())
                pred=prediction_cache[id(s)]
                # Canonical one-to-one align, no positional assumptions from prediction API.
                aligned=pre[KEY].merge(pred,on=KEY,how='outer',sort=False,validate='one_to_one',indicator=True)
                if len(aligned)!=len(pre) or not aligned['_merge'].eq('both').all() or not aligned[KEY].reset_index(drop=True).equals(pre[KEY].reset_index(drop=True)):
                    raise ValueError('Prediction identity/order mismatch')
                for variant,col in PROBABILITY_COLUMNS.items():
                    p=aligned[col].to_numpy(dtype=float)
                    skipped=variant!='uncalibrated' and not s.calibration_support_sufficient
                    if skipped and not np.isnan(p).all(): raise ValueError('Invalid skipped calibration probabilities')
                    if not skipped and (not np.isfinite(p).all() or ((p<0)|(p>1)).any()):
                        raise ValueError('Invalid probability vector')
                    for rho in config.rho:
                        sel=None if skipped else select_with_cold_start_fallback(decision,history,p,rho)
                        h2_status, paired = 'not_requested', None
                        if h2_enabled:
                            if skipped:
                                h2_status = 'INSUFFICIENT_CALIBRATION_SUPPORT'
                            elif sel.selection_mode == 'fallback_retest_all':
                                h2_status, paired = _h2_pair(sel,None,model,variant,rho,index)
                            else:
                                if budget_inputs is None: budget_inputs=_budget_inputs(pre,decision,history)
                                h2_status, paired = _h2_pair(sel,budget_inputs,model,variant,rho,index)
                        choices.append(Choice(arm,model,variant,rho,s.last_development_build_index,
                                              'INSUFFICIENT_CALIBRATION_SUPPORT' if skipped else 'computed',sel,
                                              () if skipped else tuple(float(x) for x in p),
                                              tuple(pre.test_id), h2_status, paired))
        frozen=tuple(choices)
        all_choices[index]=frozen
        on_fixed(index,frozen)
        # Only now call oracle. No future oracle table or schema inspection in _run.
        realized=_truth(current,reveal(index))
        for c in frozen:
            row=dict(build_index=index,arm=c.arm,model=c.model,variant=c.variant,rho=c.rho,status=c.status)
            if c.selection is not None:
                row['metrics']=evaluate_post_selection(realized,c.selection,rho=c.rho,model=c.model,
                    probability=PROBABILITY_COLUMNS[c.variant],cost_policy='historical_median',baseline='risk_density',observation_regime=REGIME)
            reports.append(row)
        history=pd.concat([history,realized],ignore_index=True)
    return fit_events,all_choices,reports


def _budget_inputs(pre, decision, history):
    # Mirror the committed bridge's validation and unseen-test convention.
    counts = pd.to_numeric(pre.prior_execution_count, errors='raise').to_numpy(dtype=float)
    rates = pd.to_numeric(pre.failure_rate_history, errors='raise').to_numpy(dtype=float, copy=True)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError('Invalid historical execution counts')
    unseen = counts == 0
    if (np.isinf(rates).any() or np.isnan(rates[~unseen]).any() or
        ((rates < 0) | (rates > 1)).any() or (np.isfinite(rates[unseen]) & (rates[unseen] != 0)).any()):
        raise ValueError('Invalid historical failure rates')
    rates[unseen] = 0.0
    estimated = estimate_build_costs(target=decision, history=history, zero_cost_floor_ms=None)
    ids = tuple(decision.test_id)
    if not estimated.index.equals(decision.index) or tuple(estimated.test_id) != ids:
        raise ValueError('H2 cost identity mismatch')
    return ids, tuple(estimated.estimated_cost_ms), tuple(rates)


def _h2_pair(sel, inputs, model, variant, rho, index):
    if sel.selection_mode == 'fallback_retest_all':
        if sel.fallback_reason not in ('zero_total_predicted_risk','no_positive_prior_cost_history'):
            raise ValueError('Unknown fallback')
        return 'not_applicable_' + sel.fallback_reason, None
    if sel.selection_mode != 'normal_risk_density' or inputs is None:
        raise ValueError('Unsupported H2 path')
    ids,costs,rates=inputs
    chosen=set(sel.selected_test_ids)
    if not np.isclose(math.fsum(c for t,c in zip(ids,costs) if t in chosen),sel.estimated_selected_cost_ms,rtol=1e-9,atol=1e-7):
        raise ValueError('Reference cost mismatch')
    # Preserve five-field seed-v1; common random numbers across retraining arms.
    key=('model='+model,'probability='+variant,f'rho={rho:.2f}',
         'cost_policy=historical_median','observation='+REGIME)
    pair=pair_h2_budget_comparators_for_build(project_id=PROJECT,build_index=index,condition_key=key,
         test_ids=ids,estimated_costs_ms=costs,historical_failure_rates=rates,
         reference_selected_test_ids=sel.selected_test_ids)
    return 'computed',pair


def _report_matrix(choices,reports,revealed):
    """Descriptive synthetic reports only; stratify existing H2 reporter by arm."""
    if set(choices)!=set(revealed): raise ValueError('Incomplete reveal')
    oracle=pd.concat([revealed[i] for i in sorted(revealed)],ignore_index=True)
    by_arm={a:[] for a in ARMS}; status=[]; prob={}; seen=set(); rq2=[]
    lookup={(r['build_index'],r['arm'],r['model'],r['variant'],r['rho']):r for r in reports}
    if len(lookup)!=len(reports): raise ValueError('Duplicate report')
    for i,cs in sorted(choices.items()):
        truth=revealed[i].set_index('test_id',verify_integrity=True)
        for c in cs:
            key=(i,c.arm,c.model,c.variant,c.rho)
            if key in seen: raise ValueError('Duplicate choice')
            seen.add(key); r=lookup[key]
            meta=dict(project=PROJECT,arm=c.arm,model=c.model,probability=c.variant,rho=c.rho,build_index=i)
            status.append(dict(**meta,h2_status=c.h2_status,status=c.status))
            if c.rho==RHOS[0]:
                g=prob.setdefault((c.arm,c.model,c.variant),dict(y=[],p=[],available=0,skipped=0))
                if c.selection is None: g['skipped']+=1
                else:
                    if len(c.candidate_ids)!=len(truth) or set(c.candidate_ids)!=set(truth.index): raise ValueError('RQ1 identity mismatch')
                    g['y'].extend(truth.loc[list(c.candidate_ids),'failed'].tolist()); g['p'].extend(c.probabilities); g['available']+=1
            if c.selection is None:
                if c.h2_status!='INSUFFICIENT_CALIBRATION_SUPPORT' or c.h2_pairing is not None: raise ValueError('Invalid skip')
                continue
            if c.h2_status=='not_requested': raise ValueError('H2 not enabled')
            rq2.append(dict(r['metrics'],arm=c.arm))
            by_arm[c.arm].append(SimpleNamespace(project_id=PROJECT,build_index=i,model=c.model,
                probability_variant=c.variant,selected_test_ids=c.selection.selected_test_ids,
                selection_mode=c.selection.selection_mode,report=r['metrics'],h2_status=c.h2_status,h2_budget_pairing=c.h2_pairing))
    if seen!=set(lookup): raise ValueError('Unexpected report')
    rq1=[]; bins=[]; h2_build=[]; h2_project=[]
    for (a,m,v),g in sorted(prob.items()):
        meta=dict(project=PROJECT,arm=a,model=m,probability=v,n_available_builds=g['available'],n_skipped_builds=g['skipped'])
        if not g['available']:
            rq1.append(dict(**meta,status='NO_AVAILABLE_PREDICTIONS')); continue
        metrics,b=evaluate_rq1_probability(g['y'],g['p'],n_bins=10)
        rq1.append(dict(**meta,status='PARTIAL' if g['skipped'] else 'AVAILABLE',**metrics)); bins.append(b.assign(**meta))
    for arm,steps in by_arm.items():
        if steps:
            out=report_h2_prototype_steps(steps,oracle)
            if not out.build_metrics.empty: h2_build.append(out.build_metrics.assign(arm=arm))
            if not out.project_metrics.empty: h2_project.append(out.project_metrics.assign(arm=arm))
    def combine(items): return pd.concat(items,ignore_index=True) if items else pd.DataFrame()
    hp=combine(h2_project); random=[]
    if not hp.empty:
        keys=['project','arm','model','probability','rho','cost_policy','observation_regime','baseline','comparison_role']
        for key,g in hp[hp.baseline.str.startswith('random_')].groupby(keys,sort=True):
            if set(g.repetition)!=set(range(30)) or len(g)!=30: raise ValueError('Incomplete Random strata')
            row=dict(zip(keys,key),n_repeats=30)
            for metric in ('failure_recall_micro','rtr_micro','tsr_micro','budget_utilization'):
                row[metric+'_repeat_mean']=g[metric].mean(); row[metric+'_defined_repeats']=int(g[metric].notna().sum())
            random.append(row)
    return dict(rq1=pd.DataFrame(rq1),bins=combine(bins),rq2=pd.DataFrame(rq2),status=pd.DataFrame(status),
         h2_build=combine(h2_build),h2_project=hp,random_project=pd.DataFrame(random))


def _timing_report(fits,events):
    if len(fits)!=len(events): raise ValueError('Timing/fit event mismatch')
    rows=[]
    for (i,m,a,c),e in zip(fits,events):
        if (m,c)!=(e['model'],e['cutoff']) or c>=i: raise ValueError('Timing identity mismatch')
        rows.append(dict(e,arm=a,target_build_index=i))
    frame=pd.DataFrame(rows)
    if frame.duplicated(['model','arm','target_build_index']).any(): raise ValueError('Duplicate timing event')
    totals=[]
    fields=['snapshot_total_ns','model_fit_ns','calibration_prediction_ns','platt_fit_ns','isotonic_fit_ns','other_snapshot_ns']
    for model,g in frame.groupby('model'):
        init=g[g.arm.eq('initial')]
        if len(init)!=1: raise ValueError('Expected one shared initial snapshot')
        for arm in ARMS:
            refresh=g[g.arm.eq(arm)]; row=dict(model=model,arm=arm,n_initial=1,n_refresh=len(refresh))
            # Logical per-arm attribution, not sum of physical work across arms.
            for k in fields:
                row[k+'_initial']=int(init[k].sum()); row[k+'_refresh']=int(refresh[k].sum())
            totals.append(row)
    return frame,pd.DataFrame(totals)


def _rq2_summary(matrix):
    status=matrix['status']; actual=matrix['rq2'].copy()
    keys=['project','arm','model','probability','rho']; ident=keys+['build_index']
    if status.empty or status.duplicated(ident).any(): raise ValueError('Invalid status matrix')
    if not status.status.isin(['computed','INSUFFICIENT_CALIBRATION_SUPPORT']).all(): raise ValueError('Unknown status')
    if not actual.empty:
        actual['probability']=actual.probability.map({v:k for k,v in PROBABILITY_COLUMNS.items()})
        # Committed reporter returns canonical build identity as a string.
        from src.tcp_ci.rq2_cold_start_reporting import _canonical_build_index
        actual['build_index']=actual.build_index.map(_canonical_build_index)
        if actual.probability.isna().any() or actual.duplicated(ident).any(): raise ValueError('Invalid RQ2 keys')
        left=set(map(tuple,actual[ident].itertuples(index=False,name=None)))
    else: left=set()
    right=set(map(tuple,status.loc[status.status.eq('computed'),ident].itertuples(index=False,name=None)))
    if left!=right: raise ValueError('Computed status/report mismatch')
    records=[]
    for key,g in status.groupby(keys,sort=True):
        part=actual
        if not part.empty:
            for k,v in zip(keys,key): part=part[part[k].eq(v)]
        n=len(part); row=dict(zip(keys,key),n_scheduled_builds=len(g),n_computed_builds=n,
            n_skipped_builds=len(g)-n,n_h2_paired_builds=int(g.h2_status.eq('computed').sum()),
            n_zero_risk_fallback=0,n_no_prior_cost_fallback=0)
        if n:
            out=summarize_selection_reports(part,require_condition_metadata=True)
            if len(out)!=1: raise ValueError('Mixed RQ2 condition metadata')
            row.update(out.iloc[0].to_dict()); row['arm']=key[1]
            row['n_zero_risk_fallback']=int(part.fallback_reason.eq('zero_total_predicted_risk').sum())
            row['n_no_prior_cost_fallback']=int(part.fallback_reason.eq('no_positive_prior_cost_history').sum())
            if row['n_fallback_builds']!=row['n_zero_risk_fallback']+row['n_no_prior_cost_fallback']: raise ValueError('Fallback accounting mismatch')
        else:
            row.update(n_evaluation_builds=0,n_fallback_builds=0,n_normal_builds=0,
                tsr_micro=np.nan,rtr_micro=np.nan,failure_recall_micro=np.nan,missed_failure_rate_micro=np.nan)
        records.append(row)
    return pd.DataFrame(records),actual


def _rq3_common_targets(status,actual,scheduled):
    """§187 descriptive proposal: fixed chronological halves and 20-target blocks.
    Pair only common computed targets, disclose coverage, no final inference.
    """
    keys=['project','model','probability','rho']; schedule=list(scheduled)
    if schedule!=sorted(set(schedule)) or not schedule: raise ValueError('Invalid schedule')
    records=[]; blocks=[]
    def pooled(part):
        if part.empty: return np.nan,np.nan,np.nan
        f=part.total_failures.sum(); d=part.observed_full_ms.sum(); n=part.n_tests.sum()
        return (part.captured_failures.sum()/f if f else np.nan,
                part.observed_selected_ms.sum()/d if d else np.nan,
                part.n_selected.sum()/n if n else np.nan)
    for key,g in status.groupby(keys,sort=True):
        for arm in ARMS:
            z=g[g.arm.eq(arm)]
            if len(z)!=len(schedule) or set(z.build_index)!=set(schedule): raise ValueError('Incomplete arm schedule')
        part=actual
        if not part.empty:
            for k,v in zip(keys,key): part=part[part[k].eq(v)]
        for arm in ('h5','h10','h20'):
            a=g[g.arm.eq('static') & g.status.eq('computed')]; b=g[g.arm.eq(arm) & g.status.eq('computed')]
            common=set(a.build_index)&set(b.build_index)
            meta=dict(zip(keys,key),arm=arm,n_scheduled=len(schedule),n_static_computed=len(a),n_periodic_computed=len(b),n_common=len(common))
            row=meta.copy()
            if part.empty: p=q=part
            else:
                p=part[part.arm.eq('static') & part.build_index.isin(common)]; q=part[part.arm.eq(arm) & part.build_index.isin(common)]
                if set(p.build_index)!=common or set(q.build_index)!=common: raise ValueError('Common target mismatch')
                cols=['build_index','total_failures','observed_full_ms','n_tests']
                if not p[cols].sort_values('build_index').reset_index(drop=True).equals(q[cols].sort_values('build_index').reset_index(drop=True)): raise ValueError('Different oracle denominators')
            for name,x,y in zip(('fr','rtr','tsr'),pooled(p),pooled(q)):
                row[name+'_static']=x; row[name+'_periodic']=y; row[name+'_delta']=y-x
            def restrict(frame,ids): return frame if frame.empty else frame[frame.build_index.isin(ids)]
            cut=len(schedule)//2; half_fr={}
            for name,ids in [('first',schedule[:cut]),('second',schedule[cut:])]:
                x=pooled(restrict(p,ids))[0]; y=pooled(restrict(q,ids))[0]
                half_fr[name]=(x,y); row[name+'_common']=len(common&set(ids)); row[name+'_fr_static']=x; row[name+'_fr_periodic']=y
            for j,name in enumerate(('static','periodic')): row['fr_decay_'+name]=half_fr['second'][j]-half_fr['first'][j]
            paired=[]
            for j in range(0,len(schedule),20):
                ids=schedule[j:j+20]; x=pooled(restrict(p,ids))[0]; y=pooled(restrict(q,ids))[0]
                valid=len(ids)==20 and np.isfinite(x) and np.isfinite(y)
                if valid: paired.append((x,y))
                blocks.append(dict(**meta,block_start=ids[0],block_end=ids[-1],block_scheduled=len(ids),
                    block_common=len(common&set(ids)),complete_block=len(ids)==20,variance_included=bool(valid),fr_static=x,fr_periodic=y))
            row['n_paired_defined_complete_blocks']=len(paired)
            for j,name in enumerate(('static','periodic')):
                row['fr_block_sample_variance_'+name]=float(np.var([r[j] for r in paired],ddof=1)) if len(paired)>=2 else np.nan
            records.append(row)
    return pd.DataFrame(records),pd.DataFrame(blocks)


def _validate_synthetic_file():
    """Reject any changed execution setting; this version cannot authorize CORE."""
    observed = load_yaml(Path('configs/walk_forward_pre_freeze_v1.yaml'))
    expected = {'version': 1, 'scope': 'PRE_FREEZE_SYNTHETIC_ONLY', 'execution_authorized': False, 'method_freeze': 'NOT_APPROVED', 'project_id': 'synthetic-wp7-4-r2', 'feature_scope': 'F1_ONLY_V1', 'models': ['logistic_regression', 'random_forest', 'xgboost'], 'arms': {'static': None, 'h5': 5, 'h10': 10, 'h20': 20}, 'rho': [0.9, 0.95, 0.99], 'initial_development_fraction': 0.8, 'random_repetitions': 30, 'observation_regime': 'retrospective_full_information', 'calibration_config': 'configs/calibration_v1.yaml', 'rq1_config': 'configs/rq1_v1.yaml', 'model_config': 'configs/rq1_models_v1.yaml', 'core_features_config': 'configs/core_features_v1.yaml'}
    # Strict bool check prevents YAML integer 0 from masquerading as False.
    if not isinstance(observed, dict) or observed.get('execution_authorized') is not False or observed != expected:
        raise ValueError('Unsupported versioned synthetic config')
    return observed
