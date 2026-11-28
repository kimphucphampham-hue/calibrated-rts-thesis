"""Pre-freeze synthetic-only incremental runner and private ephemeral report spool."""
from functools import partial
import math
import pandas as pd
import numpy as np
from src.tcp_ci.walk_forward_pre_freeze_v1 import (WorkingConfig, ARMS, RHOS, MODELS, RAW, CTX, ORACLE, KEY, Choice, REGIME, F1_ONLY_FEATURES, PROBABILITY_COLUMNS, _identity, _truth, _configs, _h2_pair, fit_rq1_snapshot, predict_from_rq1_snapshot, evaluate_post_selection)
from src.tcp_ci.rq2_cold_start_orchestrator import NO_POSITIVE_HISTORY_ERROR
from src.tcp_ci.walk_forward_causal_state_v1 import CausalState
from src.tcp_ci.rq2_cold_start_orchestrator import _select_with_cost_provider
from src.tcp_ci.walk_forward_pre_freeze_v1 import _budget_inputs_with_provider

def _run_incremental(dev, contexts, reveal, *, config=WorkingConfig(), on_fixed=lambda i, choices:None, h2_enabled=False, snapshot_fitter=None, scored_sink=None):
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
    state=CausalState()
    training_parts=[]
    def remember(pre, realized):
        # Preserve the reference sort order and binary target dtype; never include current truth in F1.
        rows=pre[KEY+list(F1_ONLY_FEATURES)].merge(realized[KEY+['failed']],on=KEY,validate='one_to_one',sort=False)
        rows['failed']=pd.to_numeric(rows.failed,errors='raise').astype('int8')
        training_parts.append(rows)
    for _, build in dev.groupby('build_index',sort=True):
        pre=state.prepare(build[CTX])
        remember(pre,state.advance(build[ORACLE]))
    snapshots={}; fit_events=[]; reports=[]; all_choices={}
    for j,index in enumerate(ordered):
        current=contexts.loc[contexts.build_index==index,CTX].copy()
        if state.last_index>=index: raise AssertionError('Future history')
        pre=state.prepare(current)
        decision=pre.loc[:,['project_id','build_index','test_id','execution_time_mean_history_ms']].copy()
        fit_arms=[a for a,h in ARMS.items() if j>0 and h is not None and j%h==0]
        if j==0 or fit_arms:
            features=pd.concat(training_parts,ignore_index=True)
            for model in config.models:
                # Share initial fit across all four arms; later refreshes are independent per arm.
                for arm in (['initial'] if j==0 else fit_arms):
                    snapshot=(snapshot_fitter or fit_rq1_snapshot)(features,model_name=model,**cfg)
                    if tuple(snapshot.feature_columns)!=tuple(F1_ONLY_FEATURES) or snapshot.last_development_build_index>=index:
                        raise AssertionError('Invalid snapshot')
                    fit_events.append((index,model,arm,snapshot.last_development_build_index))
                    for a in (ARMS if arm=='initial' else [arm]): snapshots[model,a]=snapshot
        choices=[]; prediction_cache={}; budget_inputs=None
        prepared_costs=None
        cost_error=None
        def cost_provider():
            nonlocal prepared_costs,cost_error
            if cost_error is not None:
                raise ValueError(cost_error)
            if prepared_costs is None:
                try:
                    prepared_costs=state.estimated_costs(pre)
                except ValueError as e:
                    if str(e)!=NO_POSITIVE_HISTORY_ERROR: raise
                    cost_error=str(e)
                    raise
            return prepared_costs
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
                        sel=None if skipped else _select_with_cost_provider(decision,cost_provider,p,rho)
                        h2_status, paired = 'not_requested', None
                        if h2_enabled:
                            if skipped:
                                h2_status = 'INSUFFICIENT_CALIBRATION_SUPPORT'
                            elif sel.selection_mode == 'fallback_retest_all':
                                h2_status, paired = _h2_pair(sel,None,model,variant,rho,index)
                            else:
                                if budget_inputs is None: budget_inputs=_budget_inputs_with_provider(pre,decision,cost_provider)
                                h2_status, paired = _h2_pair(sel,budget_inputs,model,variant,rho,index)
                        choices.append(Choice(arm,model,variant,rho,s.last_development_build_index,
                                              'INSUFFICIENT_CALIBRATION_SUPPORT' if skipped else 'computed',sel,
                                              () if skipped else tuple(float(x) for x in p),
                                              tuple(pre.test_id), h2_status, paired))
        frozen=tuple(choices)
        if scored_sink is None: all_choices[index]=frozen
        on_fixed(index,frozen)
        # Only now call oracle. No future oracle table or schema inspection in _run.
        realized=_truth(current,reveal(index))
        target_reports=[]
        for c in frozen:
            row=dict(build_index=index,arm=c.arm,model=c.model,variant=c.variant,rho=c.rho,status=c.status)
            if c.selection is not None:
                row['metrics']=evaluate_post_selection(realized,c.selection,rho=c.rho,model=c.model,
                    probability=PROBABILITY_COLUMNS[c.variant],cost_policy='historical_median',baseline='risk_density',observation_regime=REGIME)
            target_reports.append(row)
        if scored_sink is None:
            reports.extend(target_reports)
        else:
            scored_sink(index,frozen,tuple(target_reports),realized[ORACLE].copy())
        remember(pre,state.advance(realized[ORACLE]))
    return fit_events,all_choices,reports


from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import pickle
from src.tcp_ci import walk_forward_pre_freeze_v1 as v1

class _ScoredSpool:
    def __init__(self):
        self._tmp = TemporaryDirectory(prefix='r6-synthetic-')
        self._path = Path(self._tmp.name)/'scored.sqlite'
        self._db = sqlite3.connect(str(self._path))
        self._db.execute('CREATE TABLE oracle (target INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        self._db.execute('CREATE TABLE records (model TEXT, variant TEXT, rho REAL, target INTEGER, payload BLOB NOT NULL, PRIMARY KEY(model,variant,rho,target))')
        self._closed=False
        self._sealed=False
        self._schedule=[]
        self._keys=None
    def __enter__(self): return self
    def __exit__(self,*args):
        self._db.close(); self._tmp.cleanup(); self._closed=True
    def append(self,index,choices,reports,oracle):
        index=int(index)
        if self._closed or self._sealed: raise ValueError('Spool not writable')
        if self._schedule and index<=self._schedule[-1]: raise ValueError('Nonchronological scored target')
        if set(oracle.columns)!=set(v1.ORACLE): raise ValueError('Oracle schema mismatch')
        if set(oracle.build_index)!={index}: raise ValueError('Oracle target mismatch')
        lookup={(r['arm'],r['model'],r['variant'],r['rho']):r for r in reports}
        if len(lookup)!=len(reports) or len(choices)!=len(reports): raise ValueError('Duplicate/incomplete reports')
        groups={}; seen=set()
        for c in choices:
            key=(c.arm,c.model,c.variant,c.rho)
            if key in seen or key not in lookup: raise ValueError('Duplicate/incomplete choices')
            seen.add(key); r=lookup[key]
            if r['build_index']!=index or r['status']!=c.status: raise ValueError('Scored identity mismatch')
            groups.setdefault(key[1:],[]).append((c,r))
        if seen!=set(lookup) or any(tuple(c.arm for c,r in group)!=tuple(v1.ARMS) for group in groups.values()): raise ValueError('Incomplete stratum arms')
        if not groups or (self._keys is not None and set(groups)!=self._keys): raise ValueError('Changing condition universe')
        with self._db:
            self._db.execute('INSERT INTO oracle VALUES (?,?)',(index,pickle.dumps(oracle,protocol=4)))
            self._db.executemany('INSERT INTO records VALUES (?,?,?,?,?)',((*k,index,pickle.dumps(group,protocol=4)) for k,group in groups.items()))
        self._keys=set(groups); self._schedule.append(index)
    def seal(self,expected_schedule):
        if self._closed or self._sealed or not self._schedule or list(expected_schedule)!=self._schedule: raise ValueError('Incomplete spool schedule')
        self._sealed=True
    def size_bytes(self): return self._path.stat().st_size
    def iter_reports(self):
        if self._closed or not self._sealed: raise ValueError('Spool must be sealed')
        for key in sorted(self._keys):
            choices={}; reports=[]; oracle={}
            query='SELECT records.target,records.payload,oracle.payload FROM records JOIN oracle USING(target) WHERE model=? AND variant=? AND rho=? ORDER BY target'
            for index,payload,truth in self._db.execute(query,key):
                group=pickle.loads(payload); oracle[index]=pickle.loads(truth)
                choices[index]=tuple(c for c,r in group); reports.extend(r for c,r in group)
            if list(choices)!=self._schedule: raise ValueError('Incomplete stratum schedule')
            matrix=v1._report_matrix(choices,reports,oracle)
            summary,actual=v1._rq2_summary(matrix)
            rq3,blocks=v1._rq3_common_targets(matrix['status'],actual,self._schedule)
            # Drop bulky selections and oracle before yielding the single stratum.
            del choices,reports,oracle,actual,group,payload,truth
            yield key,dict(**matrix,rq2_summary=summary,rq3=rq3,rq3_blocks=blocks)
            del matrix,summary,rq3,blocks


def run_partitioned_synthetic(dev, contexts, reveal, *, report_sink, config=WorkingConfig(), on_fixed=lambda i, choices: None):
    """Run generated synthetic data only; consume each report stratum synchronously.

    H2 comparators are required by the existing report matrix.
    report_sink receives (model, variant, rho) and tables for one stratum.
    The caller must not accumulate tables if reduced retention is desired.
    No data loader or persistent result format. On failure the private spool is cleaned.
    Returns physical fit schedule, timing event table, logical timing summary, spool bytes.
    """
    if not callable(report_sink):
        raise TypeError('report_sink must be callable')
    events = []
    with _ScoredSpool() as spool:
        fits, choices, reports = _run_incremental(
            dev, contexts, reveal, config=config, on_fixed=on_fixed,
            h2_enabled=True, scored_sink=spool.append,
            snapshot_fitter=partial(fit_rq1_snapshot, timing_sink=events.append))
        if choices or reports:
            raise AssertionError('Partitioned runner retained project records')
        spool.seal(sorted(set(contexts.build_index)))
        physical, logical = v1._timing_report(fits, events)
        size = spool.size_bytes()
        for key, tables in spool.iter_reports():
            report_sink(key, tables)
            del tables
    return fits, physical, logical, size
