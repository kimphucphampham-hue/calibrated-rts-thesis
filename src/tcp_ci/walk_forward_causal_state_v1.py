"""Versioned synthetic-only F1/cost state. No loader or execution authorization."""
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.tcp_ci import walk_forward_pre_freeze_v1 as v1
from src.tcp_ci.rq2_causal_feature_materializer import F1_ONLY_DECISION_COLUMNS
from src.tcp_ci.rq2_cold_start_orchestrator import NO_POSITIVE_HISTORY_ERROR

@dataclass(frozen=True)
class TestState:
    count: int
    failures: int
    last: int
    streak: int
    first_build: int
    last_failure: object
    duration_sum: int

class CausalState:
    """One synthetic project; update once per fully revealed contiguous build.

    prepare has no oracle input. advance validates matching oracle before mutation.
    Caller still must freeze/score ALL selections before advance.
    """
    def __init__(self):
        self.tests = {}
        self.durations = Counter()
        self.last_index = 0
        self.last_time = None
        self.build_ids = set()
        self._pending = None
        self._prepared = None
        self._median = None
        self._median_valid = False

    def prepare(self, context):
        if self._pending is not None:
            raise ValueError('Previous build not advanced')
        ids = v1._identity(context, v1.CTX, 'context')
        if ids != {self.last_index + 1}:
            raise ValueError('Expected next build')
        ctx = context.loc[:, v1.CTX].copy().sort_values('test_id', kind='stable').reset_index(drop=True)
        ts = pd.to_datetime(ctx.build_started_at, utc=True, errors='raise')
        if ts.isna().any() or ts.nunique() != 1 or (self.last_time is not None and ts.iloc[0] < self.last_time):
            raise ValueError('Invalid chronological time')
        if ctx.build_id.iloc[0] in self.build_ids:
            raise ValueError('Reused build identity')
        ctx['build_started_at'] = ts
        i = self.last_index + 1
        nan = float('nan')
        values = []
        for t in ctx.test_id:
            s = self.tests.get(t)
            if s is None:
                values.append((nan,nan,0,0,0,nan,nan,0,0,0))
            else:
                values.append((s.failures/s.count,float(s.last),s.streak if s.last else 0,
                    s.streak if not s.last else 0,i-s.first_build,
                    float(i-s.last_failure) if s.last_failure is not None else nan,
                    float(s.duration_sum)/float(s.count),1,s.count,s.failures))
        cols = list(F1_ONLY_DECISION_COLUMNS[len(v1.CTX):])
        assert cols == ['failure_rate_history','recent_failure','consecutive_failures','consecutive_passes','test_age_builds','time_since_last_failure_builds','execution_time_mean_history_ms','history_available','prior_execution_count','prior_failure_count']
        f = pd.DataFrame(values,columns=cols)
        f['history_available'] = f.history_available.astype('int8')
        self._pending = ctx
        self._prepared = pd.concat([ctx, f],axis=1).loc[:,list(F1_ONLY_DECISION_COLUMNS)]
        return self._prepared.copy(deep=True)

    def estimated_costs(self, pre):
        if self._pending is None or not pre.equals(self._prepared):
            raise ValueError('Cost context mismatch')
        # Same missing/zero -> strictly earlier positive project median policy.
        d = pre[['project_id','build_index','test_id','execution_time_mean_history_ms']].copy()
        means = d.execution_time_mean_history_ms.astype(float)
        if np.isinf(means).any() or (means<0).any():
            raise ValueError('Invalid mean')
        missing = means.isna() | means.eq(0)
        if missing.any():
            if not self.durations:
                raise ValueError(NO_POSITIVE_HISTORY_ERROR)
            if not self._median_valid:
                n = sum(self.durations.values())
                targets = ((n-1)//2,n//2)
                vals = []
                count = 0
                for duration, freq in sorted(self.durations.items()):
                    count += freq
                    while len(vals)<2 and targets[len(vals)]<count:
                        vals.append(float(duration))
                    if len(vals)==2:
                        break
                self._median = float(np.median(vals))
                self._median_valid = True
            means = means.mask(missing,self._median)
        d['estimated_cost_ms'] = means
        return d

    def advance(self, oracle):
        if self._pending is None:
            raise ValueError('No prepared build')
        realized = v1._truth(self._pending,oracle)
        i = self.last_index + 1
        updates = {}
        increments = Counter()
        for row in realized.itertuples(index=False):
            t,y,d = row.test_id,int(row.failed),int(row.execution_time_ms)
            # Fail closed outside the exact-integer synthetic domain or frozen int64 sum.
            if d > 2**53-1:
                raise ValueError('Duration outside candidate exact integer domain')
            s = self.tests.get(t)
            total = d + (s.duration_sum if s else 0)
            if total > 2**63-1:
                raise ValueError('Historical duration sum overflow')
            updates[t] = TestState(1+(s.count if s else 0),y+(s.failures if s else 0),y,
                1+(s.streak if s and s.last==y else 0),s.first_build if s else i,
                i if y else (s.last_failure if s else None),total)
            if d>0:
                increments[d] += 1
        self.tests.update(updates)
        self.durations.update(increments)
        if increments:
            self._median_valid = False
        self.last_index = i
        self.last_time = self._pending.build_started_at.iloc[0]
        self.build_ids.add(self._pending.build_id.iloc[0])
        self._pending = None
        self._prepared = None
        return realized
