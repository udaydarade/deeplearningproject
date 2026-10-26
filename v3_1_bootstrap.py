"""Paired crash-event bootstrap for V3.1 baseline/TDA score files."""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd

from v3_download_data import OUT
from v3_1_eval import alarm_days, _merge, _runs, evaluate


def event_hits(frame: pd.DataFrame, alarm: np.ndarray) -> np.ndarray:
    f=frame.copy(); f['date']=pd.to_datetime(f.date); f['alarm']=alarm
    hits=[]
    for _,g in f.groupby('asset',sort=False):
        g=g.sort_values('date'); y=g.y.to_numpy(bool); a=g.alarm.to_numpy(bool)
        for s,e in _merge(_runs(y),5):
            hits.append(bool((y[s:e+1]&a[s:e+1]).any()))
    return np.asarray(hits,bool)


def paired_bootstrap(base: pd.DataFrame, treatment: pd.DataFrame, *,
                     sims: int=1000, seed: int=7301) -> dict:
    keys=['asset','date','fold','y']
    b=base[keys+['score']].rename(columns={'score':'score_b'})
    t=treatment[keys+['score']].rename(columns={'score':'score_t'})
    f=b.merge(t,on=keys,how='inner',validate='one_to_one').sort_values(['asset','date'])
    if len(f)!=len(base) or len(f)!=len(treatment):
        raise ValueError('paired score files do not have identical rows')
    vb=OUT/'scores'/'_bootstrap_val_base.parquet'
    # test alarms use each model's own validation score history at RQ 5%.
    fold=str(f.fold.iloc[0]); fi=int(fold.split('-')[0])
    def val_for(which):
        p=OUT/'scores'/f"step3_{json.loads((OUT/'v31_best_classical.json').read_text())['feature_set']}_gbm_fold{fi}_val.parquet" if which=='base' else None
        return pd.read_parquet(p) if p and p.exists() else None
    # Treatment feature-set name is inferred from the caller-supplied test frame attrs.
    valb=base.attrs.get('validation'); valt=treatment.attrs.get('validation')
    if valb is None or valt is None:
        raise ValueError('attach matching validation frames as .attrs["validation"]')
    ab=alarm_days(f.rename(columns={'score_b':'score'}),'RQ',.05,validation=valb)
    at=alarm_days(f.rename(columns={'score_t':'score'}),'RQ',.05,validation=valt)
    eb=event_hits(f.rename(columns={'score_b':'score'}),ab)
    et=event_hits(f.rename(columns={'score_t':'score'}),at)
    if len(eb)!=len(et): raise ValueError('event lists are not paired')
    rng=np.random.default_rng(seed); n=len(eb)
    diffs=np.empty(sims)
    for i in range(sims):
        ix=rng.integers(0,n,n)
        diffs[i]=np.mean(et[ix])-np.mean(eb[ix])
    mb=evaluate(f.rename(columns={'score_b':'score'}),ab,sims=200,seed=seed)
    mt=evaluate(f.rename(columns={'score_t':'score'}),at,sims=200,seed=seed)
    # The alarm-budget random correction is held at the observed per-model
    # baseline; paired resampling targets uncertainty from crash episodes.
    center=mt['excess_event_recall']-mb['excess_event_recall']
    offset=center-(np.mean(et)-np.mean(eb))
    diffs+=offset
    return {'events':n,'mean_excess_diff':center,
            'ci_low':float(np.quantile(diffs,.025)),
            'ci_high':float(np.quantile(diffs,.975)),
            'folds_won':int(center>0),'baseline_excess':mb['excess_event_recall'],
            'treatment_excess':mt['excess_event_recall'],
            'baseline_alarm_rate':mb['realized_alarm_day_rate'],
            'treatment_alarm_rate':mt['realized_alarm_day_rate'],
            'baseline_rand_recall':mb['rand_event_recall'],
            'treatment_rand_recall':mt['rand_event_recall'],
            'bootstrap_samples':diffs}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--treatment',required=True); p.add_argument('--baseline-val',required=True); p.add_argument('--treatment-val',required=True); p.add_argument('--out',default='data/v3/v31_step4_bootstrap.csv'); a=p.parse_args()
    base=pd.read_parquet(a.baseline); tr=pd.read_parquet(a.treatment)
    base.attrs['validation']=pd.read_parquet(a.baseline_val); tr.attrs['validation']=pd.read_parquet(a.treatment_val)
    result=paired_bootstrap(base,tr)
    pd.DataFrame([result]).to_csv(a.out,index=False); print(result)

if __name__=='__main__': main()
