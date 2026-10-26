"""Summarize V3.1 step CSVs and run the registered paired TDA bootstrap."""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from v3_download_data import OUT
from v3_1_bootstrap import paired_bootstrap
from v3_1_eval import off_budget


def main():
    p=argparse.ArgumentParser(); p.add_argument('step',choices=['2','3','4','5','6','7']); a=p.parse_args()
    path=OUT/f'v31_step{a.step}.csv'
    if not path.exists(): raise SystemExit(f'missing {path}')
    df=pd.read_csv(path)
    primary=df.loc[(df.rule=='RQ')&(df.alarm_budget==.05)&(~df.off_budget)].copy()
    summary=primary.groupby(['feature_set','model','subset'],dropna=False).agg(
        folds=('fold','nunique'),mean_excess_event_recall=('excess_event_recall','mean'),
        mean_event_recall=('event_recall','mean'),mean_event_precision=('event_precision','mean'),
        mean_day_lift=('day_lift','mean'),mean_rand_event_recall=('rand_event_recall','mean'),
        mean_alarm_day_rate=('realized_alarm_day_rate','mean'),off_budget_count=('off_budget','sum')).reset_index()
    summary.to_csv(OUT/f'v31_step{a.step}_headline.csv',index=False)
    print(summary.to_string(index=False))
    print(f"Off-budget runs excluded from headline: {int(df.loc[(df.rule=='RQ')&(df.alarm_budget==.05),'off_budget'].sum())}")
    if a.step=='2':
        cols=['feature_set','model','fold','excess_event_recall','event_precision','day_lift','rand_event_recall','realized_alarm_day_rate','off_budget']
        print('Per-fold RQ@5% results (off-budget rows shown but excluded from headline means):')
        print(df.loc[(df.rule=='RQ')&(df.alarm_budget==.05),cols].sort_values(['feature_set','model','fold']).to_string(index=False))
    if a.step=='5':
        best=json.loads((OUT/'v31_best_classical.json').read_text())['feature_set']
        cfg=pd.read_csv(path); variants=[x for x in cfg.feature_set.unique() if x!='BEST_CLASSICAL']
        entries=[]
        for variant in variants:
            safe=variant.replace('+','_').replace('/','_')
            samples=[]; means=[]; wins=0; apdiff=[]; rates=[]; precision=[]; random=[]; off_folds=0
            for fi in range(5):
                base=OUT/'scores'/f'step5_BEST_CLASSICAL_gbm_fold{fi}_test.parquet'
                basev=OUT/'scores'/f'step5_BEST_CLASSICAL_gbm_fold{fi}_val.parquet'
                tr=OUT/'scores'/f'step5_{safe}_gbm_fold{fi}_test.parquet'
                trv=OUT/'scores'/f'step5_{safe}_gbm_fold{fi}_val.parquet'
                if not all(x.exists() for x in (base,basev,tr,trv)): continue
                b=pd.read_parquet(base); t=pd.read_parquet(tr)
                b.attrs['validation']=pd.read_parquet(basev); t.attrs['validation']=pd.read_parquet(trv)
                r=paired_bootstrap(b,t,seed=8500+fi)
                if off_budget(r['baseline_alarm_rate'],.05) or off_budget(r['treatment_alarm_rate'],.05):
                    off_folds+=1; continue
                samples.append(r.pop('bootstrap_samples'))
                means.append(r['mean_excess_diff']); wins+=int(r['mean_excess_diff']>0)
                apdiff.append(float(average_precision_score(t.y,t.score)-average_precision_score(b.y,b.score)))
                rates.append(r['treatment_alarm_rate']); precision.append(r['treatment_event_precision']); random.append(r['treatment_rand_recall'])
            if off_folds>1:
                entries.append({'feature_set':variant,'model':'gbm','eligible_folds':len(samples),
                    'off_budget_folds':off_folds,'experiment_stopped':True})
            elif len(samples)>=3:
                avg=np.mean(np.vstack(samples),axis=0)
                entries.append({'feature_set':variant,'model':'gbm','mean_diff':float(np.mean(means)),
                    'ci_low':float(np.quantile(avg,.025)),'ci_high':float(np.quantile(avg,.975)),
                    'folds_won':wins,'eligible_folds':len(samples),'off_budget_folds':off_folds,
                    'experiment_stopped':False,'mean_ap_diff':float(np.mean(apdiff)),
                    'mean_event_precision':float(np.mean(precision)),'mean_alarm_day_rate':float(np.mean(rates)),
                    'mean_rand_event_recall':float(np.mean(random))})
        pd.DataFrame(entries).to_csv(OUT/'v31_step5_paired_bootstrap.csv',index=False)
        if entries: print(pd.DataFrame(entries).to_string(index=False))
        return
    if a.step!='4': return
    cfg=json.loads((OUT/'v31_best_classical.json').read_text()); best=cfg['feature_set']
    blocks=json.loads((OUT/'v31_tda_blocks.json').read_text())['blocks']
    entries=[]
    for b,cols in blocks.items():
        if not cols: continue
        treatment_name=(best+'_'+b).replace('+','_')
        for model in ('logreg','gbm'):
            diffs=[]; means=[]; apdiff=[]; event_prec=[]; rand=[]; rates=[]; trates=[]; off_folds=0
            for fi in range(5):
                base_path=OUT/'scores'/f'step3_{best}_{model}_fold{fi}_test.parquet'
                base_val=OUT/'scores'/f'step3_{best}_{model}_fold{fi}_val.parquet'
                tr_path=OUT/'scores'/f'step4_{treatment_name}_{model}_fold{fi}_test.parquet'
                tr_val=OUT/'scores'/f'step4_{treatment_name}_{model}_fold{fi}_val.parquet'
                if not all(x.exists() for x in (base_path,base_val,tr_path,tr_val)):
                    print(f'bootstrap pending {treatment_name}/{model} fold={fi}: missing score file')
                    continue
                base=pd.read_parquet(base_path); treat=pd.read_parquet(tr_path)
                base.attrs['validation']=pd.read_parquet(base_val); treat.attrs['validation']=pd.read_parquet(tr_val)
                r=paired_bootstrap(base,treat,seed=8300+fi)
                if off_budget(r['baseline_alarm_rate'],.05) or off_budget(r['treatment_alarm_rate'],.05):
                    off_folds+=1
                    print(f'paired bootstrap excludes off-budget fold {treatment_name}/{model}/{fi}')
                    continue
                diffs.append(r.pop('bootstrap_samples')); means.append(r['mean_excess_diff'])
                event_prec.append(r['treatment_event_precision']); rand.append(r['treatment_rand_recall']); rates.append(r['baseline_alarm_rate']); trates.append(r['treatment_alarm_rate'])
                apdiff.append(float(average_precision_score(treat.y, treat.score)-average_precision_score(base.y, base.score)))
            if off_folds>1:
                entries.append({'block':b,'model':model,'mean_diff':np.nan,'ci_low':np.nan,'ci_high':np.nan,
                    'folds_won':int(sum(m>0 for m in means)),'eligible_folds':len(diffs),
                    'off_budget_folds':off_folds,'experiment_stopped':True})
            elif len(diffs)>=3:
                avg=np.mean(np.vstack(diffs),axis=0)
                entries.append({'block':b,'model':model,'mean_diff':float(np.mean(means)),
                    'ci_low':float(np.quantile(avg,.025)),'ci_high':float(np.quantile(avg,.975)),
                    'folds_won':int(sum(m>0 for m in means)),
                    'eligible_folds':len(diffs),'off_budget_folds':off_folds,'experiment_stopped':False,
                    'mean_ap_diff':float(np.mean(apdiff)),
                    'mean_treatment_event_precision':float(np.mean(event_prec)),
                    'mean_treatment_alarm_rate':float(np.mean(trates)),
                    'mean_baseline_alarm_rate':float(np.mean(rates)),
                    'mean_treatment_random_recall':float(np.mean(rand))})
    pd.DataFrame(entries).to_csv(OUT/'v31_step4_paired_bootstrap.csv',index=False)
    if entries:
        table=pd.DataFrame(entries)
        print(table.to_string(index=False))
        passed=table.loc[(table.folds_won>=3)&(table.eligible_folds>=3)&(~table.experiment_stopped)&(table.ci_low>0),'block'].drop_duplicates().tolist()
        manifest=json.loads((OUT/'feature_sets.json').read_text(encoding='utf-8'))
        blocks=json.loads((OUT/'v31_tda_blocks.json').read_text(encoding='utf-8'))['blocks']
        no_block=not passed
        if no_block:
            vv=pd.read_csv(OUT/'v31_step4_validation.csv')
            vv=vv.loc[(vv.model=='gbm')&(~vv.off_budget)&(~vv.feature_set.eq('BASELINE'))]
            means=vv.groupby('feature_set').excess_event_recall.mean().sort_values(ascending=False)
            fallback=means.index[0] if len(means) else None
            passed=[fallback.split('+',1)[1]] if fallback and '+' in fallback else []
        base=manifest[best]
        selected=[c for b in passed for c in blocks.get(b,[])]
        f4c=list(dict.fromkeys(base+selected))
        (OUT/'v31_F4c.json').write_text(json.dumps({'feature_set':'F4c','no_block_passed':no_block,
            'baseline':best,'selected_blocks':passed,'features':f4c},indent=2))
        print('TDA adds value beyond BEST_CLASSICAL:', 'yes' if not no_block else 'no',
              '(registered block criterion)' if not no_block else '(no block passed; F4c uses best validation block)')

if __name__=='__main__': main()
