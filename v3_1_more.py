"""V3.1 Steps 5–7: homology/scale ablations, universes, and label sensitivity."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd

from v3_download_data import OUT
from v3_run_experiments import FOLDS
from v3_1_train import joined, fit_scores, write_scores, eval_pair, append_results
from v3_1_blocks import build_blocks


def fit_grid(panel, shared, named_features, model, step, resume=False):
    for name,features in named_features.items():
        data=joined(panel,shared,features)
        off_count=0; stopped=False
        for fi,(first,last) in enumerate(FOLDS):
            if stopped: break
            fold=f"{first}-{last if last!=9999 else 'latest'}"
            safe=name.replace("+","_").replace("/","_")
            stem=f"step{step}_{safe}_{model}_fold{fi}"
            pv=OUT/"scores"/f"{stem}_val.parquet"; pt=OUT/"scores"/f"{stem}_test.parquet"
            if resume and pv.exists() and pt.exists():
                val=pd.read_parquet(pv); test=pd.read_parquet(pt)
                print(f"reuse scores {stem}",flush=True)
            else:
                _,val,test=fit_scores(data,features,first,last,model,3600+fi)
                write_scores(f"step{step}",safe,model,fi,val,test)
            rows=eval_pair(val,test,step=step,feature_set=name,model=model,
                fold=fold,fold_i=fi,seed=3600)
            append_results(rows,f'v31_step{step}.csv')
            off_count+=sum(r['off_budget'] for r in rows if r['rule']=='RQ' and r['alarm_budget']==.05)
            if off_count>1:
                stopped=True; print(f'STOP {name}/{model}: >20% RQ@5% folds off-budget',flush=True)
            print(f"step{step} {safe}/{model}/{fold}",flush=True)


def step5(resume):
    panel=pd.read_parquet(OUT/"panel.parquet"); shared=pd.read_parquet(OUT/"market_features.parquet")
    manifest=json.loads((OUT/"feature_sets.json").read_text(encoding="utf-8"))
    block_manifest=json.loads((OUT/"v31_tda_blocks.json").read_text(encoding="utf-8"))['blocks']
    best=json.loads((OUT/"v31_best_classical.json").read_text())['feature_set']; base=manifest[best]
    f4=manifest['F4']
    variants={"BEST_CLASSICAL":base}
    # Cloud-size: A and T H1 scalar summaries at w60/w120 for all available k.
    for cloud in ('A','T'):
        for k in ('6','12','24','48','full'):
            cols=[c for c in f4 if c.startswith(f'tda_{cloud}_k{k}_') and '_h1_' in c
                  and any(x in c for x in ('totalpers','maxpers','entropy','bar_count','landscape_l1','landscape_l2'))
                  and not any(c.endswith(s) for s in ('_slope5','_std5','_ac15','_z250','_velocity'))]
            if cols: variants[f'{cloud}_size{k}']=base+cols
    # Homology dimension: A/T H1, H0+H1, and H0+H1+H2 using scale k12,w60/120 summaries.
    for cloud in ('A','T'):
        for hs in ((1,),(0,1),(0,1,2)):
            cols=[]
            for h in hs:
                cols += [c for c in f4 if c.startswith(f'tda_{cloud}_k12_') and '_h'+str(h)+'_' in c
                         and ('_w60_' in c or '_w120_' in c)
                         and any(x in c for x in ('totalpers','maxpers','entropy','bar_count','landscape_l1','landscape_l2'))
                         and not any(c.endswith(s) for s in ('_slope5','_std5','_ac15','_z250','_velocity'))]
            if cols: variants[f'{cloud}_homology_{"".join(map(str,hs))}']=base+cols
        # Dynamics versus static at a small H1 block.
        static=[c for c in f4 if c.startswith(f'tda_{cloud}_k24_w60_h1_') and any(x in c for x in ('totalpers','landscape_l2')) and not any(c.endswith(s) for s in ('_slope5','_std5','_ac15','_z250','_velocity'))]
        dyn=[c for c in f4 if c.startswith(f'tda_{cloud}_k24_w60_h1_') and any(c.endswith(s) for s in ('_slope5','_std5','_ac15','_z250','_velocity'))]
        if static: variants[f'{cloud}_H1_static']=base+static
        if static and dyn: variants[f'{cloud}_H1_with_dynamics']=base+static+dyn
    for cols in variants.values():
        if len(set(cols)-set(base))>60: raise ValueError('Step 5 TDA add-on exceeds 60 columns')
    (OUT/'v31_step5_config.json').write_text(json.dumps({k:len(set(v)-set(base)) for k,v in variants.items()},indent=2))
    fit_grid(panel,shared,variants,'gbm','5',resume)


def step6(resume):
    panel=pd.read_parquet(OUT/'panel.parquet'); shared=pd.read_parquet(OUT/'market_features.parquet')
    reg=pd.read_csv(OUT/'registry.csv'); manifest=json.loads((OUT/'feature_sets.json').read_text())
    best=json.loads((OUT/'v31_best_classical.json').read_text())['feature_set']; features=manifest[best]
    assets=reg.loc[~reg.is_state.fillna(False)].copy()
    eq=assets.loc[assets.asset_class.eq('equity_index')].sort_values('n_obs',ascending=False)
    etf=assets.loc[assets.asset_class.eq('us_etf')].sort_values('n_obs',ascending=False)
    universes={'us4':{'^GSPC','^DJI','^IXIC','^RUT'},
      'approx30_index_etf':set(eq.head(20).ticker)|set(etf.head(10).ticker),
      'all_nonstock':set(assets.loc[assets.asset_class.ne('stock'),'ticker']),
      'all_assets':set(assets.ticker)}
    result=[]
    for uname,tickers in universes.items():
        sub=panel.loc[panel.asset.isin(tickers)].copy()
        data=joined(sub,shared,features)
        off_counts={'all':0,'us4':0}; stopped_views=set()
        for fi,(first,last) in enumerate(FOLDS):
            fold=f"{first}-{last if last!=9999 else 'latest'}"
            _,val,test=fit_scores(data,features,first,last,'gbm',4900+fi)
            write_scores(f"step6_{uname}",best,'gbm',fi,val,test)
            for view in ('all','us4'):
                if view in stopped_views: continue
                rows=eval_pair(val,test,step='6',feature_set=best,model='gbm',fold=fold,
                    fold_i=fi,seed=4900,subset=view)
                for row in rows: row['universe']=uname; row['test_view']=view
                result.extend(rows)
                append_results(rows,'v31_step6.csv')
                off_counts[view]+=sum(r['off_budget'] for r in rows if r['rule']=='RQ' and r['alarm_budget']==.05)
                if off_counts[view]>1:
                    stopped_views.add(view); print(f'STOP {uname}/{view}: >20% RQ@5% folds off-budget',flush=True)
            print(f"step6 {uname}/{fold} train_assets={sub.asset.nunique()}",flush=True)
    # Explicit >=1997 history cut, folds 2–5 only as preregistered for comparison.
    sub=panel.loc[panel.asset.isin(universes['all_assets']) & (panel.date>=pd.Timestamp('1997-01-01'))].copy()
    data=joined(sub,shared,features); hist=[]
    for fi in range(1,5):
        first,last=FOLDS[fi]; fold=f"{first}-{last if last!=9999 else 'latest'}"
        _,val,test=fit_scores(data,features,first,last,'gbm',5100+fi)
        hist.extend(eval_pair(val,test,step='6_history1997',feature_set=best,model='gbm',fold=fold,fold_i=fi,seed=5100))
    append_results(hist,'v31_step6_history.csv')


def step7(resume):
    panel=pd.read_parquet(OUT/'panel.parquet'); shared=pd.read_parquet(OUT/'market_features.parquet')
    labels=pd.read_parquet(OUT/'labels.parquet')
    manifest=json.loads((OUT/'feature_sets.json').read_text())
    val=pd.read_csv(OUT/'v31_step3_validation.csv')
    best2=list(val.loc[~val.off_budget].groupby('feature_set').excess_event_recall.mean().sort_values(ascending=False).head(2).index)
    specs=[(f'N{n}_p2',f'y_N{n}_p2',n,.02) for n in (5,10,30)] + [(f'N15_p{p}',f'y_N15_p{p}',15,p/100) for p in (1,5,10)]
    for fs in best2:
        for tag,label,n,p in specs:
            data=joined(panel,shared,manifest[fs]).drop(columns=['y_BASE'])
            lab=labels[['asset','date',label]].rename(columns={label:'y_BASE'})
            data=data.merge(lab,on=['asset','date'],how='left',validate='one_to_one')
            off_count=0
            for fi,(first,last) in enumerate(FOLDS):
                _,v,t=fit_scores(data,manifest[fs],first,last,'gbm',6200+fi)
                write_scores(f'step7_{tag}',fs,'gbm',fi,v,t)
                rows=eval_pair(v,t,step='7',feature_set=fs,model='gbm',fold=f'{first}-{last if last!=9999 else "latest"}',fold_i=fi,seed=6200,label_n=n,label_p=p)
                append_results(rows,'v31_step7.csv')
                print(f'step7 {fs}/{tag}/fold{fi}',flush=True)
                off_count+=sum(r['off_budget'] for r in rows if r['rule']=='RQ' and r['alarm_budget']==.05)
                if off_count>1:
                    print(f'STOP {fs}/{tag}: >20% RQ@5% folds off-budget',flush=True)
                    break


def main():
    p=argparse.ArgumentParser(); p.add_argument('step',choices=['5','6','7']); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    {'5':step5,'6':step6,'7':step7}[a.step](a.resume)

if __name__=='__main__': main()
