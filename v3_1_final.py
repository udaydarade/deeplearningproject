"""V3.1 blocked-CV robustness and final GBM export (Steps 8, P2)."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from v3_download_data import OUT
from v3_run_experiments import _tabular_view
from v3_1_train import joined, eval_pair, append_results
from v3_1_eval import _merge, _runs, alarm_days, evaluate

ERAS=[('pre1995',None,'1994-12-31'),('1995-2001','1995-01-01','2001-12-31'),
      ('2002-2008','2002-01-01','2008-12-31'),('2009-2014','2009-01-01','2014-12-31'),
      ('2015-2019','2015-01-01','2019-12-31'),('2020_plus','2020-01-01',None)]

def blocked():
    panel=pd.read_parquet(OUT/'panel.parquet'); shared=pd.read_parquet(OUT/'market_features.parquet')
    manifest=json.loads((OUT/'feature_sets.json').read_text())
    best=json.loads((OUT/'v31_best_classical.json').read_text())['feature_set']
    f4c=json.loads((OUT/'v31_F4c.json').read_text())['features']
    configs={'BEST_CLASSICAL':manifest[best],'F4c':f4c}
    all_rows=[]
    for name,features in configs.items():
        data=joined(panel,shared,features)
        for fi,(era,start,end) in enumerate(ERAS):
            test=data.y_BASE.notna()
            if start: test &= data.date.ge(pd.Timestamp(start))
            if end: test &= data.date.le(pd.Timestamp(end))
            # Test retains the full held-out era; embargo removes nearby dates
            # only from the training pool.
            train=data.y_BASE.notna() & ~data.date.between(pd.Timestamp(start) if start else data.date.min(),pd.Timestamp(end) if end else data.date.max())
            if start: train &= ~data.date.between(pd.Timestamp(start)-pd.offsets.BDay(55),pd.Timestamp(start)+pd.offsets.BDay(55))
            if end: train &= ~data.date.between(pd.Timestamp(end)-pd.offsets.BDay(55),pd.Timestamp(end)+pd.offsets.BDay(55))
            tr=np.flatnonzero(train.to_numpy()); te=np.flatnonzero(test.to_numpy())
            if not len(tr) or not len(te):
                print(f'blocked skip {name}/{era}: train={len(tr)} test={len(te)}'); continue
            y=data.y_BASE.fillna(0).to_numpy(np.int8); x,names=_tabular_view(data,features)
            npos=max(int(y[tr].sum()),1); nneg=max(len(tr)-npos,1)
            model=lgb.LGBMClassifier(objective='binary',learning_rate=.03,num_leaves=15,
                min_child_samples=500,feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,
                reg_lambda=5,scale_pos_weight=min(nneg/npos,10),n_estimators=600,max_bin=63,
                n_jobs=8,random_state=8800+fi,verbosity=-1)
            model.fit(x[tr],y[tr])
            testf=data.loc[test,['date','asset','asset_class','source_dataset','y_BASE']].copy()
            testf['y']=testf.pop('y_BASE').astype(np.int8); testf['score']=model.predict_proba(x[te])[:,1]
            # Scheme-B RQ uses available earlier held/validation history from this fitted model.
            hist=data.loc[data.date.lt(testf.date.min()),['date','asset','y_BASE','asset_class','source_dataset']].copy()
            hist=hist.loc[hist.y_BASE.notna()].copy(); hist['y']=hist.pop('y_BASE').astype(np.int8)
            if len(hist): hist['score']=model.predict_proba(x[np.flatnonzero(data.index.isin(hist.index))])[:,1]
            rows=[]
            for a in (.02,.05,.10):
                alarms=alarm_days(testf,'RQ',a,validation=hist)
                m=evaluate(testf,alarms,sims=200,seed=8800+fi)
                off=not (.5*a <= m['realized_alarm_day_rate'] <= 2*a)
                rows.append({'step':'8','scheme':'B','label_N':15,'label_p':.02,
                    'feature_set':name,'model':'gbm','rule':'RQ','alarm_budget':a,
                    'fold':era,'seed':8800+fi,'subset':'all','off_budget':off,**m})
            all_rows.extend(rows)
            append_results(rows,'v31_step8_blocked.csv')
            print(f'blocked {name}/{era}: train={len(tr)} test={len(te)} positives={int(y[te].sum())}',flush=True)

def final_model():
    panel=pd.read_parquet(OUT/'panel.parquet'); shared=pd.read_parquet(OUT/'market_features.parquet')
    manifest=json.loads((OUT/'feature_sets.json').read_text())
    best=json.loads((OUT/'v31_best_classical.json').read_text())['feature_set']
    config={'feature_set':best,'features':manifest[best],
            'selection_metric':'mean validation RQ excess_event_recall@5%'}
    valfile=OUT/'v31_step4_validation.csv'
    blockfile=OUT/'v31_tda_blocks.json'
    if valfile.exists() and blockfile.exists():
        vv=pd.read_csv(valfile); vv=vv.loc[(vv.model=='gbm')&(~vv.off_budget)]
        means=vv.groupby('feature_set').excess_event_recall.mean().sort_values(ascending=False)
        blocks=json.loads(blockfile.read_text())['blocks']
        for name in means.index:
            if name==best: break
            if '+' in name:
                block=name.split('+',1)[1]
                if block in blocks and blocks[block]:
                    config={'feature_set':name,'features':list(dict.fromkeys(manifest[best]+blocks[block])),
                            'selection_metric':'mean validation RQ excess_event_recall@5%'}
                    break
    features=config['features']; data=joined(panel,shared,features); train=data.y_BASE.notna().to_numpy()
    y=data.y_BASE.fillna(0).to_numpy(np.int8); x,names=_tabular_view(data,features)
    npos=max(int(y[train].sum()),1); nneg=max(int(train.sum())-npos,1)
    model=lgb.LGBMClassifier(objective='binary',learning_rate=.03,num_leaves=15,
        min_child_samples=500,feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,
        reg_lambda=5,scale_pos_weight=min(nneg/npos,10),n_estimators=600,max_bin=63,n_jobs=8,
        random_state=9901,verbosity=-1).fit(x[train],y[train])
    scores=model.predict_proba(x)[:,1].astype(np.float32)
    data['score']=scores
    hist=(data.sort_values(['asset','date']).groupby('asset',sort=False).tail(250)
          [['date','asset','score']].copy())
    dest=OUT/'final_model'; dest.mkdir(exist_ok=True)
    joblib.dump(model,dest/'model.joblib'); joblib.dump(None,dest/'scaler.joblib')
    (dest/'feature_list.json').write_text(json.dumps({'features':features,'expanded_features':names,
        'model':'LightGBM','scaler':None,'rule':'RQ a=0.05, past-only per-asset 250 sessions'},indent=2))
    hist.to_parquet(dest/'score_history.parquet',index=False)
    print(f'final model saved: {dest}; rows={len(data):,}; features={len(features)}')

if __name__=='__main__':
    blocked(); final_model()
