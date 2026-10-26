"""Select BEST_CLASSICAL using the completed validation-only RQ grid."""
import json
import pandas as pd
from v3_download_data import OUT

path=OUT/'v31_step3_validation.csv'
if not path.exists(): raise SystemExit('no Step 3 validation rows saved')
d=pd.read_csv(path)
d=d.loc[~d.off_budget]
coverage=d.groupby(['feature_set','model']).fold.nunique()
complete=coverage[coverage>=5].index
d=d.set_index(['feature_set','model']).loc[complete].reset_index() if len(complete) else d.iloc[0:0]
if d.empty: raise SystemExit('no feature-set/model has five on-budget validation folds')
means=d.groupby('feature_set').excess_event_recall.mean().sort_values(ascending=False)
best=str(means.index[0]); value=float(means.iloc[0])
cfg={'feature_set':best,'validation_mean_excess_event_recall':value,
     'eligible_set_means':means.to_dict(),'selection_rule':'maximum mean validation RQ excess_event_recall@5%; require five in-budget folds per model'}
(OUT/'v31_best_classical.json').write_text(json.dumps(cfg,indent=2))
print(f'BEST_CLASSICAL={best}; validation mean excess event recall@5%={value:.4f}')
