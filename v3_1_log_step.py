"""Append a compact V3.1 experiment entry to the existing experiment log."""
import json, sys
from pathlib import Path
import pandas as pd

step=sys.argv[1]
root=Path('data/v3')
log=Path('GLM_EXPERIMENT_LOG.md')
marker=f'## EXP-V3.1-Step-{step} '
content=log.read_text(encoding='utf-8')
if marker in content:
    print(f'{marker}already logged'); raise SystemExit(0)
lines=[f'\n{marker}({pd.Timestamp.now().date()})']
path=root/f'v31_step{step}.csv'
if path.exists():
    d=pd.read_csv(path)
    x=d[(d.rule=='RQ')&(d.alarm_budget==.05)]
    on=x[~x.off_budget]
    lines.append(f'- Recorded rows: {len(d)}; completed folds/cells: {x[["feature_set","model","fold"]].drop_duplicates().shape[0]}; on-budget headline rows: {len(on)}/{len(x)}.')
    if len(on):
        cols=[c for c in ('excess_event_recall','event_recall','event_precision','day_lift','rand_event_recall','realized_alarm_day_rate') if c in on]
        by=on.groupby(['feature_set','model'])[cols].mean().round(4)
        lines.append('- On-budget RQ@5% means by feature set/model: '+ '; '.join(f'{i[0]}/{i[1]} '+', '.join(f'{k}={v}' for k,v in r.items()) for i,r in by.iterrows())+'.')
if step=='3' and (root/'v31_best_classical.json').exists():
    cfg=json.loads((root/'v31_best_classical.json').read_text()); lines.append(f'- BEST_CLASSICAL selected on validation-only RQ@5% excess event recall: {cfg["feature_set"]} (mean={cfg["validation_mean_excess_event_recall"]:.4f}).')
if step=='4' and (root/'v31_step4_paired_bootstrap.csv').exists():
    b=pd.read_csv(root/'v31_step4_paired_bootstrap.csv'); passed=b[(b.folds_won>=3)&(b.ci_low>0)]
    lines.append('- Paired TDA-block bootstrap table: `data/v3/v31_step4_paired_bootstrap.csv`; blocks passing the preregistered rule: '+(', '.join(passed.block.unique()) if len(passed) else 'none')+'.')
    if (root/'v31_F4c.json').exists():
        cfg=json.loads((root/'v31_F4c.json').read_text()); lines.append(f'- F4c selected blocks: {cfg["selected_blocks"]}; no_block_passed={cfg["no_block_passed"]}.')
log.open('a',encoding='utf-8').write('\n'.join(lines)+'\n')
print('\n'.join(lines))
