"""Print V3.1 Step 0 inventory from the preserved V3 snapshot."""
import json
import pandas as pd
from v3_run_experiments import FOLDS
from v3_1_eval import _merge, _runs
from v3_download_data import OUT

def main():
    files=sorted(p.name for p in OUT.iterdir() if p.is_file())
    print('data/v3 files:',', '.join(files))
    print('scripts:',', '.join(sorted(p.name for p in __import__('pathlib').Path('.').glob('v3_*.py'))))
    sets=json.loads((OUT/'feature_sets.json').read_text())
    print('feature groups/manifest columns:',', '.join(f'{k}={len(v)}' for k,v in sets.items()))
    for name in ('F0','F1','F2','F4'):
        cols=sets[name]
        groups={'CSD':sum(c.startswith(('csd_','realized_','ewma_')) for c in cols),
            'price/relative/peer':sum(c.startswith(('price_','relative_','peer_')) for c in cols),
            'market/cluster':sum(c.startswith(('market_','cluster_')) for c in cols),
            'TDA':sum(c.startswith('tda_') for c in cols),
            'asset/region one-hot':sum(c.startswith(('class_','region_')) for c in cols),
            'states':sum(c.startswith('state_') for c in cols)}
        print(f'  {name} groups: '+', '.join(f'{k}={v}' for k,v in groups.items() if v))
    panel=pd.read_parquet(OUT/'panel.parquet',columns=['asset','date','y_BASE'])
    panel.date=pd.to_datetime(panel.date)
    print('BASE label, fold inventory:')
    for first,last in FOLDS:
        end=panel.date.max() if last==9999 else pd.Timestamp(f'{last}-12-31')
        d=panel.loc[(panel.date>=pd.Timestamp(f'{first}-01-01'))&(panel.date<=end)&panel.y_BASE.notna()]
        events=sum(len(_merge(_runs(g.sort_values('date').y_BASE.to_numpy(bool)),5)) for _,g in d.groupby('asset',sort=False))
        print(f'  {first}-{last if last!=9999 else "latest"}: rows={len(d):,}, positives={int(d.y_BASE.sum()):,}, events={events:,}')
    print('A vs D, three lines:')
    print('  EXP-A fits the full panel with seed 1701+fold; EXP-D restricts to registry-built asset universes and fits all_assets with seed 2718+fold.')
    print('  Both call the same tabular F0/LightGBM helper and chronological split function, but their stochastic seeds differ and D applies its own universe filter.')
    print('  Old result CSVs retained aggregates only; with no saved row scores, the recall/alarm-rate gap cannot be decomposed into specific dates or events.')

if __name__=='__main__': main()
