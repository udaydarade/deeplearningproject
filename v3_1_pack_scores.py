"""Pack existing per-split score files into the registered combined parquet form."""
from pathlib import Path
import pandas as pd
from v3_download_data import OUT

root=OUT/'scores'; count=0
for v in root.glob('*_val.parquet'):
    stem=v.name[:-len('_val.parquet')]
    t=root/f'{stem}_test.parquet'
    if not t.exists(): continue
    base=stem
    # Keep old V3.1 split suffix in the canonical experiment/set/model/fold name.
    out=root/f'{base}.parquet'
    a=pd.read_parquet(v); b=pd.read_parquet(t)
    if 'split' not in a: a['split']='val'
    if 'split' not in b: b['split']='test'
    pd.concat([a,b],ignore_index=True).sort_values(['split','date','asset'],kind='stable').to_parquet(out,index=False)
    count+=1
print(f'packed {count} per-fold validation/test score pairs')
