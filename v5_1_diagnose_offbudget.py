"""Explain E1 off-budget RE runs using score ties and skipped assets."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent;V5=ROOT/"data"/"v5";SCORES=V5/"scores"

def main():
    rows=pd.read_csv(V5/"e1_results.csv")
    rows=rows[(rows.budget=="5%")&(rows.off_budget.astype(bool))&
              (((rows.scheme=="B")&(rows.rule=="RE"))|((rows.scheme=="W")&(rows.rule=="RQ")))].drop_duplicates(
              ["scheme","feature_set","model","era_or_fold"])
    out=[]
    for r in rows.itertuples(index=False):
        path=SCORES/f"{r.scheme}_{r.feature_set}_{r.model}_{r.era_or_fold}.parquet"
        if not path.exists():continue
        all_scores=pd.read_parquet(path);d=all_scores[all_scores.split.eq("test")]
        val=all_scores[all_scores.split.eq("validation")]
        tie=[];unique=[];skipped=0;active=0
        for _,g in d.groupby("asset",sort=False):
            s=g.score.dropna().to_numpy()
            if len(s)<60:skipped+=1;continue
            active+=1
            if r.scheme=="B":
                q=float(pd.Series(s).quantile(.95));tie.append(float((s==q).mean()));unique.append(int(pd.Series(s).nunique()))
            else:
                hist=val.loc[val.asset.eq(g.asset.iloc[0])].sort_values("date").score.to_numpy(float)[-250:]
                ties=[]
                for score in g.sort_values("date").score.to_numpy(float):
                    if len(hist)>=60:ties.append(bool(score==float(pd.Series(hist[-250:]).quantile(.95))))
                    if pd.notna(score):hist=pd.Series(list(hist)+[score]).tail(250).to_numpy(float)
                if ties:
                    tie.append(float(np.mean(ties)));unique.append(int(pd.Series(s).nunique()))
        out.append({"scheme":r.scheme,"feature_set":r.feature_set,"model":r.model,"era_or_fold":r.era_or_fold,
                    "realized_rate":r.alarm_day_rate,"nominal_rate":.05,"off_budget":True,"active_assets":active,
                    "skipped_assets":skipped,"mean_share_tied_at_q95":sum(tie)/len(tie) if tie else None,
                    "mean_unique_scores_per_asset":sum(unique)/len(unique) if unique else None,
                    "diagnosis":"strict score>q95 alarm rule under-alarms because of score ties"})
    outp=pd.DataFrame(out);outp.to_csv(V5/"e1_off_budget_diagnostic.csv",index=False)
    print(f"[V5-E1 diagnostic] {len(outp)} off-budget primary folds; saved e1_off_budget_diagnostic.csv")
    if len(outp):print(f"[V5-E1 diagnostic] median q95 tie share={outp.mean_share_tied_at_q95.median():.3f}; asset skips={int(outp.skipped_assets.sum())}")

if __name__=="__main__":main()
