"""Choose BASE_SET on validation and create pooled E1 headline tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/"data"/"v5"


def main():
    val=pd.read_csv(OUT/"e1_validation.csv")
    primary=val[(val.scheme=="B")&val.feature_set.isin(["F0","F1","F2"])]
    paired=primary.pivot_table(index=["model","era_or_fold"],columns="feature_set",
                               values="excess_event_recall",aggfunc="mean")
    means=paired.dropna(subset=["F0","F1","F2"]).mean().sort_values(ascending=False)
    base=str(means.index[0]) if len(means) else "F0"
    (OUT/"base_set.json").write_text(json.dumps({"feature_set":base,
        "selection":"highest mean Scheme-B validation excess_event_recall across common model/era cells among F0/F1/F2",
        "common_validation_cells":int(len(paired.dropna(subset=["F0","F1","F2"]))),
        "validation_means":means.to_dict()},indent=2),encoding="utf-8")
    results=pd.read_csv(OUT/"e1_results.csv")
    primary=results[(results.budget=="5%")&(
        ((results.scheme=="B")&(results.rule=="RE"))|
        ((results.scheme=="W")&(results.rule=="RQ")))].copy()
    primary=primary[~primary.off_budget.astype(bool)]
    rows=[]
    for (scheme,fs,model),g in primary.groupby(["scheme","feature_set","model"]):
        crashes=int(g.crashes_total.sum());caught=int(g.crashes_caught.sum())
        alarm_events=int(g.alarms_total.sum());correct=int(g.alarms_correct.sum())
        n=int(g.n.sum());n_pos=int(g.n_pos.sum())
        rand_expected=float((g.rand_event_recall*g.crashes_total).sum())
        alarm_days=float((g.alarm_day_rate*g.n).sum())
        day_tp=float((g.precision*g.alarm_day_rate*g.n).sum())
        event_recall=caught/crashes if crashes else np.nan
        rand=rand_expected/crashes if crashes else np.nan
        day_precision=day_tp/alarm_days if alarm_days else np.nan
        day_recall=day_tp/n_pos if n_pos else np.nan
        prevalence=n_pos/n if n else np.nan
        rows.append({"scheme":scheme,"scheme_label":"blocked CV (uses later eras for training)" if scheme=="B" else "walk-forward",
            "feature_set":fs,"model":model,"folds":g.era_or_fold.nunique(),"n":n,
            "n_pos":n_pos,"crashes_total":crashes,"crashes_caught":caught,
            "alarms_total":alarm_events,"alarms_correct":correct,"alarms_wrong":alarm_events-correct,
            "alarm_day_rate":alarm_days/n if n else np.nan,
            "day_precision":day_precision,"day_recall":day_recall,
            "day_lift":day_precision/prevalence if prevalence and np.isfinite(day_precision) else np.nan,
            "event_recall":event_recall,"event_precision":correct/alarm_events if alarm_events else np.nan,
            "rand_event_recall":rand,"excess_event_recall":event_recall-rand,
            "wrong_per_asset_year":(alarm_events-correct)/(n/252) if n else np.nan,
            "ap":float((g.ap*g.n).sum()/n) if n else np.nan,"auc":float((g.auc*g.n).sum()/n) if n else np.nan,
            "off_budget_folds":int((~g.off_budget.astype(bool)).sum())})
    summary=pd.DataFrame(rows);summary.to_csv(OUT/"e1_headline.csv",index=False)
    b=results[(results.scheme=="B")&(results.rule=="RE")&(results.budget=="5%")&(~results.off_budget.astype(bool))]
    pair=[]
    for model in sorted(b.model.unique()):
        base_rows=b[(b.feature_set=="B3")&(b.model==model)].set_index("era_or_fold")
        for fs in ("F1","F2"):
            one=b[(b.feature_set==fs)&(b.model==model)].set_index("era_or_fold")
            common=base_rows.index.intersection(one.index)
            wins=int((one.loc[common,"excess_event_recall"]>base_rows.loc[common,"excess_event_recall"]).sum())
            pair.append({"model":model,"feature_set":fs,"eras_compared":len(common),"eras_won":wins,
                         "win_share":wins/len(common) if len(common) else np.nan,
                         "beats_b3_in_at_least_60pct":wins/len(common)>=.6 if len(common) else False})
    pd.DataFrame(pair).to_csv(OUT/"e1_f1_f2_vs_b3.csv",index=False)
    print(f"[V5-E1 summary] BASE_SET={base}; Scheme-B validation means={means.round(4).to_dict()}")
    print(f"[V5-E1 summary] pooled 5% RQ/RE headline rows={len(summary)}; on-budget only; see e1_headline.csv")
    print(f"[V5-E1 summary] F1/F2 vs B3 era-win comparisons={len(pair)}; see e1_f1_f2_vs_b3.csv")


if __name__=="__main__":main()
