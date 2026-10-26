"""V5 whole-history and walk-forward event evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ERAS = [
    ("E0", 1928, 1949), ("E1", 1950, 1979), ("E2", 1980, 1993),
    ("E3", 1994, 1999), ("E4", 2000, 2003), ("E5", 2004, 2009),
    ("E6", 2010, 2014), ("E7", 2015, 2017), ("E8", 2018, 2019),
    ("E9", 2020, 2021), ("E10", 2022, 9999),
]
W_ERAS = [f"E{i}" for i in range(4, 11)]


def assign_eras(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, first, last in ERAS:
        out.loc[d.dt.year.between(first, last)] = name
    if out.isna().any():
        bad = d.loc[out.isna()]
        raise ValueError(f"dates outside V5 eras: {bad.min()}..{bad.max()}")
    return out


def _date_positions(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, int]]:
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date).drop_duplicates().sort_values())
    return dates, {d: i for i, d in enumerate(dates)}


def blocked_split(frame: pd.DataFrame, test_era: str, purge_sessions: int,
                  deep: bool = False, validation_purge_sessions: int|None=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scheme B: test one era, train on other eras, with date-session embargoes."""
    f = frame.reset_index(drop=True).copy()
    f["date"] = pd.to_datetime(f.date)
    era = assign_eras(f.date).to_numpy()
    if deep and test_era in {"E0", "E1", "E2"}:
        raise ValueError("deep Scheme B is restricted to E3-E10")
    dates, loc = _date_positions(f)
    test_dates = dates[[assign_eras(pd.Series(dates)).to_numpy() == test_era][0]]
    if not len(test_dates):
        return np.array([], int), np.array([], int), np.array([], int)
    lo, hi = loc[test_dates[0]], loc[test_dates[-1]]
    test = np.flatnonzero(era == test_era)
    train_parts, val_parts = [], []
    eligible_eras = ([x[0] for x in ERAS if x[0] in {"E3","E4","E5","E6","E7","E8","E9","E10"}]
                     if deep else [x[0] for x in ERAS])
    embargoed = set(dates[max(0, lo-purge_sessions):min(len(dates), hi+purge_sessions+1)])
    for name in eligible_eras:
        if name == test_era:
            continue
        idx = np.flatnonzero(era == name)
        if not len(idx):
            continue
        idx = idx[~f.iloc[idx].date.isin(embargoed).to_numpy()]
        if not len(idx):
            continue
        era_dates = np.sort(f.iloc[idx].date.unique())
        cut = max(1, int(np.floor(0.8 * len(era_dates))))
        val_start = pd.Timestamp(era_dates[min(cut, len(era_dates)-1)])
        val_dates = era_dates[cut:]
        if not len(val_dates):
            continue
        vstart_pos = loc[pd.Timestamp(val_dates[0])]
        vp=purge_sessions if validation_purge_sessions is None else validation_purge_sessions
        train_end_pos = vstart_pos - vp
        train_date_cut = dates[max(0, train_end_pos)]
        v = idx[f.iloc[idx].date.isin(val_dates).to_numpy()]
        t = idx[(f.iloc[idx].date < train_date_cut).to_numpy()]
        if len(t): train_parts.append(t)
        if len(v): val_parts.append(v)
    train = np.concatenate(train_parts) if train_parts else np.array([], int)
    val = np.concatenate(val_parts) if val_parts else np.array([], int)
    return train, val, test


def walk_forward_split(frame: pd.DataFrame, test_era: str, purge_sessions: int,
                       validation_years: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scheme W: expanding past-only train/validation and a contiguous era test."""
    if test_era not in W_ERAS:
        raise ValueError("Scheme W test eras are E4-E10")
    f = frame.reset_index(drop=True).copy()
    f.date = pd.to_datetime(f.date)
    era = assign_eras(f.date).to_numpy()
    test = np.flatnonzero(era == test_era)
    if not len(test):
        return np.array([], int), np.array([], int), test
    dates, loc = _date_positions(f)
    first_test = f.iloc[test].date.min()
    start_pos = loc[first_test]
    val_end = dates[max(0, start_pos - purge_sessions - 1)]
    val_start = val_end - pd.DateOffset(years=validation_years)
    train_end_pos = max(0, loc.get(pd.Timestamp(val_start), np.searchsorted(dates, val_start)) - purge_sessions)
    train_end = dates[train_end_pos]
    val = np.flatnonzero(((f.date >= val_start) & (f.date <= val_end)).to_numpy())
    train = np.flatnonzero((f.date < train_end).to_numpy())
    return train, val, test


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8)))
    return [(int(edges[i]), int(edges[i+1]-1)) for i in range(0, len(edges), 2)]


def _merge(runs: list[tuple[int, int]], gap: int = 5) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in runs:
        if out and s - out[-1][1] - 1 <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _alarm_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result=[]
    for start, end in _merge(_runs(mask), 5):
        while start <= end:
            stop=min(end,start+14)
            result.append((start,stop)); start=stop+1
    return result


def alarms_re(frame: pd.DataFrame, a: float) -> tuple[np.ndarray, int]:
    """Label-free within-era per-asset threshold; skip assets with <60 rows."""
    f=frame.reset_index(drop=True)
    out=np.zeros(len(f),bool); skipped=0
    for _, idx in f.groupby("asset",sort=False).groups.items():
        ix=np.asarray(list(idx),int); s=f.iloc[ix].score.to_numpy(float)
        finite=np.isfinite(s)
        if finite.sum()<60:
            skipped+=1; continue
        threshold=np.quantile(s[finite],1-a)
        out[ix]=finite & (s>threshold)
    return out,skipped


def alarms_rv(frame: pd.DataFrame, validation: pd.DataFrame, a: float) -> np.ndarray:
    f=frame.reset_index(drop=True); v=validation.copy()
    out=np.zeros(len(f),bool)
    for asset, idx in f.groupby("asset",sort=False).groups.items():
        ix=np.asarray(list(idx),int)
        scores=f.iloc[ix].score.to_numpy(float)
        hist=v.loc[v.asset.eq(asset),"score"].to_numpy(float) if len(v) else np.array([])
        hist=hist[np.isfinite(hist)]
        if len(hist): out[ix]=np.isfinite(scores)&(scores>np.quantile(hist,1-a))
    return out


def alarms_rq(frame: pd.DataFrame, validation: pd.DataFrame, a: float,
              window: int = 250) -> np.ndarray:
    f=frame.reset_index(drop=True).copy(); f.date=pd.to_datetime(f.date)
    v=validation.copy();
    if len(v): v.date=pd.to_datetime(v.date)
    out=np.zeros(len(f),bool)
    for asset, idx in f.groupby("asset",sort=False).groups.items():
        test=f.iloc[np.asarray(list(idx),int)].sort_values("date",kind="stable")
        hist=v.loc[v.asset.eq(asset)].sort_values("date").score.to_numpy(float)[-window:] if len(v) else np.array([])
        hist=hist[np.isfinite(hist)]
        scores=test.score.to_numpy(float); alarm=np.zeros(len(test),bool)
        for j,score in enumerate(scores):
            if len(hist)>=60 and np.isfinite(score): alarm[j]=score>np.quantile(hist[-window:],1-a)
            if np.isfinite(score): hist=np.r_[hist,score]
        out[test.index.to_numpy()]=alarm
    return out


def evaluate(frame: pd.DataFrame, alarm: np.ndarray, sims: int = 200,
             seed: int = 1701) -> dict:
    f=frame.reset_index(drop=True).copy(); f.date=pd.to_datetime(f.date)
    f=f.loc[f.y.notna()].reset_index(drop=True)
    f["alarm"]=np.asarray(alarm,bool)
    if f.duplicated(["asset","date"]).any(): raise ValueError("duplicate asset/date rows")
    crashes=caught=alarm_events=correct=wrong=0; random_caught=0.0
    rng=np.random.default_rng(seed)
    for _,g in f.groupby("asset",sort=False):
        g=g.sort_values("date"); y=g.y.to_numpy(bool); a=g.alarm.to_numpy(bool)
        cr=_merge(_runs(y),5); ar=_alarm_runs(a)
        caught+=sum(bool((a[s:e+1]&y[s:e+1]).any()) for s,e in cr)
        correct+=sum(bool((y[s:e+1]&a[s:e+1]).any()) for s,e in ar)
        crashes+=len(cr); alarm_events+=len(ar)
        n_alarm=int(a.sum())
        if cr and n_alarm:
            draws=np.zeros(sims,float)
            for sim in range(sims):
                ix=rng.choice(len(y),size=min(n_alarm,len(y)),replace=False)
                ra=np.zeros(len(y),bool); ra[ix]=True
                draws[sim]=sum(bool((ra[s:e+1]&y[s:e+1]).any()) for s,e in cr)
            random_caught += float(draws.mean())
    n=len(f); npos=int(f.y.sum()); n_alarm=int(f.alarm.sum())
    event_recall=caught/crashes if crashes else np.nan
    event_precision=correct/alarm_events if alarm_events else np.nan
    rand_recall=random_caught/crashes if crashes else np.nan
    day_precision=float((f.alarm&f.y.astype(bool)).sum()/n_alarm) if n_alarm else np.nan
    prevalence=npos/n if n else np.nan
    try:
        ap=float(average_precision_score(f.y,f.score)) if npos else np.nan
        auc=float(roc_auc_score(f.y,f.score)) if f.y.nunique()==2 else np.nan
    except ValueError: ap=auc=np.nan
    asset_years=f.groupby("asset").size().sum()/252
    return {"n":n,"prevalence":prevalence,"n_pos":npos,"crashes_total":crashes,
            "crashes_caught":caught,"alarms_total":alarm_events,"alarms_correct":correct,
            "alarms_wrong":alarm_events-correct,"alarm_day_rate":n_alarm/n if n else np.nan,
            "precision":day_precision,"recall":float((f.alarm&f.y.astype(bool)).sum()/npos) if npos else np.nan,
            "day_lift":day_precision/prevalence if prevalence and np.isfinite(day_precision) else np.nan,
            "wrong_per_asset_year":(alarm_events-correct)/asset_years if asset_years else np.nan,
            "event_recall":event_recall,"event_precision":event_precision,
            "rand_event_recall":rand_recall,
            "excess_event_recall":event_recall-rand_recall if np.isfinite(event_recall) else np.nan,
            "ap":ap,"auc":auc}


def off_budget(rate: float, a: float) -> bool:
    return (not np.isfinite(rate)) or rate < .5*a or rate > 2*a
