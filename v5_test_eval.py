import unittest

import numpy as np
import pandas as pd

from v5_eval import (ERAS, _merge, _runs, alarms_re, assign_eras,
                     blocked_split, evaluate, off_budget)


class V5MetricTests(unittest.TestCase):
    def test_oracle_score_catches_events(self):
        n=2000; y=np.zeros(n,int); y[10:13]=1; y[700:704]=1; y[1500:1502]=1
        f=pd.DataFrame({"asset":"A","date":pd.bdate_range("2001-01-01",periods=n),
                        "y":y,"score":np.where(y,2.,np.linspace(0.,1.,n))})
        alarm,_=alarms_re(f,.05)
        m=evaluate(f,alarm,sims=30)
        self.assertEqual(m["crashes_total"],3)
        self.assertEqual(m["crashes_caught"],3)

    def test_random_scores_approx_random_baseline(self):
        rng=np.random.default_rng(41); rows=[]
        for a in range(8):
            n=180; y=np.zeros(n,int)
            for start in (20,65,110,155): y[start:start+3]=1
            rows.append(pd.DataFrame({"asset":str(a),"date":pd.bdate_range("2000-01-01",periods=n),
                                      "y":y,"score":rng.normal(size=n)}))
        f=pd.concat(rows,ignore_index=True); alarm,_=alarms_re(f,.05)
        m=evaluate(f,alarm,sims=200,seed=99)
        self.assertLess(abs(m["event_recall"]-m["rand_event_recall"]),.15)

    def test_always_alarm_is_off_budget(self):
        self.assertTrue(off_budget(1.0,.05))

    def test_hand_counted_toy(self):
        y=np.array([0,1,1,0,0,0,0,0,0,0,1,0,0]); alarm=np.array([0,1,0,0,0,0,0,0,0,0,0,1,0])
        f=pd.DataFrame({"asset":"A","date":pd.bdate_range("2001-01-01",periods=13),
                        "y":y,"score":np.arange(13,dtype=float)})
        self.assertEqual(len(_merge(_runs(y==1),5)),2)
        m=evaluate(f,alarm,sims=20)
        expected={"n":13,"n_pos":3,"crashes_total":2,"crashes_caught":1,
                  "alarms_total":2,"alarms_correct":1,"alarms_wrong":1,
                  "alarm_day_rate":2/13,"precision":.5,"recall":1/3,
                  "event_recall":.5,"event_precision":.5}
        for key,value in expected.items(): self.assertAlmostEqual(m[key],value)

    def test_blocked_split_embargo_and_validation_separation(self):
        dates=pd.bdate_range("1928-01-03","2026-10-01")
        f=pd.DataFrame({"asset":"A","date":dates,"y":0,"score":0.})
        train,val,test=blocked_split(f,"E5",purge_sessions=55,validation_purge_sessions=40)
        self.assertGreater(len(train),0); self.assertGreater(len(val),0); self.assertGreater(len(test),0)
        eras=assign_eras(f.date).to_numpy(); test_dates=f.iloc[test].date
        test_era=f.iloc[test].date.dt.year
        all_dates=pd.DatetimeIndex(f.date.unique()); pos={d:i for i,d in enumerate(all_dates)}
        lo,hi=pos[test_dates.min()],pos[test_dates.max()]
        for ix in (train,val):
            self.assertTrue(all(pos[d]<lo-55 or pos[d]>hi+55 for d in f.iloc[ix].date))
        tr=f.iloc[train].assign(era=eras[train]); va=f.iloc[val].assign(era=eras[val])
        for name in tr.era.unique():
            left=tr.loc[tr.era.eq(name),"date"]
            right=va.loc[va.era.eq(name),"date"]
            if len(left) and len(right):
                self.assertGreater((right.min()-left.max()).days,0)
                left_pos=max(pos[d] for d in left);right_pos=min(pos[d] for d in right)
                self.assertGreaterEqual(right_pos-left_pos,40)

    def test_era_coverage_exactly_once(self):
        dates=pd.Series(pd.to_datetime(["1928-01-03","1949-12-30","1950-01-02",
            "1979-12-31","1980-01-01","1993-12-31","1994-01-03","1999-12-31",
            "2000-01-03","2003-12-31","2004-01-01","2009-12-31","2010-01-01",
            "2014-12-31","2015-01-01","2017-12-31","2018-01-01","2019-12-31",
            "2020-01-01","2021-12-31","2022-01-01","2026-01-01"]))
        labels=assign_eras(dates)
        self.assertEqual(len(labels),len(dates))
        self.assertEqual(labels.nunique(),len(ERAS))
        self.assertFalse(labels.isna().any())


if __name__=="__main__": unittest.main(verbosity=2)
