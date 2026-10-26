import unittest

import numpy as np

from pipeline_v5.price1d import persistence_bars,price1d_window


def brute_bars(values):
    x=np.asarray(values,float); active=np.zeros(len(x),bool); comps=[]; bars=[]
    for level in np.sort(np.unique(x)):
        active |= x==level
        labels=np.full(len(x),-1,int); current=[]; cid=0
        i=0
        while i<len(x):
            if not active[i]:i+=1;continue
            j=i
            while j+1<len(x) and active[j+1]:j+=1
            ids=np.arange(i,j+1); labels[ids]=cid
            current.append(ids);cid+=1;i=j+1
        old_births=[]
        for old_ids,birth in comps:
            mapped=np.unique(labels[old_ids])
            for target in mapped:
                if target>=0:old_births.append((int(target),float(birth)))
        next_comps=[]
        for k,ids in enumerate(current):
            births=[b for target,b in old_births if target==k]
            # A newly born vertex contributes a component at this level only if
            # it was not already connected to a pre-existing component.
            n_new=int(np.sum((x[ids]==level)&~np.isin(ids,np.concatenate([c[0] for c in comps]) if comps else np.array([],int))))
            births.extend([float(level)]*n_new)
            if not births:births=[float(level)]
            elder=min(births)
            for birth in births:
                if birth>elder:bars.append((birth,float(level)))
            next_comps.append((ids,elder))
        comps=next_comps
    out=np.asarray(bars,float).reshape(-1,2)
    return out[out[:,1]>out[:,0]] if len(out) else out


class Price1DTests(unittest.TestCase):
    def test_required_example_valley(self):
        np.testing.assert_allclose(persistence_bars(np.array([0.,2.,1.,3.])),[[1.,2.]])

    def test_required_example_peak(self):
        np.testing.assert_allclose(persistence_bars(np.array([1.,0.,2.,.5,3.])),[[.5,2.]])

    def test_200_random_windows_against_brute_force(self):
        rng=np.random.default_rng(711)
        for _ in range(200):
            x=rng.normal(size=int(rng.integers(4,31)))
            a=persistence_bars(x); b=brute_bars(x)
            a=np.array(sorted((round(float(d-b),10) for b,d in a)))
            c=np.array(sorted((round(float(d-b),10) for b,d in b)))
            np.testing.assert_allclose(a,c,atol=1e-10,rtol=0)

    def test_price_ratios_scale_free(self):
        r=np.random.default_rng(55).normal(0,.01,59); p=np.r_[0,np.cumsum(r)]
        a=price1d_window(p,r,60,"x")
        b=price1d_window(3*p,3*r,60,"x")
        for suffix in ("bars_over_scale","max_over_scale"):
            for kind in ("valley","peak"):
                self.assertAlmostEqual(a[f"x_{kind}_{suffix}"],b[f"x_{kind}_{suffix}"])


if __name__=="__main__":unittest.main(verbosity=2)
