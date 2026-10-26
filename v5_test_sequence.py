import unittest

import numpy as np

from pipeline_v5.sequence import LazyWindows,fit_lazy_sequence,model_for


class LazySequenceTests(unittest.TestCase):
    def test_all_required_architectures_accept_lazy_batch(self):
        import torch
        for name in ("lstm","gru","attn_bilstm","tcn","transformer"):
            model=model_for(name,4,24)
            out=model(torch.zeros((2,20,4),dtype=torch.float32))
            self.assertEqual(tuple(out.shape),(2,))

    def test_lazy_window_shape_and_fit_smoke(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(120,4)).astype(np.float32)
        y=np.zeros(120,np.int8);y[20:23]=1;y[45:48]=1;y[86:89]=1;y[105:108]=1
        assets=np.full(120,"A",dtype=object);pos=np.arange(120,dtype=np.int32)
        ds=LazyWindows({"A":x},assets,pos,y,np.array([19,20,21]))
        self.assertEqual(tuple(ds[0][0].shape),(20,4))
        tr=np.arange(19,80);va=np.arange(80,100);te=np.arange(100,120)
        _,ap,vp,tp,detail=fit_lazy_sequence("gru",{"A":x},assets,pos,x,y,tr,va,te,
                                             hidden=24,seed=19,epochs=1,patience=1,batch_size=32)
        self.assertEqual(len(vp),len(va));self.assertEqual(len(tp),len(te))
        self.assertEqual(detail["negative_sequences_per_epoch"],int(round(.2*np.sum(y[tr]==0))))
        self.assertGreaterEqual(ap,0.)


if __name__=="__main__":unittest.main(verbosity=2)
