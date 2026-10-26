"""Lazy sequence datasets and V5 deep-model training helpers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader,Dataset

from pipeline.models import (AttnBiLSTMModel,BiLSTMModel,GRUModel,LSTMModel,
                             TCNModel,TransformerModel)


class LazyWindows(Dataset):
    """Yield sequence slices from per-asset arrays; never builds N x T x F."""
    def __init__(self, arrays:dict, assets:np.ndarray, positions:np.ndarray,
                 labels:np.ndarray, indices:np.ndarray, seq_len:int=20):
        self.arrays=arrays; self.assets=assets; self.positions=positions
        self.labels=labels; self.indices=np.asarray(indices,dtype=np.int64); self.seq_len=seq_len
    def __len__(self):return len(self.indices)
    def __getitem__(self,ix):
        row=int(self.indices[ix]); asset=self.assets[row]; pos=int(self.positions[row])
        seq=self.arrays[asset][pos-self.seq_len+1:pos+1]
        return torch.from_numpy(seq),torch.tensor(self.labels[row],dtype=torch.float32)


def model_for(name:str,n_features:int,hidden:int)->nn.Module:
    factories={"lstm":lambda:LSTMModel(n_features,hidden),
        "gru":lambda:GRUModel(n_features,hidden),
        "attn_bilstm":lambda:AttnBiLSTMModel(n_features,hidden),
        "bilstm":lambda:BiLSTMModel(n_features,hidden),
        "tcn":lambda:TCNModel(n_features,hidden),
        "transformer":lambda:TransformerModel(n_features,d_model=hidden,heads=4,layers=2)}
    return factories[name]().cpu()


def _predict(model,arrays,assets,positions,labels,indices,batch_size=512):
    ds=LazyWindows(arrays,assets,positions,labels,indices)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in loader:preds.extend(torch.sigmoid(model(xb)).cpu().numpy())
    return np.asarray(preds,np.float32)


def fit_lazy_sequence(name:str,arrays:dict,assets:np.ndarray,positions:np.ndarray,
                      x_rows:np.ndarray,y:np.ndarray,train_idx:np.ndarray,val_idx:np.ndarray,
                      test_idx:np.ndarray,hidden:int,seed:int,epochs:int=12,
                      patience:int=4,batch_size:int=512):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    train_idx=np.asarray(train_idx,dtype=np.int64); val_idx=np.asarray(val_idx,dtype=np.int64)
    test_idx=np.asarray(test_idx,dtype=np.int64)
    pos=train_idx[y[train_idx]==1]; neg=train_idx[y[train_idx]==0]
    if not len(pos):raise ValueError("no positive training sequences")
    neg_used=max(1,int(round(.2*len(neg)))) if len(neg) else 0
    pos_weight=min(neg_used/max(len(pos),1),20.)
    model=model_for(name,x_rows.shape[1],hidden)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32))
    best_ap=-1.; best_state=None; stale=0
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        sampled=rng.choice(neg,size=min(neg_used,len(neg)),replace=False) if len(neg) else np.array([],np.int64)
        chosen=np.r_[pos,sampled]; rng.shuffle(chosen)
        loader=DataLoader(LazyWindows(arrays,assets,positions,y,chosen),
                          batch_size=batch_size,shuffle=True,num_workers=0)
        model.train()
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        valp=_predict(model,arrays,assets,positions,y,val_idx)
        ap=float(average_precision_score(y[val_idx],valp)) if len(val_idx) and y[val_idx].sum() else 0.
        if ap>best_ap+1e-5:
            best_ap=ap; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=patience:break
    if best_state is not None:model.load_state_dict(best_state)
    valp=_predict(model,arrays,assets,positions,y,val_idx)
    testp=_predict(model,arrays,assets,positions,y,test_idx)
    return model,best_ap,valp,testp,{"epochs":epoch+1,"pos_weight":pos_weight,
                                    "train_positive_sequences":len(pos),
                                    "negative_sequences_per_epoch":neg_used}

