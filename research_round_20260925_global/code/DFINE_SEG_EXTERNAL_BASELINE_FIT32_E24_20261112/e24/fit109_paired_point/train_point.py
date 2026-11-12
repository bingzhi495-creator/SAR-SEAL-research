"""Matched source-group supervision for equal-size residual point heads."""
from hashlib import sha256

import numpy as np
import torch
from torch import nn


HEADS=("independent","shuffled","paired")
WIDTH=24
STEPS=200
BATCH=64


class PointHead(nn.Module):
    def __init__(self,dimensions):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(2*dimensions,WIDTH),nn.ReLU(),
                                   nn.Linear(WIDTH,WIDTH),nn.ReLU(),nn.Linear(WIDTH,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,own,partner,probabilities):
        residual=self.network(torch.cat((own,partner),dim=-1)).squeeze(-1)
        return torch.logit(probabilities.clamp(1e-4,1-1e-4))+residual


def group_moments(own,indices):
    first=np.asarray([own[ix].mean(axis=(0,1),dtype=np.float64) for ix in indices]).mean(axis=0)
    second=np.asarray([np.square(own[ix],dtype=np.float64).mean(axis=(0,1)) for ix in indices]).mean(axis=0)
    scale=np.sqrt(np.maximum(0.,second-first**2))
    scale[scale<1e-12]=1.
    return first.astype(np.float32),scale.astype(np.float32)


def train(slots,fold):
    train_slots=[s for s in slots if s["fold"]!=fold and s["truth"] is not None and s["data"] is not None]
    group_map={}
    end=0
    for s in train_slots:
        n=len(s["truth"])
        group_map.setdefault(s["group"],[]).extend(range(end,end+n))
        end+=n
    groups=[np.asarray(group_map[k]) for k in sorted(group_map)]
    own=np.concatenate([s["data"]["features"] for s in train_slots])
    q=np.concatenate([s["truth"] for s in train_slots]).astype(np.float32)
    center,scale=group_moments(own,groups)
    standard=np.asarray((own-center)/scale,dtype=np.float32)
    probabilities=torch.from_numpy(own[:,:,0].copy())
    targets=torch.from_numpy(q)
    own_t=torch.from_numpy(standard)
    rng=np.random.Generator(np.random.PCG64(20261112+fold))
    draws=rng.integers(len(groups),size=(STEPS,BATCH))
    batches=np.asarray([[groups[g][rng.integers(len(groups[g]))] for g in row] for row in draws],dtype=np.int64)
    digest=sha256(batches.astype("<i8",copy=False).tobytes()).hexdigest()
    models={}
    audit={}
    for arm in HEADS:
        partner=np.concatenate([s["data"]["contexts"][arm] for s in train_slots])
        partner_t=torch.from_numpy(np.asarray((partner-center)/scale,dtype=np.float32))
        torch.manual_seed(5217+fold)
        model=PointHead(own.shape[2])
        start=sha256(b"".join(t.detach().numpy().tobytes() for t in model.parameters())).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        loss_fn=nn.BCEWithLogitsLoss()
        for indices in batches:
            opt.zero_grad(set_to_none=True)
            logits=model(own_t[indices],partner_t[indices],probabilities[indices])
            loss=loss_fn(logits,targets[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError("Finite original CPU training loss")
            loss.backward()
            opt.step()
        steps={int(v["step"].item()) for v in opt.state.values()}
        if steps!={STEPS}:
            raise ValueError("Exact observed optimizer update count")
        audit[arm]=dict(steps=STEPS,observed_pair_examples=STEPS*BATCH,
                        observed_point_examples=2*STEPS*BATCH,parameters=sum(x.numel() for x in model.parameters()),
                        initial_tensor_sha256=start,batch_sha256=digest,last_batch_loss=float(loss.detach()))
        models[arm]=model.eval()
    if len({a["initial_tensor_sha256"] for a in audit.values()})!=1 or len({a["parameters"] for a in audit.values()})!=1:
        raise ValueError("Exact equal initialization and architecture")
    return models,center,scale,dict(groups=len(groups),unique_train_pairs=len(q),heads=audit)


def gains(model,features,partner,center,scale):
    own=torch.from_numpy(np.asarray((features-center)/scale,dtype=np.float32))
    context=torch.from_numpy(np.asarray((partner-center)/scale,dtype=np.float32))
    with torch.inference_mode():
        points=model(own,context,torch.from_numpy(features[:,:,0].copy())).sigmoid().numpy()
    return points[:,0]-points[:,1]