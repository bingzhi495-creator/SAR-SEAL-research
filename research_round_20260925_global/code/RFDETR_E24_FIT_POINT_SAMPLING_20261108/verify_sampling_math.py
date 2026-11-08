"""Use autograd and exhaustive ordered draws to check the analytic census."""
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
from torch.nn import functional as F

from sampling_risk import BilinearRisk


def main():
    coarse=(2,3)
    target_shape=(2,2)
    z=torch.tensor([[.6,-.4,1.3],[.7,0.,-.6]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[True,True],[False,False]])
    band=np.array([[True,False],[True,False]])
    axis=torch.tensor([-.5,.5],dtype=torch.float64)
    gy,gx=torch.meshgrid(axis,axis,indexing="ij")
    logits=F.grid_sample(z[None,None],torch.stack((gx,gy),dim=-1)[None],align_corners=False)[0,0]
    direct=F.binary_cross_entropy_with_logits(logits,y.double(),reduction="none")
    vectors=np.stack([torch.autograd.grad(loss,z,retain_graph=True)[0].detach().numpy().ravel()
                      for loss in direct.ravel()])
    dense=vectors.mean(axis=0)
    boundary=(vectors*band.ravel()[:,None]).mean(axis=0)
    raw_grid=range(vectors.shape[0])
    selected=np.flatnonzero(band.ravel())
    other=np.flatnonzero(~band.ravel())
    sums=dict(global_uniform=[],global_stratified=[],boundary_uniform=[],boundary_stratified=[])
    for i,j in product(raw_grid,repeat=2):
        pred=vectors[[i,j]].mean(axis=0)
        pred_b=(vectors[[i,j]]*band.ravel()[[i,j],None]).mean(axis=0)
        sums["global_uniform"].append(np.sum((pred-dense)**2))
        sums["boundary_uniform"].append(np.sum((pred_b-boundary)**2))
    for i,j in product(selected,other):
        pred=(vectors[i]+vectors[j])*.5
        pred_b=vectors[i]*.5
        sums["global_stratified"].append(np.sum((pred-dense)**2))
        sums["boundary_stratified"].append(np.sum((pred_b-boundary)**2))
    probe=BilinearRisk(coarse,target_shape)
    actual=probe.measure(z.detach().numpy(),y.numpy(),band,2)
    np.testing.assert_allclose(probe.dense_logits(z.detach().numpy()),logits.detach().numpy(),rtol=0,atol=1e-14)
    for name,mse in sums.items():
        np.testing.assert_allclose(actual[name],np.mean(mse),rtol=0,atol=1e-14)
    print(json.dumps(dict(status="AUTOGRAD_ALL_FOUR_VARIANCE_TRACES_VERIFIED",ordered_uniform_draws=16,
        ordered_stratified_draws=4,GPU=False,real_data=False)))


if __name__=="__main__":
    main()
