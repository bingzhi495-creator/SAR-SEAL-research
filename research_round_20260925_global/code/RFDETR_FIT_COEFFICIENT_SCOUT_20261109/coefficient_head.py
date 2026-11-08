"""Equal-capacity frozen-basis residual heads with prediction-only support."""
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS=256
WIDTH=32
ARMS=("query","foreground","contrast")


class MaskCoefficientHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(CHANNELS,WIDTH)
        self.output=nn.Linear(WIDTH,CHANNELS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x):
        return self.output(F.gelu(self.input(x)))


def candidate_context(basis,q,raw):
    shape=raw>0
    shape_f=shape[:,None].float()
    local=F.max_pool2d(shape_f,5,stride=1,padding=2)[:,0].bool()
    interior=~F.max_pool2d(1-shape_f,5,stride=1,padding=2)[:,0].bool()
    inner=shape & ~interior
    outer=local & ~shape
    bank=basis.flatten(1).T
    in_count=inner.sum((1,2))
    out_count=outer.sum((1,2))
    fg=(inner.flatten(1).float()@bank)/in_count[:,None].clamp_min(1)
    bg=(outer.flatten(1).float()@bank)/out_count[:,None].clamp_min(1)
    normalize=lambda a:F.layer_norm(a,(CHANNELS,))
    stock=normalize(q)
    radius=.5*raw.abs().masked_fill(~local,float("nan")).flatten(1).nanmedian(1).values.nan_to_num(1.).clamp(1.,4.)
    return dict(query=stock,foreground=normalize(stock+.5*normalize(fg)),
                contrast=normalize(stock+.5*normalize(fg-bg)),
                local=local,radius=radius,active=(in_count>=4)&(out_count>=4))


def bounded_readout(basis,q,raw,context,delta):
    change=torch.einsum("chw,nc->nhw",basis,delta)
    local=context["local"]
    radius=context["radius"]
    local_energy=(change.square()*local).sum((1,2))/local.sum((1,2)).clamp_min(1)
    frame_energy=change.square().mean((1,2))
    coeff_energy=delta.square().sum(1)/q.square().sum(1).clamp_min(1e-12)
    energy=.5*(local_energy+frame_energy)/radius.square()+coeff_energy
    scale=torch.rsqrt(energy.clamp_min(1.))*context["active"]
    return raw+change*scale[:,None,None]


def mask_loss(edited,raw,query_rows,gt_rows):
    z=edited[query_rows]
    truth=gt_rows.float()
    fg=truth.sum((1,2)).clamp_min(1)
    bg=(1-truth).sum((1,2)).clamp_min(1)
    weights=.5*(truth/fg[:,None,None]+(1-truth)/bg[:,None,None])
    bce=(F.binary_cross_entropy_with_logits(z,truth,reduction="none")*weights).sum((1,2)).mean()
    prob=z.sigmoid()
    dice=(1-(2*(prob*truth).sum((1,2))+1)/(prob.sum((1,2))+truth.sum((1,2))+1)).mean()
    return bce+dice+.05*(edited-raw).square().mean()
