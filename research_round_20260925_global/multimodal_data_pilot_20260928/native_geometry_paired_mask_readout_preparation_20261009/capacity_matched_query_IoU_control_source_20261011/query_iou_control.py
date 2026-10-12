"""Mask-Scoring-style capacity-matched scalar IoU control, not a reproduction.

Six parameter tensors / 9441 scalars: 260->32->32->1, with biases and SiLU.
Only scalar parameter count matches the pixel candidate: tensor count (6 vs 8)
and computation differ; this query scalar head is not a compute-matched control.
Native queries and descriptors are frozen inputs. All 100 queries are retained;
no masks, class probabilities, query identities, filtering or NMS are changed.
Targets are supplied by the caller: each native binary mask's maximum geometric
IoU over the image's raw GT instances (zero when there is no GT). Duplicates may
have high targets; this is not Hungarian-unmatched=0 supervision. This module
does not compute targets, access data/files, promise calibration, or claim novelty.
"""
import torch
from torch import nn
from torch.nn import functional as F

DESCRIPTOR_ORDER = ('native_class_probability', 'binary_area_fraction',
                    'official_mask_quality', 'zero_learning_posterior_budget')


def _floating(name, value, ndim, reference=None, bounded=False):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.ndim != ndim:
        raise ValueError(name + ' must be a dense tensor of the specified rank')
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(name + ' must be FP32 or FP64')
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(name + ' must share the reference dtype/device')
    if not bool(torch.isfinite(value).all()) or (bounded and bool(((value < 0) | (value > 1)).any())):
        raise ValueError(name + ' must be finite' + (' and in [0,1]' if bounded else ''))


def assemble_native_descriptors(class_probability, binary_masks, official_quality, zero_budget):
    """Only native inputs: three [B,100] scalars and bool[B,100,H,W] masks.

    Compute area fraction and pack the four descriptors; preserve class probability
    for an empty mask, whose other three descriptors must be zero. No GT access.
    """
    _floating('class_probability', class_probability, 2, bounded=True)
    if class_probability.shape[0] <= 0 or class_probability.shape[1] != 100:
        raise ValueError('class_probability must be [B,100] with B>0')
    for name, value in (('official_quality', official_quality), ('zero_budget', zero_budget)):
        _floating(name, value, 2, class_probability, bounded=True)
        if value.shape != class_probability.shape:
            raise ValueError(name + ' must have shape [B,100]')
    if not isinstance(binary_masks, torch.Tensor) or binary_masks.layout != torch.strided or \
            binary_masks.dtype != torch.bool or binary_masks.device != class_probability.device or \
            binary_masks.ndim != 4 or tuple(binary_masks.shape[:2]) != tuple(class_probability.shape) or \
            min(binary_masks.shape[-2:]) <= 0:
        raise ValueError('binary_masks must be bool[B,100,H,W] with positive raster dimensions')
    area = binary_masks.to(class_probability.dtype).mean(dim=(-2, -1))
    empty = area == 0
    if bool(((official_quality[empty] != 0) | (zero_budget[empty] != 0)).any()):
        raise ValueError('empty native masks require zero official quality and zero budget')
    return torch.stack((class_probability, area, official_quality, zero_budget), dim=-1).detach()


def all_query_mse(quality, targets):
    """All 100 queries mean -> images mean; global accumulation scaling is external.

    Targets are geometry quality, not object presence or assignment labels. No GT
    processing or optimizer step occurs here; the supervision tensor is detached.
    """
    _floating('quality', quality, 2, bounded=True)
    _floating('targets', targets, 2, quality, bounded=True)
    if quality.shape[0] <= 0 or quality.shape[1] != 100 or targets.shape != quality.shape:
        raise ValueError('quality and targets must have identical shape [B,100], B>0')
    return ((quality - targets.detach()) ** 2).mean(dim=1).mean()


class CapacityMatchedQueryIoUReadout(nn.Module):
    """GT-free inference scalar head; freshly initialized on CPU in FP32.

    A zero final weight/bias starts every query at quality 0.5. That constant is
    neither a measured IoU nor a guarantee. External .double()/.to() is supported.
    """
    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError('seed must be a non-bool integer in [0,2**64-1]')
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            torch.random.set_rng_state(generator.get_state())
            super().__init__()
            self.hidden1 = nn.Linear(260, 32, bias=True, device='cpu', dtype=torch.float32)
            self.hidden2 = nn.Linear(32, 32, bias=True, device='cpu', dtype=torch.float32)
            self.final = nn.Linear(32, 1, bias=True, device='cpu', dtype=torch.float32)
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)
        self.seed = seed

    def forward(self, native_queries, descriptors):
        """e0[B,100,256], descriptors[B,100,4] -> quality[B,100], in query order."""
        reference = self.hidden1.weight
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError('head parameters must be FP32 or FP64')
        if any(p.dtype != reference.dtype or p.device != reference.device for p in self.parameters()):
            raise ValueError('all head parameters must share dtype/device')
        _floating('native_queries', native_queries, 3, reference)
        _floating('descriptors', descriptors, 3, reference, bounded=True)
        batch = native_queries.shape[0]
        if batch <= 0 or tuple(native_queries.shape) != (batch, 100, 256) or \
                tuple(descriptors.shape) != (batch, 100, 4):
            raise ValueError('inputs must be [B,100,256] and [B,100,4], B>0')
        inputs = torch.cat((native_queries.detach(), descriptors.detach()), dim=-1)
        hidden = F.silu(self.hidden1(inputs))
        logits = self.final(F.silu(self.hidden2(hidden))).squeeze(-1)
        _floating('quality logits', logits, 2, reference)
        return logits.sigmoid()
