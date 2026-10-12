"""Feature-pooled scalar IoU strong control; not a novel or full MSRCNN model.

Eight parameter tensors / 9477 scalars, 36 (0.3813%) more than the 9441 pixel
head. Inputs, output, tensor count and computation are not all matched controls.
F/e0 project 256->16 each; original-raster [pixel16,query16,L,p,B,contrast] passes
through 36->32 SiLU. Native B inner/outer mean32 concatenate with the four old
scalar descriptors, then 68->1 sigmoid. Empty support gives a zero mean vector.
Caller trains all100 max-native-GT-IoU MSE (no-GT/empty-union target zero), and
forces native-empty predicted quality/score zero at inference, retaining all100.

The optional renderer acts on projected F16, preserves batch/dtype/device and
its live gradient. For the old batch-dropping renderer, caller wraps unsqueeze.
F/e0/native L/descriptors detach internally; learned projections remain live.
No GT enters forward; no masks/classes/query identities are modified.

Query chunks bound forward hidden workspaces, not the sum of saved backward
activations. Optional Torch2.0.1 nonreentrant checkpoint recomputes block hidden
activations in backward. Full native L, projected/rendered F, common hidden terms
and their graphs still occupy memory; no 4GiB, speed or trainability guarantee.
At 100 queries on an 800x800 original frame, hidden32 has 2.048 billion values.
Checkpoint saves storage, not the main arithmetic; backward also recomputes it.
Direct concatenation vs factorization has toleranced FP32/FP64 agreement, not
bitwise equivalence. No data/model/optimizer/target generation or file access.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DESCRIPTOR_ORDER = ('native_class_probability', 'binary_area_fraction',
                    'official_mask_quality', 'zero_learning_posterior_budget')


def _tensor(name, value, ndim, reference=None, bounded=False):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.ndim != ndim:
        raise TypeError(name + ' must be a dense strided tensor of the specified rank')
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(name + ' must be FP32 or FP64')
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(name + ' must share the reference dtype/device')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite')
    if bounded and bool(((value < 0) | (value > 1)).any()):
        raise ValueError(name + ' must be in [0,1]')


def all_query_mse(quality, targets):
    """All100 query mean -> image mean; caller performs global accumulation/2.

Targets are caller's max IoU over all raw GT, including duplicate queries, not
Hungarian-unmatched zero labels. No GT gives100 zeros; empty union is target0.
"""
    _tensor('quality', quality, 2, bounded=True)
    _tensor('targets', targets, 2, quality, bounded=True)
    if quality.shape != targets.shape or quality.shape[0] <= 0 or quality.shape[1] != 100:
        raise ValueError('quality/targets must have matching positive shape [B,100]')
    return ((quality - targets.detach()) ** 2).mean(dim=1).mean()


class FeaturePooledQueryIoUReadout(nn.Module):
    """CPU FP32/private RNG constructor; externally call .double() or .to(...).

forward(F[B,256,Hf,Wf],e0[B,100,256],L[B,100,H,W],stats[B,100,4])
returns quality[B,100]. Stats use DESCRIPTOR_ORDER; they are caller's native,
GT-free measurements, whose semantic correctness this module does not prove.
"""
    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError('seed must be a non-bool integer in [0,2**64-1]')
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            torch.random.set_rng_state(generator.get_state())
            super().__init__()
            self.feature_projection = nn.Conv2d(256, 16, 1, bias=True, device='cpu', dtype=torch.float32)
            self.query_projection = nn.Linear(256, 16, bias=True, device='cpu', dtype=torch.float32)
            self.hidden = nn.Linear(36, 32, bias=True, device='cpu', dtype=torch.float32)
            self.final = nn.Linear(68, 1, bias=True, device='cpu', dtype=torch.float32)
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)
        self.seed = seed

    def _reference(self):
        reference = self.hidden.weight
        for name, parameter in self.named_parameters():
            _tensor('parameter ' + name, parameter, parameter.ndim, reference)
        return reference

    def _pool_block(self, pixel_hidden, query_hidden, native):
        batch, queries, height, width = native.shape
        probability = native.sigmoid()
        binary = (native > 0).to(native.dtype)
        neighborhood = F.avg_pool2d(probability.reshape(batch*queries, 1, height, width),
                                    3, stride=1, padding=1, count_include_pad=False)
        contrast = probability - neighborhood.reshape(batch, queries, height, width)
        hidden = pixel_hidden[:, None] + query_hidden[:, :, :, None, None]
        for index, scalar in enumerate((native, probability, binary, contrast)):
            hidden = hidden + scalar[:, :, None] * self.hidden.weight[:, 32+index].view(1, 1, 32, 1, 1)
        hidden = F.silu(hidden)
        foreground = (hidden * binary[:, :, None]).sum((-2, -1))
        foreground = foreground / binary.sum((-2, -1)).clamp_min(1)[:, :, None]
        background = (hidden * (1-binary)[:, :, None]).sum((-2, -1))
        background = background / (1-binary).sum((-2, -1)).clamp_min(1)[:, :, None]
        return torch.cat((foreground, background), dim=-1)

    def forward(self, native_features, native_coefficients, native_original_logits, descriptors,
                *, renderer=None, query_chunk_size=4, activation_checkpoint=False):
        reference = self._reference()
        for name, value, ndim in (('native_features', native_features, 4),
                                  ('native_coefficients', native_coefficients, 3),
                                  ('native_original_logits', native_original_logits, 4)):
            _tensor(name, value, ndim, reference)
        _tensor('descriptors', descriptors, 3, reference, bounded=True)
        if type(query_chunk_size) is not int or query_chunk_size <= 0:
            raise ValueError('query_chunk_size must be a positive non-bool integer')
        if type(activation_checkpoint) is not bool:
            raise TypeError('activation_checkpoint must be bool')
        batch = native_features.shape[0]
        if native_features.shape[1] != 256 or any(d <= 0 for d in native_features.shape):
            raise ValueError('native_features must have positive shape [B,256,Hf,Wf]')
        if native_coefficients.shape != (batch, 100, 256) or descriptors.shape != (batch, 100, 4):
            raise ValueError('native_coefficients/stats must match [B,100,256]/[B,100,4]')
        if native_original_logits.shape[:2] != (batch, 100) or any(d <= 0 for d in native_original_logits.shape):
            raise ValueError('native_original_logits must have positive shape [B,100,H,W]')
        native = native_original_logits.detach()
        stats = descriptors.detach()
        pixel = self.feature_projection(native_features.detach())
        query = self.query_projection(native_coefficients.detach())
        if renderer is not None:
            if not callable(renderer):
                raise TypeError('renderer must be callable or None')
            pixel = renderer(pixel)
        _tensor('rendered pixel projection', pixel, 4, reference)
        if pixel.shape != (batch, 16, *native.shape[-2:]):
            raise ValueError('renderer must return matching original-raster [B,16,H,W]')
        if torch.is_grad_enabled() and self.feature_projection.weight.requires_grad and not pixel.requires_grad:
            raise ValueError('renderer detached the learned feature projection')
        _tensor('query projection', query, 3, reference)
        pixel_hidden = F.linear(pixel.permute(0, 2, 3, 1), self.hidden.weight[:, :16],
                                self.hidden.bias).permute(0, 3, 1, 2)
        query_hidden = F.linear(query, self.hidden.weight[:, 16:32])
        pools = []
        for start in range(0, 100, query_chunk_size):
            end = min(start + query_chunk_size, 100)
            arguments = (pixel_hidden, query_hidden[:, start:end], native[:, start:end])
            if activation_checkpoint and torch.is_grad_enabled():
                block = checkpoint(self._pool_block, *arguments,
                                   use_reentrant=False, preserve_rng_state=True)
            else:
                block = self._pool_block(*arguments)
            pools.append(block)
        pooled = torch.cat(pools, dim=1)
        _tensor('pooled context', pooled, 3, reference)
        quality_logits = self.final(torch.cat((pooled, stats), dim=-1)).squeeze(-1)
        _tensor('quality logits', quality_logits, 2, reference)
        quality = quality_logits.sigmoid()
        _tensor('quality', quality, 2, reference, bounded=True)
        return quality
