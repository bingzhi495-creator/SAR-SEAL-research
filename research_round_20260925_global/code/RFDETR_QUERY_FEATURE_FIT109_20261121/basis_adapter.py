"""Frozen CPU final-basis observation and ordered native endpoint sampling only."""
from contextlib import contextmanager
from dataclasses import dataclass
from operator import index

import torch
from torch.nn import functional as F

CHANNELS = 256
NATIVE_QUERIES = 164
REFERENCE_ATOL = 2e-5
REFERENCE_RTOL = 2e-5
INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


@dataclass(frozen=True)
class FrozenBasis:
    spatial: torch.Tensor
    projected_queries: torch.Tensor
    bias: torch.Tensor


def need(ok, message):
    if not ok:
        raise ValueError(message)


def spatial_tensor(value):
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float32,
         "Frozen CPU FP32 spatial tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    need(value.ndim == 4 and value.shape[0] == 1 and value.shape[1] == CHANNELS
         and min(value.shape[-2:]) > 0 and bool(torch.isfinite(value).all()),
         "One finite projected 256-channel spatial map")
    return value


def select_projected_queries(basis, query_ids):
    """Index the already projected full bank, preserving IDs, order and repeats."""
    bank=basis.projected_queries
    need(bank.device.type=="cpu" and bank.dtype==torch.float32 and
         tuple(bank.shape)==(1,NATIVE_QUERIES,CHANNELS),"Full native projected query bank")
    ids=torch.as_tensor(query_ids)
    need(ids.device.type=="cpu" and ids.ndim==1 and ids.numel()>0 and ids.dtype in INTEGER_DTYPES,
         "Nonempty original integer query IDs")
    ids=ids.to(torch.int64)
    need(not bool(((ids<0)|(ids>=NATIVE_QUERIES)).any()),"Original query ID outside bank")
    with torch.inference_mode(False),torch.no_grad():
        return bank[0].index_select(0,ids)


class _BasisObservation:
    def __init__(self,head,capture):
        self.head,self.capture=head,capture
        self.closed=False
        self.sealed=False

    def seal(self,native_logits):
        need(self.closed and not self.sealed,"Seal one closed final-only observation once")
        with torch.inference_mode(False),torch.no_grad():
            cache=self.capture.seal(native_logits)
            spatial=spatial_tensor(cache.spatial)
            projection=self.head.query_features_proj
            need(type(projection) is torch.nn.Linear and projection.bias is not None
                 and projection.out_features==CHANNELS,"Original biased 256-channel query Linear")
            queries=cache.queries
            need(queries.device.type=="cpu" and queries.dtype==torch.float32
                 and tuple(queries.shape)==(1,NATIVE_QUERIES,projection.in_features)
                 and bool(torch.isfinite(queries).all()),"Original complete pre-projection query inputs")
            projected=projection(queries)  # Project all Q164 before any ID selection.
            bias=cache.bias
            need(bias.device.type=="cpu" and bias.dtype==torch.float32 and bias.numel()==1
                 and bool(torch.isfinite(bias).all()),"Original finite shared mask bias")
            need(native_logits.device.type=="cpu" and native_logits.dtype==torch.float32
                 and tuple(native_logits.shape)==(1,NATIVE_QUERIES,*spatial.shape[-2:])
                 and bool(torch.isfinite(native_logits).all()),"Finite native all-query logits")
            basis=FrozenBasis(spatial.detach().clone(),projected.detach().clone(),bias.detach().clone())
            reconstructed=torch.einsum("bchw,bnc->bnhw",basis.spatial,basis.projected_queries)+basis.bias
            need(torch.equal(reconstructed,native_logits),"Exact full native contraction before sampling")
        self.sealed=True
        return basis


@contextmanager
def capture_frozen_basis(head,capture_final_query):
    """Inject the existing final-query context; never alter or call a model.

    The caller makes its one normal forward under no_grad/inference_mode and
    seals only after this context has exited. The helper supplies projected
    spatial output plus pre-Linear query inputs, matching its existing ABI.
    """
    need(isinstance(head,torch.nn.Module) and not head.training and not getattr(head,"_export",False),
         "Existing eval-mode non-export head")
    need(not torch.is_grad_enabled() and callable(capture_final_query),"Frozen observation and injected helper")
    with capture_final_query(head) as capture:
        observed=_BasisObservation(head,capture)
        yield observed
    observed.closed=True


def sample_sealed_endpoints(spatial, pairs, original_hw):
    """Return K x [add,drop] x 256; pairs are immutable native flat pixel indices.

    Coordinates are native integer pixel centers normalized in float64, then
    converted to FP32 for one batched bilinear border grid_sample. This has a
    documented tolerance to dense interpolate-and-index, not bitwise equality.
    """
    values = spatial_tensor(spatial)
    need(len(original_hw) == 2 and not any(isinstance(v, bool) for v in original_hw),
         "Original integer height and width")
    try:
        height, width = map(index, original_hw)
    except TypeError as error:
        raise ValueError("Original integer height and width") from error
    need(min(height,width) > 0, "Positive original canvas")
    cells = torch.as_tensor(pairs)
    need(cells.device.type == "cpu" and cells.ndim == 2 and cells.shape[1] == 2
         and cells.dtype in INTEGER_DTYPES, "Sealed K-by-2 integer add/drop flat indices")
    cells = cells.to(torch.int64)
    need(not bool(((cells < 0) | (cells >= height*width)).any()), "Native flat endpoint outside canvas")
    if not len(cells):
        with torch.inference_mode(False),torch.no_grad():
            return torch.empty((0,2,CHANNELS), dtype=torch.float32)
    flat = cells.reshape(-1)
    xy = torch.stack((flat % width, torch.div(flat,width,rounding_mode="floor")),dim=1)
    unit = ((xy.to(torch.float64)+.5)/torch.tensor([width,height],dtype=torch.float64)).to(torch.float32)
    grid = (2*unit-1)[None,:,None,:]
    with torch.inference_mode(False),torch.no_grad():
        result = F.grid_sample(values,grid,mode="bilinear",padding_mode="border",align_corners=False)
        return result[0,:,:,0].T.contiguous().reshape(len(cells),2,CHANNELS)
