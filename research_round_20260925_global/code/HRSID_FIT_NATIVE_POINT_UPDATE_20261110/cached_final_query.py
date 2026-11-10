"""Cache the frozen inputs of RF-DETR's genuine final query projection."""

from contextlib import contextmanager
from dataclasses import dataclass
from operator import index

import torch
from torch import nn
from rfdetr.models.heads.segmentation import SegmentationHead, point_sample


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


@dataclass(frozen=True)
class FinalQueryCache:
    spatial: torch.Tensor
    queries: torch.Tensor
    bias: torch.Tensor


def projection_parameters(projection):
    """Return exactly the native query Linear weight and bias."""
    if type(projection) is not nn.Linear or projection.bias is None:
        raise ValueError("Expected the RF-DETR native biased query Linear")
    return projection.weight, projection.bias


def _native_dense(projection, cache):
    weight, _ = projection_parameters(projection)
    if (cache.queries.shape[-1] != projection.in_features
            or cache.spatial.shape[1] != projection.out_features
            or cache.queries.dtype != weight.dtype or cache.queries.device != weight.device):
        raise ValueError("The query projection must match the frozen carrier")
    projected = projection(cache.queries)
    return torch.einsum("bchw,bnc->bnhw", cache.spatial, projected) + cache.bias


class _FinalQueryCapture:
    def __init__(self, head):
        self.head = head
        self.spatial_hits = 0
        self.query_hits = 0
        self.spatial = self.queries = None
        self.closed = False

    def spatial_hook(self, _module, _inputs, output):
        if torch.is_grad_enabled():
            raise ValueError("Capture the frozen spatial carrier under no_grad or inference_mode")
        self.spatial_hits += 1
        if self.spatial_hits == 1:
            self.spatial = output

    def query_hook(self, _module, inputs):
        if torch.is_grad_enabled():
            raise ValueError("Capture frozen query inputs under no_grad or inference_mode")
        self.query_hits += 1
        if self.query_hits == 1:
            (self.queries,) = inputs

    def seal(self, native_masks):
        """Verify exact all-query logits, then return detached, trainable-use inputs."""
        if not self.closed or (self.spatial_hits, self.query_hits) != (1, 1):
            raise ValueError("Seal one closed final-only native segmentation forward")
        spatial, queries = self.spatial, self.queries
        self.spatial = self.queries = None
        self.closed = False
        if (spatial.ndim != 4 or queries.ndim != 3 or spatial.shape[0] != 1
                or queries.shape[0] != 1 or queries.shape[1] < 1 or min(spatial.shape[-2:]) < 1):
            raise ValueError("Expected one view with a nonempty mask canvas and query set")
        with torch.inference_mode(False), torch.no_grad():
            cache = FinalQueryCache(
                spatial.detach().clone(),
                queries.detach().clone(),
                self.head.bias.detach().clone(),
            )
            reconstructed = _native_dense(self.head.query_features_proj, cache)
            if not torch.equal(reconstructed, native_masks) or not bool(torch.isfinite(reconstructed).all()):
                raise ValueError("Cached reconstruction must exactly match all finite native mask logits")
        return cache


@contextmanager
def capture_final_query(head):
    """Observe the native projected spatial map and input to its final query Linear."""
    if type(head) is not SegmentationHead or head.training or head._export:
        raise ValueError("Expected the native RF-DETR non-export segmentation head in eval mode")
    projection_parameters(head.query_features_proj)
    capture = _FinalQueryCapture(head)
    spatial = head.spatial_features_proj.register_forward_hook(capture.spatial_hook)
    try:
        query = head.query_features_proj.register_forward_pre_hook(capture.query_hook)
        try:
            yield capture
            capture.closed = True
        finally:
            query.remove()
    finally:
        spatial.remove()


def matched_dense_logits(projection, cache, query_ids):
    """Replay the native dense contraction and select query planes in caller order."""
    ids = torch.as_tensor(query_ids, device=cache.queries.device)
    if ids.ndim != 1 or not ids.numel() or ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("Expected a nonempty vector of integer query IDs")
    ids = ids.long()
    if bool((ids < 0).any()) or bool((ids >= cache.queries.shape[1]).any()):
        raise ValueError("A matched query ID is outside the cached final-query bank")
    # Preserve the native all-query contraction shape and rounding before selection.
    return _native_dense(projection, cache)[0].index_select(0, ids)


def sample_integer_cells(mask_logits, cells_xy, image_hw):
    """Sample matched mask planes at the fixed final-image pixel centers."""
    height, width = map(index, image_hw)
    if (mask_logits.ndim != 3 or not mask_logits.is_floating_point()
            or not min(mask_logits.shape) or min(height, width) < 1):
        raise ValueError("Expected nonempty floating mask planes and a positive image canvas")
    cells = torch.as_tensor(cells_xy, device=mask_logits.device)
    if (cells.ndim not in (2, 3) or cells.shape[-1] != 2 or not cells.shape[-2]
            or cells.dtype not in _INTEGER_DTYPES):
        raise ValueError("Expected shared K-by-2 or matched M-by-K-by-2 integer cells")
    if cells.ndim == 3 and cells.shape[0] != mask_logits.shape[0]:
        raise ValueError("Per-query cells must match the number of mask planes")
    cells = cells.long()
    bounds = cells.new_tensor([width, height])
    if bool(((cells < 0) | (cells >= bounds)).any()):
        raise ValueError("A requested integer cell lies outside the final image grid")
    coordinates = ((cells.to(torch.float64) + .5) / bounds).to(mask_logits.dtype)
    if cells.ndim == 2:
        coordinates = coordinates[None].expand(mask_logits.shape[0], -1, -1)
    return point_sample(mask_logits[:, None], coordinates, align_corners=False)[:, 0]
