"""Unbound fixed-candidate contract; no SAM, tensor, image or data access."""
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class Candidate:
    query_id: int
    category_id: int
    score: float
    mask: object


@dataclass(frozen=True)
class BackendResult:
    query_ids: Tuple[int, ...]
    masks: Tuple[object, ...]


@dataclass(frozen=True)
class RefinementResult:
    candidates: Tuple[Candidate, ...]
    empty_policy: str
    is_adaptation: bool
    backend_calls: int
    configured_iterations: int


def _mask(mask, image_size):
    height, width = image_size
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError('Positive integer image dimensions are required')
    if not isinstance(mask, (list, tuple)) or len(mask) != height:
        raise ValueError('Mask height differs from declared original geometry')
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in mask):
        raise ValueError('Mask width differs from declared original geometry')
    if any(type(value) not in (bool, int) or value not in (0, 1)
           for row in mask for value in row):
        raise ValueError('Masks must contain builtin bool/int zero or one')
    return tuple(tuple(row) for row in mask)


def refine_candidates(candidates, image_size, backend, *, empty_policy='reject'):
    if empty_policy not in ('reject', 'passthrough'):
        raise ValueError('Unsupported empty-mask policy')
    original = tuple(candidates)
    if len(original) != 100:
        raise ValueError('Exactly 100 original candidates are required')
    ids = tuple(c.query_id for c in original)
    if len(set(ids)) != len(ids):
        raise ValueError('Original query identities must be unique')
    masks = tuple(_mask(c.mask, image_size) for c in original)
    active = tuple(i for i, mask in enumerate(masks)
                   if any(value for row in mask for value in row))
    if len(active) != len(original) and empty_policy == 'reject':
        raise ValueError('Empty masks are rejected by the explicit default policy')
    output_masks = list(masks)
    calls = 0
    if active:
        submitted_ids = tuple(ids[i] for i in active)
        submitted_masks = tuple(masks[i] for i in active)
        result = backend(query_ids=submitted_ids, masks=submitted_masks,
                         image_size=image_size, iters=5)
        calls = 1
        if tuple(result.query_ids) != submitted_ids:
            raise ValueError('Backend query identities or order changed')
        if len(result.masks) != len(submitted_ids):
            raise ValueError('Backend mask count differs from submitted candidates')
        validated = tuple(_mask(mask, image_size) for mask in result.masks)
        for index, mask in zip(active, validated):
            output_masks[index] = mask
    refined = tuple(replace(c, mask=m) for c, m in zip(original, output_masks))
    return RefinementResult(refined, empty_policy, empty_policy == 'passthrough', calls, 5)
