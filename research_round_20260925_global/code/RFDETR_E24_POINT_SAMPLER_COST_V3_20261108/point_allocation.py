"""Cached P66 uniform and area-weighted stratified BCE point allocation."""
import torch


class PreparedStrata:
    """Microbatch-local sparse boundary members for the exact target tensor."""

    __slots__ = ("_masks", "_version", "_members", "_gaps")

    def __init__(self, masks, members, gaps):
        self._masks = masks
        self._version = _mask_version(masks)
        self._members = members
        self._gaps = gaps


def _check_masks(masks):
    if (not isinstance(masks, torch.Tensor) or masks.ndim != 3 or masks.dtype != torch.bool
            or min(masks.shape[1:]) <= 0):
        raise ValueError("Expected Boolean target masks [T,H,W] with positive spatial axes")


def _mask_version(masks):
    return None if torch.is_inference(masks) else masks._version


def _boundary_quota(points, band_size, total):
    if band_size in (0, total):
        return points if band_size else 0
    proportional = (2 * points * band_size + total) // (2 * total)
    return min(points - 1, max((points + 19) // 20, proportional))


def _select_complement(gaps, ranks):
    # A boundary b_i has exactly b_i-i complement positions before it.
    # Include all equal gaps from consecutive boundary positions.
    return ranks + torch.searchsorted(gaps, ranks, right=True)


def prepare_strata(masks):
    """Cache one zero-padded radius-two boundary per immutable transformed GT."""
    _check_masks(masks)
    count, height, width = masks.shape
    if count == 0:
        return PreparedStrata(masks, (), ())
    # In the 5x5 window, outside-image zeros are background. Intersecting both
    # classes is equivalent to zero-padded dilation minus zero-padded erosion.
    pooled = torch.nn.functional.avg_pool2d(
        masks[:, None].to(torch.float32), 5, stride=1, padding=2, count_include_pad=True,
    )
    bands = (pooled > 0) & (pooled < 1)
    del pooled
    total = height * width
    occupied = bands.flatten().nonzero(as_tuple=True)[0]
    sizes = torch.bincount(occupied // total, minlength=count).tolist()
    members = occupied.remainder(total).split(sizes)
    gaps = tuple(row - torch.arange(row.numel(), device=masks.device) for row in members)
    return PreparedStrata(masks, members, gaps)


def sample_target_points_prepared(masks, target_indices, K, *, arm, generator, strata):
    """Return normalized [M,K,2] x/y centers, [M,K] weights and flat indices.

    Target indices select from the unique transformed [T,H,W] masks in matched
    order. D draws repeated target rows independently in batches grouped by
    first occurrence, then restores match order. Reuse one D cache throughout
    the same immutable microbatch. B takes strata=None.
    Multiply unreduced BCE by weights before the within-instance mean.
    """
    _check_masks(masks)
    if type(K) is not int or K < 2 or arm not in ("B", "D"):
        raise ValueError("Expected arm B or D with a common integer K of at least two")
    device = masks.device
    if not isinstance(generator, torch.Generator) or generator.device != device:
        raise ValueError("Expected an explicit generator on the target mask device")
    if (not isinstance(target_indices, torch.Tensor) or target_indices.ndim != 1
            or target_indices.dtype != torch.int64
            or (target_indices.device.type != "cpu" and target_indices.device != device)):
        raise ValueError("Expected 1-D int64 match indices on CPU or the target mask device")
    targets = target_indices.tolist()
    count, height, width = masks.shape
    if any(target < 0 or target >= count for target in targets):
        raise ValueError("Match index outside the transformed target population")
    if arm == "D":
        if (not isinstance(strata, PreparedStrata) or strata._masks is not masks
                or strata._version != _mask_version(masks)):
            raise ValueError("Expected fresh D strata for the exact immutable target tensor")
    elif strata is not None:
        raise ValueError("Uniform arm B does not use D strata")

    total = height * width
    shape = (len(targets), K)
    weights = torch.ones(shape, device=device, dtype=torch.float32)
    if arm == "B" or not targets:
        indices = torch.randint(total, shape, device=device, generator=generator)
    else:
        indices = torch.empty(shape, device=device, dtype=torch.int64)
        groups = {}
        for row, target in enumerate(targets):
            groups.setdefault(target, []).append(row)
        restore_order = [0] * len(targets)
        start = 0
        for target, rows in groups.items():
            size = len(rows)
            span = slice(start, start + size)
            for packed, original in enumerate(rows, start=start):
                restore_order[original] = packed
            start += size

            members, gaps = strata._members[target], strata._gaps[target]
            band_size = members.numel()
            if band_size in (0, total):
                indices[span] = torch.randint(total, (size, K), device=device, generator=generator)
                continue
            k_band = _boundary_quota(K, band_size, total)
            k_other = K - k_band
            indices[span, :k_band] = members[torch.randint(
                band_size, (size, k_band), device=device, generator=generator)]
            other_ranks = torch.randint(total - band_size, (size, k_other), device=device, generator=generator)
            indices[span, k_band:] = _select_complement(gaps, other_ranks)
            weights[span, :k_band] = K * (band_size / total) / k_band
            weights[span, k_band:] = K * ((total - band_size) / total) / k_other

        if restore_order != list(range(len(targets))):
            order = torch.tensor(restore_order, device=device, dtype=torch.int64)
            indices = indices.index_select(0, order)
            weights = weights.index_select(0, order)

    x = (indices.remainder(width).to(torch.float32) + 0.5) / width
    y = ((indices // width).to(torch.float32) + 0.5) / height
    return torch.stack((x, y), dim=-1), weights, indices
