"""Matched point-head losses for equal-count native pixel additions and removals."""

import torch
from torch.nn import functional as F


def _row_counts(values, rows, device, name):
    counts = torch.as_tensor(values, device=device)
    if counts.shape != (rows,) or counts.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise ValueError(f"{name} must have one integer entry per row")
    return counts.to(torch.long)


def _top_k_mask(scores, valid, k):
    # Each side's input order is native flat pixel order. A stable sort gives
    # the deployed selection on ties, including a tie spanning the cutoff.
    order = scores.masked_fill(~valid, -torch.inf).argsort(
        dim=1, descending=True, stable=True
    )
    ranks = torch.arange(scores.size(1), device=scores.device)[None, :, None]
    take = (ranks < k[:, None, None]).expand_as(scores)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, order, take)


def rank_objectives(logits, occupancy, lengths, k):
    """Return scalar total losses keyed by 'bce', 'pairwise' and 'spo'.

    Inputs are [rows, max_points, 2], with addition then removal columns.
    The first lengths[r] points in *each* column must be in native flat-index
    order. Each column may be independently permuted into that order. Finite
    logits and continuous cell occupancies in [0, 1] are required on support.
    Padding is ignored. k[r] is a caller-supplied budget in [0,min(8,length)].

    Each row has equal weight. BCE averages both columns on its valid support.
    Pairwise averages over strictly ordered pairs within each side, then over
    the two sides. A side without preferences contributes zero. The SPO+
    term uses normalized, signed *probability* utilities at the row's k.
    A zero budget gives zero SPO+ and leaves available point/pair supervision
    intact. An empty row contributes zero to all three row means.
    """
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 3 or logits.size(-1) != 2 or logits.size(0) == 0):
        raise ValueError("logits must be floating [rows, max_points, 2] with rows > 0")
    if (not isinstance(occupancy, torch.Tensor) or not occupancy.is_floating_point()
            or occupancy.shape != logits.shape or occupancy.device != logits.device):
        raise ValueError("occupancy must be floating with the logits' shape and device")
    rows, width, _ = logits.shape
    lengths = _row_counts(lengths, rows, logits.device, "lengths")
    k = _row_counts(k, rows, logits.device, "k")
    if ((lengths < 0) | (lengths > width) | (k < 0) | (k > 8) | (k > lengths)).any():
        raise ValueError("require 0 <= lengths <= width and 0 <= k <= min(8, lengths)")

    positions = torch.arange(width, device=logits.device)
    valid = (positions[None, :] < lengths[:, None]).unsqueeze(-1)
    z = logits.masked_fill(~valid, 0)
    q = occupancy.detach().masked_fill(~valid, 0)
    if not torch.isfinite(z).all() or not torch.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("require finite logits and finite occupancies in [0, 1] on support")

    point_loss = F.binary_cross_entropy_with_logits(z, q, reduction="none")
    bce = (point_loss.masked_fill(~valid, 0).sum((1, 2)) / (2 * lengths).clamp_min(1)).mean()

    # Utilities differ from q and -q by a side-specific constant. Using -q
    # for removals preserves fractional label differences near zero exactly.
    sign = z.new_tensor([1.0, -1.0])
    true_utility = q * sign
    signed_logits = z * sign
    preferred = (true_utility[:, :, None, :] > true_utility[:, None, :, :])
    preferred &= valid[:, :, None, :] & valid[:, None, :, :]
    margin = signed_logits[:, :, None, :] - signed_logits[:, None, :, :]
    pair_loss = F.softplus(-margin).masked_fill(~preferred, 0).sum((1, 2))
    pair_loss = (pair_loss / preferred.sum((1, 2)).clamp_min(1)).mean()

    predicted_utility = z.sigmoid() * sign
    augmented = 2 * predicted_utility - true_utility
    truth_choice = _top_k_mask(true_utility, valid, k)
    augmented_choice = _top_k_mask(augmented, valid, k)
    change = augmented_choice.to(augmented.dtype) - truth_choice.to(augmented.dtype)
    spo_loss = ((augmented * change).sum((1, 2)) / (2 * k).clamp_min(1)).mean()

    return dict(bce=bce, pairwise=bce + 0.25 * pair_loss, spo=bce + 0.25 * spo_loss)
