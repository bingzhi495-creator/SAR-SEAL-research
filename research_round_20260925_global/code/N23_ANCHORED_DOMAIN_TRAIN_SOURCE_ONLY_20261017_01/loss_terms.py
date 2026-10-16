# encoding: utf-8
"""UNBOUND anchored PR-domain loss; no I/O, backend imports or training.

All arms retain anchor_loss with coefficient one. The returned extension is
unscaled; total = anchor_loss + extension_weight * extension. A_COMMON divides
Omega-only weighted BCE by all-C weight; A_ACTIVE divides it by Omega weight;
B_C divides all-C weighted BCE by all-C weight. Outside-Omega constants are not
included in either A extension. These are fixed empirical-pool means, with no
unbiased-estimator claim. Production locks extension_weight=.25 externally.
"""

from math import isfinite
from numbers import Real


def _floating(value):
    dtype = getattr(value, "dtype", None)
    return (getattr(dtype, "kind", None) == "f" or
            getattr(dtype, "is_floating_point", False) is True)


def combine_loss(anchor_loss, new_point_losses, omega, weights, arm,
                 extension_weight=.25):
    """Combine one scalar anchor and a shared vector pool, preserving its graph.

    Inputs are NumPy-like arrays or production tensors; this module imports
    neither backend. Empty/zero-denominator extensions use losses.sum()*0.
    Losses/weights must be finite nonnegative floating vectors and omega a
    boolean vector of the same shape. Zero weights are allowed; zero total
    weight contributes zero. Validation does not detach the loss expression.
    """
    if arm not in ("A_COMMON", "A_ACTIVE", "B_C"):
        raise ValueError("arm must be A_COMMON, A_ACTIVE or B_C")
    if (isinstance(extension_weight, bool) or not isinstance(extension_weight, Real)
            or not isfinite(extension_weight) or extension_weight < 0):
        raise ValueError("extension_weight must be a finite nonnegative real scalar")
    if isinstance(anchor_loss, Real) and not isinstance(anchor_loss, bool):
        anchor_finite = isfinite(anchor_loss)
    else:
        anchor_finite = (getattr(anchor_loss, "shape", None) == () and
                         _floating(anchor_loss) and bool(abs(anchor_loss) < float("inf")))
    if not anchor_finite:
        raise ValueError("anchor_loss must be a finite scalar")
    shape = getattr(new_point_losses, "shape", None)
    if (shape is None or len(shape) != 1 or getattr(omega, "shape", None) != shape
            or getattr(weights, "shape", None) != shape):
        raise ValueError("losses, omega and weights must have the same 1D shape")
    if not _floating(new_point_losses) or not _floating(weights):
        raise ValueError("losses and weights must have floating dtype")
    dtype = getattr(omega, "dtype", None)
    if not (getattr(dtype, "kind", None) == "b" or str(dtype) == "torch.bool"):
        raise ValueError("omega must have boolean dtype")
    for value in (new_point_losses, weights):
        if not bool((abs(value) < float("inf")).all()) or not bool((value >= 0).all()):
            raise ValueError("losses and weights must be finite and nonnegative")
    selected_weights = weights if arm == "B_C" else weights * omega
    denom = selected_weights.sum() if arm == "A_ACTIVE" else weights.sum()
    if not bool(abs(denom) < float("inf")):
        raise ValueError("weight denominator overflow")
    if bool(denom > 0):
        numerator = (new_point_losses * selected_weights).sum()
        if not bool(abs(numerator) < float("inf")):
            raise ValueError("weighted loss numerator overflow")
        extension = numerator / denom
    else:
        extension = new_point_losses.sum() * 0
    return {"total": anchor_loss + extension_weight * extension,
            "extension": extension, "denom": denom}
