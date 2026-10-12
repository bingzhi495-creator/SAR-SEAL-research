"""Fixed-query GT-oracle IoU envelope for a bounded additive logit repair.

Pure Python standard library; no data/model/artifact I/O or scientific imports.
The mathematical model allows an independent delta in [-kappa, +kappa] at every
pixel and thresholds the corrected logit strictly above zero. It relaxes a
shared learned head, and also relaxes finite-real kappa*tanh(u), whose interval
is open. It is not a claim about optimization, deployment, AP, or kappa choice.

For GT area t, forced_FN = count(GT and L <= -kappa) and forced_FP =
count(BG and L > kappa). Intersection <= t-forced_FN and union >= t+forced_FP,
so (t-forced_FN)/(t+forced_FP) is an upper bound when the denominator is nonzero.
Independent per-pixel repairs attain it in this relaxed closed-interval model;
the actual shared coefficient/readout family need not attain it.

The positive endpoint is deliberately NOT a forced FP: L=kappa can become zero
with delta=-kappa, and zero is background. L=-kappa cannot become positive.
Finite-real tanh never reaches +/-1, while FP32 tanh can saturate there. These
facts motivate the conservative closed interval, but do not certify an arbitrary
FP32 pipeline: kappa representation, multiplication, addition/rounding and the
origin of supplied logits need separate analysis. Inputs are converted to finite
Python floats, treated as exact real constants for interval classification.

Default both-empty IoU=1 matches a GT-best geometry diagnostic convention.
It is explicitly configurable to 0; neither convention is a COCO AP definition.
"""

from collections.abc import Sequence
import math
from numbers import Real


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError('{} must be a real number, not bool/string'.format(name))
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError('{} is not representable as a finite float'.format(name)) from error
    if not math.isfinite(converted):
        raise ValueError('{} must be finite'.format(name))
    return converted


def _sequence(value, name):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError('{} must be a non-string sequence'.format(name))
    return value


def bounded_ceiling(native_logits, ground_truth, kappa, *, empty_union_iou=1.0):
    """Return counts and a conservative fixed-query IoU upper bound.

    native_logits and ground_truth must be same-length sequences; each logit is
    finite, and each GT value is real 0/1 (bool is accepted for GT). kappa must
    be finite and strictly positive. An empty pixel sequence is permitted and
    follows the explicitly selected both-empty convention. No query selection,
    matching, AP, data loading, or kappa search is performed.
    """
    kappa = _finite_float(kappa, 'kappa')
    if kappa <= 0.0:
        raise ValueError('kappa must be strictly positive')
    empty_union_iou = _finite_float(empty_union_iou, 'empty_union_iou')
    if empty_union_iou not in (0.0, 1.0):
        raise ValueError('empty_union_iou must explicitly be 0 or 1')
    native_logits = _sequence(native_logits, 'native_logits')
    ground_truth = _sequence(ground_truth, 'ground_truth')
    if len(native_logits) != len(ground_truth):
        raise ValueError('native_logits and ground_truth must have equal lengths')
    logits = [_finite_float(value, 'native_logits[{}]'.format(index))
              for index, value in enumerate(native_logits)]
    truth = []
    for index, value in enumerate(ground_truth):
        if not isinstance(value, Real) or value not in (0, 1):
            raise ValueError('ground_truth[{}] must be exactly 0 or 1'.format(index))
        truth.append(int(value))

    gt_area = sum(truth)
    forced_fn = sum(gt == 1 and logit <= -kappa
                    for logit, gt in zip(logits, truth))
    forced_fp = sum(gt == 0 and logit > kappa
                    for logit, gt in zip(logits, truth))
    editable = [index for index, logit in enumerate(logits)
                if -kappa < logit <= kappa]
    endpoints = {}
    for name, endpoint in (('L_eq_negative_kappa', -kappa),
                           ('L_eq_positive_kappa', kappa)):
        indices = [index for index, logit in enumerate(logits)
                   if logit == endpoint]
        endpoints[name] = dict(
            total=len(indices), GT=sum(truth[index] for index in indices),
            BG=sum(1 - truth[index] for index in indices))

    native = [int(logit > 0.0) for logit in logits]
    tp = sum(pred == 1 and gt == 1 for pred, gt in zip(native, truth))
    fp = sum(pred == 1 and gt == 0 for pred, gt in zip(native, truth))
    fn = sum(pred == 0 and gt == 1 for pred, gt in zip(native, truth))
    native_union = tp + fp + fn
    numerator = gt_area - forced_fn
    denominator = gt_area + forced_fp
    upper = numerator / denominator if denominator else empty_union_iou
    return dict(
        model='fixed_query_closed_interval_strict_positive', kappa=kappa,
        pixel_count=len(logits), GT_area=gt_area,
        forced_FN=forced_fn, forced_FP=forced_fp,
        forced_negative_count=sum(logit <= -kappa for logit in logits),
        forced_positive_count=sum(logit > kappa for logit in logits),
        editable_counts=dict(
            total=len(editable), GT=sum(truth[index] for index in editable),
            BG=sum(1 - truth[index] for index in editable)),
        endpoints=endpoints,
        native=dict(TP=tp, FP=fp, FN=fn, prediction_area=sum(native),
                    IoU=tp / native_union if native_union else empty_union_iou),
        bound_numerator=numerator, bound_denominator=denominator,
        conservative_IoU_upper_bound=upper,
        empty_union_iou=empty_union_iou, empty_union_applied=(denominator == 0),
        scope=dict(fixed_query_GT_oracle=True, closed_interval_model=True,
                   AP_bound=False, kappa_selected=False,
                   general_FP32_pipeline_certified=False))
