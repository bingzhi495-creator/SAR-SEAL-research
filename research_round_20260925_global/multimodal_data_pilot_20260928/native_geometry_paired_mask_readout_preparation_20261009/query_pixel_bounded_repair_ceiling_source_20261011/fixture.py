"""Synthetic stdlib fixture; authored without executing its numeric checks.

Run only when the owner chooses to execute this fixture. It does not load data,
models, scientific libraries, or experiment artifacts, and writes no files.
The oracle below independently enumerates the closed-interval binary states.
"""

import itertools
import json
import math


def _allowed_states(logit, kappa):
    # Independent endpoint construction, rather than the tool's GT categories.
    if logit + kappa <= 0.0:
        return (0,)
    if logit - kappa > 0.0:
        return (1,)
    return (0, 1)


def _iou(mask, truth, empty_union_iou):
    intersection = sum(int(pred == 1 and gt == 1)
                       for pred, gt in zip(mask, truth))
    union = sum(int(pred == 1 or gt == 1)
                for pred, gt in zip(mask, truth))
    return intersection / union if union else empty_union_iou


def _check_case(tool, logits, truth, kappa, empty_union_iou):
    result = tool(logits, truth, kappa, empty_union_iou=empty_union_iou)
    allowed = [_allowed_states(logit, kappa) for logit in logits]
    all_scores = [_iou(mask, truth, empty_union_iou)
                  for mask in itertools.product(*allowed)]
    best = max(all_scores)
    upper = result['conservative_IoU_upper_bound']
    assert math.isclose(upper, best, rel_tol=0.0, abs_tol=1e-15)
    assert all(score <= upper + 1e-15 for score in all_scores)
    assert result['pixel_count'] == len(logits)
    assert result['GT_area'] == sum(truth)
    assert result['forced_FN'] == sum(gt == 1 and states == (0,)
                                      for gt, states in zip(truth, allowed))
    assert result['forced_FP'] == sum(gt == 0 and states == (1,)
                                      for gt, states in zip(truth, allowed))
    editable = [index for index, states in enumerate(allowed)
                if states == (0, 1)]
    assert result['editable_counts'] == dict(
        total=len(editable), GT=sum(truth[index] for index in editable),
        BG=sum(1 - truth[index] for index in editable))
    for name, endpoint in (('L_eq_negative_kappa', -kappa),
                           ('L_eq_positive_kappa', kappa)):
        indices = [index for index, logit in enumerate(logits)
                   if logit == endpoint]
        assert result['endpoints'][name] == dict(
            total=len(indices), GT=sum(truth[index] for index in indices),
            BG=sum(1 - truth[index] for index in indices))
    native = tuple(int(logit > 0.0) for logit in logits)
    tp = sum(pred == 1 and gt == 1 for pred, gt in zip(native, truth))
    fp = sum(pred == 1 and gt == 0 for pred, gt in zip(native, truth))
    fn = sum(pred == 0 and gt == 1 for pred, gt in zip(native, truth))
    assert result['native'] == dict(
        TP=tp, FP=fp, FN=fn, prediction_area=sum(native),
        IoU=_iou(native, truth, empty_union_iou))
    assert result['native']['IoU'] <= upper + 1e-15
    assert result['bound_numerator'] == sum(truth) - result['forced_FN']
    assert result['bound_denominator'] == sum(truth) + result['forced_FP']
    assert result['empty_union_iou'] == empty_union_iou
    assert result['empty_union_applied'] == (result['bound_denominator'] == 0)
    assert result['scope'] == dict(
        fixed_query_GT_oracle=True, closed_interval_model=True,
        AP_bound=False, kappa_selected=False,
        general_FP32_pipeline_certified=False)
    return len(all_scores)


def _rejects(tool, args, kwargs=None):
    try:
        tool(*args, **(kwargs or {}))
    except (TypeError, ValueError):
        return
    raise AssertionError('Invalid input was accepted: {!r}'.format(args))


def run_fixture():
    from bounded_ceiling import bounded_ceiling

    cases = 0
    masks = 0
    # Binary-exact scales avoid conflating endpoint arithmetic with the oracle.
    for kappa in (0.5, 1.0, 2.0):
        grid = tuple(kappa * value for value in (-2, -1, -0.5, 0, 0.5, 1, 2))
        for length in range(4):
            for logits in itertools.product(grid, repeat=length):
                for truth in itertools.product((0, 1), repeat=length):
                    for empty_union_iou in (0.0, 1.0):
                        masks += _check_case(bounded_ceiling, logits, truth,
                                             kappa, empty_union_iou)
                        cases += 1

    # These literal values straddle +/-1 at exactly representable FP32 values.
    # This exercises the interval classifier, not a floating-point GPU pipeline.
    logits = (-1.0000001192092896, -1.0, -0.9999999403953552,
              0.9999999403953552, 1.0, 1.0000001192092896)
    for truth in itertools.product((0, 1), repeat=len(logits)):
        masks += _check_case(bounded_ceiling, logits, truth, 1.0, 1.0)
        cases += 1

    # Explicit asymmetry: +kappa can reach zero, -kappa cannot become >zero.
    endpoint = bounded_ceiling([-1.0, 1.0], [1, 0], 1.0)
    assert endpoint['forced_FN'] == 1 and endpoint['forced_FP'] == 0
    assert endpoint['editable_counts'] == dict(total=1, GT=0, BG=1)
    assert bounded_ceiling([], [], 1.0)['conservative_IoU_upper_bound'] == 1.0
    assert bounded_ceiling([2.0], [0], 1.0)['conservative_IoU_upper_bound'] == 0.0
    assert bounded_ceiling([-2.0], [0], 1.0)['conservative_IoU_upper_bound'] == 1.0
    assert bounded_ceiling([-2.0], [0], 1.0,
                           empty_union_iou=0.0)['conservative_IoU_upper_bound'] == 0.0
    assert bounded_ceiling([0], [True], 1)['GT_area'] == 1

    invalid = [([], [0], 1), ([0], [], 1), ([0], [2], 1),
               ([0], [-1], 1), ([0], [0.5], 1), ([0], ['1'], 1),
               ([True], [0], 1), (['0'], [0], 1),
               ([math.nan], [0], 1), ([math.inf], [0], 1),
               ([-math.inf], [0], 1), ([0], [0], 0),
               ([0], [0], -1), ([0], [0], math.nan),
               ([0], [0], math.inf), ([0], [0], True),
               ([0], [0], '1'), ('0', [0], 1), ([0], '0', 1),
               ((value for value in [0]), [0], 1)]
    for args in invalid:
        _rejects(bounded_ceiling, args)
    for invalid_empty in (-1, 0.5, 2, math.nan, math.inf, '1', True):
        _rejects(bounded_ceiling, ([], [], 1),
                 dict(empty_union_iou=invalid_empty))
    return dict(status='BOUNDED_CEILING_SYNTHETIC_STDLIB_PASS',
                synthetic_only=True, enumerated_cases=cases,
                enumerated_allowed_masks=masks, invalid_cases=len(invalid) + 7,
                actual_data_loaded=False, scientific_libraries_imported=False,
                AP_bound_claimed=False, general_FP32_certification_claimed=False)


if __name__ == '__main__':
    print(json.dumps(run_fixture(), sort_keys=True))
