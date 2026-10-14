"""Independent SYNTHETIC_ONLY bit-mask oracle audit; stdlib, no SAR inputs.

Enumerates actual GT/native/fixed-after bit labels and physical candidate sets.
Reference maxima come from enumerating every real selected pixel subset, not
from the helper's class priority or ratio derivation. No scientific execution.
"""
from fractions import Fraction
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).with_name('N15_COUNT_BASED_ACTION_ORACLE_UNBOUND_DRAFT_20261014_01.py')
SPEC = importlib.util.spec_from_file_location('_N15_independent_synthetic_oracle', SOURCE)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
CLASSES = ('a', 'b', 'c', 'd', 'z')


def bit_classes(gt, before, after, pool):
    return dict(a=before & ~gt & ~after & pool,
                b=~before & gt & after & pool,
                c=before & gt & ~after & pool,
                d=~before & ~gt & after & pool,
                z=~(before ^ after) & pool)


def counts(gt, prediction):
    return ((gt & prediction).bit_count(),
            (prediction & ~gt).bit_count(),
            (gt & ~prediction).bit_count())


def iou(gt, prediction):
    union = (gt | prediction).bit_count()
    return Fraction((gt & prediction).bit_count(), union) if union else Fraction(1)


def selected_prediction(before, after, selected):
    return (before & ~selected) | (after & selected)


def real_subsets(pool):
    selected = pool
    while True:
        yield selected
        if not selected:
            return
        selected = (selected - 1) & pool


def maximum_ratios(gt, before, after, pool):
    best = [None] * (pool.bit_count() + 1)
    for subset in real_subsets(pool):
        number = subset.bit_count()
        score = iou(gt, selected_prediction(before, after, subset))
        best[number] = score if best[number] is None else max(best[number], score)
    assert all(value is not None for value in best)
    return best


def call(gt, before, after, pool, K, mode):
    native = counts(gt, before)
    bins = bit_classes(gt, before, after, pool)
    classes = {key: value.bit_count() for key, value in bins.items()}
    assert sum(classes.values()) == pool.bit_count()
    result = HELPER.oracle_counts(*native, classes, K, mode)
    selected = 0
    for key in CLASSES:
        number = result['selected_counts'][key]
        assert type(number) is int and 0 <= number <= classes[key]
        candidates = bins[key]
        for _ in range(number):
            pixel = candidates & -candidates
            assert pixel
            selected |= pixel
            candidates ^= pixel
    assert selected & ~pool == 0
    assert selected.bit_count() == result['selected_support']
    assert selected.bit_count() == K if mode == 'exact' else selected.bit_count() <= K
    prediction = selected_prediction(before, after, selected)
    tp, fp, fn = counts(gt, prediction)
    assert result['after_counts'] == dict(TP=tp, FP=fp, FN=fn)
    assert result['intersection'] == tp and result['union'] == tp + fp + fn
    ratio = iou(gt, prediction)
    assert result['iou_fraction'] == dict(numerator=ratio.numerator, denominator=ratio.denominator)
    assert result['iou'] == float(ratio)
    assert result['empty_union'] == (tp + fp + fn == 0)
    flips = (selected & (before ^ after)).bit_count()
    assert result['actual_flips'] == flips
    assert result['selected_noops'] == selected.bit_count() - flips
    assert result['helpful_flips'] == result['selected_counts']['a'] + result['selected_counts']['b']
    assert result['harmful_flips'] == result['selected_counts']['c'] + result['selected_counts']['d']
    assert result['physical_pool_verified_by_helper'] is False
    return ratio


def exhaustive_pixels(maximum_pixels=4):
    cases = calls = 0
    for number in range(maximum_pixels + 1):
        states = range(1 << number)
        for gt in states:
            for before in states:
                for after in states:
                    for pool in states:
                        exact = maximum_ratios(gt, before, after, pool)
                        for K, expected in enumerate(exact):
                            actual = call(gt, before, after, pool, K, 'exact')
                            assert actual == expected, ('exact', number, gt, before, after, pool, K, actual, expected)
                            expected_atmost = max(exact[:K + 1])
                            actual_atmost = call(gt, before, after, pool, K, 'atmost')
                            assert actual_atmost == expected_atmost, ('atmost', number, gt, before, after, pool, K, actual_atmost, expected_atmost)
                            calls += 2
                        cases += 1
        print('PIXEL_EXHAUSTIVE_OK', number, 'cumulative_physical_cases', cases, 'oracle_calls', calls, flush=True)
    return cases, calls


def paired_pool_invariants(maximum_pixels=3):
    pairs = 0
    for number in range(maximum_pixels + 1):
        states = range(1 << number)
        for gt in states:
            for before in states:
                for after in states:
                    native = iou(gt, before)
                    for candidate in states:
                        for omega in real_subsets(candidate):
                            K = omega.bit_count()
                            old_atmost = call(gt, before, after, omega, K, 'atmost')
                            new_atmost = call(gt, before, after, candidate, K, 'atmost')
                            new_exact = call(gt, before, after, candidate, K, 'exact')
                            old_exact = call(gt, before, after, omega, K, 'exact')
                            unrestricted_action_atmost = call(gt, before, gt, candidate, K, 'atmost')
                            baseline_apply_omega = iou(gt, selected_prediction(before, after, omega))
                            assert native <= old_atmost <= new_atmost <= unrestricted_action_atmost
                            assert old_exact == baseline_apply_omega <= new_exact <= new_atmost
                            pairs += 1
        print('PAIRED_POOL_INVARIANTS_OK', number, 'cumulative_pairs', pairs, flush=True)
    return pairs


if __name__ == '__main__':
    cases, calls = exhaustive_pixels()
    pairs = paired_pool_invariants()
    print('SYNTHETIC_ONLY_COMPLETE', 'physical_cases', cases, 'oracle_calls', calls,
          'paired_pool_cases', pairs, 'SAR_science_executed', False, flush=True)
