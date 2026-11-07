"""Exact fixed-query pixel-edit oracle for two nonempty GT masks."""
BUDGETS = (0, 1, 2, 4, 8, 16, 32)
THRESHOLD = (9, 10)


def _minimum_edits(deficit, fn_capacity, fp_capacity):
    numerator, denominator = THRESHOLD
    additions = min(fn_capacity, (deficit + denominator - 1) // denominator)
    remaining = max(0, deficit - denominator * additions)
    removals = (remaining + numerator - 1) // numerator
    return additions + removals if removals <= fp_capacity else None


def signed_edit_oracle(integer_iou, float_iou, fn_capacity, fp_capacity):
    """Return exact per-pair 9/10 edit distances and capped best IoUs.

    The IoU inputs are (intersection, union) tuples for one unchanged prediction.
    Capacities count eligible common foreground additions and background removals.
    IoU outputs are unreduced integer pairs and unreachable distances are None.
    """
    for pair in (integer_iou, float_iou):
        if (type(pair) is not tuple or len(pair) != 2 or
                any(type(value) is not int for value in pair) or not 0 <= pair[0] <= pair[1] or pair[1] == 0):
            raise ValueError('Expected a valid integer (intersection, union) tuple')
    if any(type(capacity) is not int or capacity < 0 for capacity in (fn_capacity, fp_capacity)):
        raise ValueError('Expected nonnegative integer edit capacities')

    # The same prediction size must permit each supplied capacity and nonempty GT.
    lower = max(pair[0] for pair in (integer_iou, float_iou)) + fp_capacity
    upper = min(min(union - fn_capacity, union + intersection - 1)
                for intersection, union in (integer_iou, float_iou))
    if lower > upper:
        raise ValueError('Incompatible IoU counts and common capacities for one prediction')

    modes = {'fn_only': (fn_capacity, 0), 'fp_only': (0, fp_capacity), 'joint': (fn_capacity, fp_capacity)}
    numerator, denominator = THRESHOLD
    deficits = {name: max(0, numerator * union - denominator * intersection)
                for name, (intersection, union) in (('integer', integer_iou), ('float', float_iou))}
    deficits['dual'] = max(deficits.values())
    minimum_edits = {
        mode: {name: _minimum_edits(deficit, additions, removals) for name, deficit in deficits.items()}
        for mode, (additions, removals) in modes.items()
    }

    curve = []
    for budget in BUDGETS:
        point = {'budget': budget}
        for mode, (add_limit, remove_limit) in modes.items():
            additions = min(budget, add_limit)
            removals = min(budget - additions, remove_limit)
            point[mode] = dict(additions=additions, removals=removals,
                               integer_iou=(integer_iou[0] + additions, integer_iou[1] - removals),
                               float_iou=(float_iou[0] + additions, float_iou[1] - removals))
        curve.append(point)

    return dict(threshold=THRESHOLD, minimum_edits=minimum_edits, curve=curve)
