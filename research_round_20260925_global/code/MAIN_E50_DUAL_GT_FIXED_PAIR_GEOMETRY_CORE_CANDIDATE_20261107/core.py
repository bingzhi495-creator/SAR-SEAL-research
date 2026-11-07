"""Fixed-pair dual-GT geometry from five row-major binary masks."""


def _binary(mask):
    if type(mask) is not tuple or len(mask) != 2:
        raise ValueError("Each mask is a (shape, pixels) tuple")
    shape, pixels = mask
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)):
        raise ValueError("Each mask has positive integer (height, width)")
    if (type(pixels) is not bytes or len(pixels) != shape[0] * shape[1]
            or pixels.translate(None, b"\x00\x01")):
        raise ValueError("Each mask supplies one row-major 0/1 byte per pixel")
    return shape, int.from_bytes(pixels, "little")


def analyze_pair(integer_gt, floating_gt, native, plain, full):
    """Return exact counts for one already-fixed annotation and prediction query.

    Use regions.GT_union as the common denominator of all error counts and
    signed error differences. Each IoU retains its own prediction/GT union.
    Threshold crossings are +1 upward, -1 downward, and 0 for no crossing.
    """
    inputs = [_binary(mask) for mask in (integer_gt, floating_gt, native, plain, full)]
    shape = inputs[0][0]
    if any(candidate != shape for candidate, _ in inputs):
        raise ValueError("All masks must share the original grid shape")
    gi, gf, native_bits, plain_bits, full_bits = (bits for _, bits in inputs)
    if not gi or not gf:
        raise ValueError("Each GT mask must be nonempty")

    shared = gi & gf
    union = gi | gf
    integer_only = gi & ~gf
    float_only = gf & ~gi
    union_size = union.bit_count()
    regions = dict(shared_foreground=shared.bit_count(), shared_background=shape[0] * shape[1] - union_size,
                   integer_only=integer_only.bit_count(), float_only=float_only.bit_count(), GT_union=union_size)
    methods = {}
    for name, prediction in (("native", native_bits), ("PLAIN", plain_bits), ("FULL", full_bits)):
        errors = dict(shared_foreground_FN=(shared & ~prediction).bit_count(),
                      shared_background_FP=(prediction & ~union).bit_count(),
                      integer_band_FN=(integer_only & ~prediction).bit_count(),
                      integer_band_FP=(float_only & prediction).bit_count(),
                      float_band_FN=(float_only & ~prediction).bit_count(),
                      float_band_FP=(integer_only & prediction).bit_count())
        iou = {}
        for gt_name, gt in (("integer", gi), ("float", gf)):
            intersection = (prediction & gt).bit_count()
            pair_union = (prediction | gt).bit_count()
            iou[gt_name] = dict(intersection=intersection, union=pair_union,
                                at_least_0_75=4 * intersection >= 3 * pair_union)
        methods[name] = dict(errors=errors, iou=iou)
    contrasts = {}
    for target, reference in (("PLAIN", "native"), ("FULL", "native"), ("FULL", "PLAIN")):
        target_row, reference_row = methods[target], methods[reference]
        reference_errors = reference_row["errors"]
        contrasts[target + "_minus_" + reference] = dict(
            errors={key: value - reference_errors[key] for key, value in target_row["errors"].items()},
            iou75_crossing={gt_name: int(row["at_least_0_75"]) - int(reference_row["iou"][gt_name]["at_least_0_75"])
                            for gt_name, row in target_row["iou"].items()})
    return dict(shape=list(shape), regions=regions, methods=methods, contrasts=contrasts)
