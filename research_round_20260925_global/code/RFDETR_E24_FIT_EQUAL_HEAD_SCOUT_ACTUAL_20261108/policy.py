"""Locked FIT selector policies and matched-versus-retained accounting."""

import numpy as np

from solver import select_edits


THRESHOLDS = tuple(percent / 100 for percent in range(50, 101, 5))
BUDGET = 16
_COHORTS = ("matched", "ignored", "unmatched")
_COUNTS = (
    "rows", "support_rows", "empty_masks", "support_pixels",
    "budget_slots", "edited_rows", "edits",
)


def _integer_vector(value, name):
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return result


def _scores(scores, positions, *, probabilities):
    result = np.asarray(scores)
    if (
        result.shape != positions.shape
        or result.dtype.kind not in "iuf"
        or not np.all(np.isfinite(result))
        or (probabilities and np.any((result < 0) | (result > 1)))
    ):
        raise ValueError("scores must be finite and aligned; calibrated scores must be probabilities")
    return result


def _spans(match_rows, size):
    expected_start = 0
    for row in match_rows:
        start, stop = row["start"], row["stop"]
        if (
            type(start) is not int or type(stop) is not int
            or start != expected_start or not start <= stop <= size
        ):
            raise ValueError("match rows must exactly partition original candidate positions")
        yield row, start, stop
        expected_start = stop
    if expected_start != size:
        raise ValueError("match rows did not cover original candidate positions")


def select_image(match_rows, positions, scores, *, threshold):
    """Return original flat indices for every retained row, in input row order.

    Threshold None forces the same min(16, support) count on every row.
    Numeric thresholds belong to the locked probability grid. This function
    reads only row boundaries, never label status or GT identity.
    """
    positions = _integer_vector(positions, "positions")
    if threshold is not None:
        threshold = np.asarray(threshold)
        if threshold.ndim != 0 or threshold.dtype.kind not in "iuf":
            raise ValueError("threshold must belong to the locked probability grid")
        threshold = float(threshold)
        if threshold not in THRESHOLDS:
            raise ValueError("threshold must belong to the locked probability grid")
    scores = _scores(scores, positions, probabilities=threshold is not None)
    return tuple(
        select_edits(positions[start:stop], scores[start:stop], threshold=threshold, budget=BUDGET)
        for _, start, stop in _spans(match_rows, len(positions))
    )


def _selection_details(match_rows, positions, label_offsets, y, edits):
    positions = _integer_vector(positions, "positions")
    label_offsets = _integer_vector(label_offsets, "label_offsets")
    y = np.asarray(y)
    if (
        len(edits) != len(match_rows)
        or len(label_offsets) != len(match_rows) + 1
        or y.ndim != 1 or y.dtype != np.bool_
        or label_offsets[0] != 0 or label_offsets[-1] != len(y)
        or np.any(label_offsets[1:] < label_offsets[:-1])
    ):
        raise ValueError("rows, edits and Boolean label offsets must align")

    for i, (row, start, stop) in enumerate(_spans(match_rows, len(positions))):
        status = row["status"]
        if status not in _COHORTS or type(row["mask_empty"]) is not bool:
            raise ValueError("invalid retained-row status or mask_empty flag")
        candidates = positions[start:stop]
        selected = _integer_vector(edits[i], "edit indices")
        if (
            (len(candidates) and candidates[0] < 0)
            or np.any(candidates[1:] <= candidates[:-1])
            or len(selected) > BUDGET
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError("invalid candidate identity or edit count")
        local = np.searchsorted(candidates, selected)
        if np.any(local >= len(candidates)) or not np.array_equal(candidates[local], selected):
            raise ValueError("edits must belong to the row's original candidate support")

        first, last = label_offsets[i:i + 2]
        expected = len(candidates) if status == "matched" else 0
        if last - first != expected:
            raise ValueError("only matched rows have one flip label per candidate")
        correct = y[first:last][local] if status == "matched" else None
        yield row, len(candidates), local, correct


def _empty_cohorts():
    return {status: dict.fromkeys(_COUNTS, 0) for status in _COHORTS}


def _summary(cohorts, helpful, harmful):
    for counts in cohorts.values():
        counts["abstained_support_rows"] = counts["support_rows"] - counts["edited_rows"]
    matched_with_support = cohorts["matched"]["support_rows"]
    net = helpful - harmful
    return {
        "cohorts": cohorts,
        "total_edits": sum(counts["edits"] for counts in cohorts.values()),
        "matched_with_support": matched_with_support,
        "helpful_edits": helpful,
        "harmful_edits": harmful,
        "net_correct": net,
        "mean_net_correct": net / matched_with_support if matched_with_support else None,
    }


def account_image(match_rows, positions, label_offsets, y, edits):
    """Count edit behavior without treating unlabelled predictions as errors."""
    cohorts = _empty_cohorts()
    helpful = harmful = 0
    for row, support, local, correct in _selection_details(
        match_rows, positions, label_offsets, y, edits,
    ):
        counts = cohorts[row["status"]]
        counts["rows"] += 1
        counts["support_rows"] += int(support > 0)
        counts["empty_masks"] += int(row["mask_empty"])
        counts["support_pixels"] += support
        counts["budget_slots"] += min(BUDGET, support)
        counts["edited_rows"] += int(len(local) > 0)
        counts["edits"] += len(local)
        if correct is not None:
            correct_count = int(np.count_nonzero(correct))
            helpful += correct_count
            harmful += len(correct) - correct_count
    return _summary(cohorts, helpful, harmful)


def merge_accounts(accounts):
    """Combine per-image integer counts before computing instance means."""
    cohorts = _empty_cohorts()
    helpful = harmful = 0
    for account in accounts:
        helpful += account["helpful_edits"]
        harmful += account["harmful_edits"]
        for status, counts in cohorts.items():
            for field in _COUNTS:
                counts[field] += account["cohorts"][status][field]
    return _summary(cohorts, helpful, harmful)


def calibrate(images):
    """Choose one grid threshold from a caller-supplied calibration fold."""
    thresholds = np.asarray(THRESHOLDS)
    helpful = np.zeros(len(THRESHOLDS), dtype=np.int64)
    harmful = np.zeros_like(helpful)
    total_edits = np.zeros_like(helpful)
    matched_with_support = 0
    for image in images:
        rows = image["match_rows"]
        positions = _integer_vector(image["positions"], "positions")
        scores = _scores(image["scores"], positions, probabilities=True)
        forced = select_image(rows, positions, scores, threshold=None)
        for row, support, local, correct in _selection_details(
            rows, positions, image["label_offsets"], image["y"], forced,
        ):
            # Thresholding after top-K equals top-K after a monotone threshold.
            accepted = scores[row["start"] + local, None] > thresholds
            count = np.count_nonzero(accepted, axis=0)
            total_edits += count
            if correct is not None:
                matched_with_support += int(support > 0)
                good = np.count_nonzero(accepted[correct], axis=0)
                helpful += good
                harmful += count - good
    if not matched_with_support:
        raise ValueError("calibration requires matched instances with candidate support")

    curve = []
    for threshold, good, bad, edits in zip(THRESHOLDS, helpful, harmful, total_edits):
        net = int(good - bad)
        curve.append({
            "threshold": threshold,
            "helpful_edits": int(good),
            "harmful_edits": int(bad),
            "total_edits": int(edits),
            "net_correct": net,
            "mean_net_correct": net / matched_with_support,
        })
    chosen = max(curve, key=lambda row: (row["net_correct"], -row["total_edits"], row["threshold"]))
    return {
        "threshold": chosen["threshold"],
        "matched_with_support": matched_with_support,
        "curve": curve,
    }


def check_dev_gate(forced, metrics):
    """Apply the locked advancement rule to complete HOLDOUT summaries.

    AP and AP75 are COCO fractions in [0,1], with no display rounding.
    The caller is responsible for using exactly the same image/row population.
    """
    variants = ("uncertainty", "mean", "rank")
    signature = None
    for variant in variants:
        account = forced[variant]
        cohorts = account["cohorts"]
        population = tuple(
            tuple(cohorts[status][field] for field in _COUNTS[:5]) for status in _COHORTS
        )
        if signature is None:
            signature = population
        if population != signature or any(
            counts["edits"] != counts["budget_slots"]
            or counts["edited_rows"] != counts["support_rows"]
            for counts in cohorts.values()
        ):
            raise ValueError("forced policies must share a population and exhaust each row's budget")
        if (
            account["matched_with_support"] <= 0
            or account["matched_with_support"] != cohorts["matched"]["support_rows"]
            or account["helpful_edits"] + account["harmful_edits"] != cohorts["matched"]["edits"]
            or account["net_correct"] != account["helpful_edits"] - account["harmful_edits"]
        ):
            raise ValueError("matched forced counts are missing or inconsistent")

    ap = {}
    for variant in ("no_edit", *variants):
        for metric in ("AP", "AP75"):
            value = metrics[variant][metric]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not 0 <= value <= 1
            ):
                raise ValueError("metrics must be finite COCO AP fractions in [0,1]")
            ap[variant, metric] = float(value)

    # The matched denominators are equal, so compare exact integer net counts.
    best_control_ap = max(ap[variant, "AP"] for variant in ("no_edit", "uncertainty", "mean"))
    checks = {
        "forced_rank_beats_uncertainty": forced["rank"]["net_correct"] > forced["uncertainty"]["net_correct"],
        "forced_rank_beats_mean": forced["rank"]["net_correct"] > forced["mean"]["net_correct"],
        "mask_ap_gain_at_least_0_1_pp": ap["rank", "AP"] >= best_control_ap + 0.001,
        "mask_ap75_at_least_no_edit": ap["rank", "AP75"] >= ap["no_edit", "AP75"],
    }
    return {
        "eligible_for_dev": all(checks.values()),
        "checks": checks,
        "mask_ap_gain_over_strongest_control_pp": (ap["rank", "AP"] - best_control_ap) * 100,
        "mask_ap75_gain_over_no_edit_pp": (ap["rank", "AP75"] - ap["no_edit", "AP75"]) * 100,
    }
