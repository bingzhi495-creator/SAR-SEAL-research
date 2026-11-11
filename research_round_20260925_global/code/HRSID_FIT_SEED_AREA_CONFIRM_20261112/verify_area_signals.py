"""Independently verify closed three-seed FIT candidate and area-signal capture."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from types import ModuleType

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import expit


PROTOCOL = "4ac12dcc198bfafc424084f613e8ca6a69fdb4ecef3ee9b62e3d8bc0ab64b185"
RLE_HELPER = "2fb790afa6efc5852222cbfd046bb154380741add9ab33b0b7d96410de03fe20"
SEEDS = (5201, 5202, 5203)
DONORS = (5201, 5203)
GRID = 800
SUPPORT = np.ones((3, 3), dtype=bool)
STOCK_DELTA_ATOL = 1e-7



def require(valid, message):
    if not valid:
        raise ValueError(message)


def read_exact(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Unexpected closed SHA256: " + path.name)
    return raw


def import_rle_helper(path):
    module = ModuleType("_independent_closed_rle_helper")
    exec(compile(read_exact(path, RLE_HELPER), str(path), "exec"), module.__dict__)
    return module


def bilinear(plane, y, x, out_size):
    """Evaluate the mathematical half-pixel bilinear kernel independently."""
    sy = np.clip((y + .5) * plane.shape[0] / out_size - .5, 0, plane.shape[0] - 1)
    sx = np.clip((x + .5) * plane.shape[1] / out_size - .5, 0, plane.shape[1] - 1)
    y0, x0 = sy.astype(np.int64), sx.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, plane.shape[0] - 1), np.minimum(x0 + 1, plane.shape[1] - 1)
    wy, wx = sy - y0, sx - x0
    top = plane[y0, x0] * (1 - wx) + plane[y0, x1] * wx
    bottom = plane[y1, x0] * (1 - wx) + plane[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def band_points(indices, side=GRID):
    """Return absolute row/column coordinates of the clipped Chebyshev band."""
    y, x = indices % side, indices // side
    y0, x0 = max(0, int(y.min()) - 2), max(0, int(x.min()) - 2)
    y1, x1 = min(side, int(y.max()) + 3), min(side, int(x.max()) + 3)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[y - y0, x - x0] = True
    outer = binary_dilation(mask, structure=SUPPORT, iterations=2, border_value=0)
    core = binary_erosion(mask, structure=SUPPORT, iterations=2, border_value=0)
    by, bx = np.nonzero(outer & ~core)
    return by + y0, bx + x0, int(np.count_nonzero(mask[by, bx]))


def overlaps(a, b):
    left, area_left = a
    right, area_right = b
    overlap = (left @ right.T).toarray()
    return overlap / (area_left[:, None] + area_right[None, :] - overlap)


def representatives(overlap):
    remaining = list(range(overlap.shape[0]))
    keep = []
    while remaining:
        first = remaining.pop(0)
        keep.append(first)
        remaining = [j for j in remaining if overlap[first, j] < .90]
    return keep


def first_exclusive(eligible, roster, iou):
    available = {seed: set(roster[seed]) for seed in DONORS}
    pairs = {}
    for anchor in eligible:
        best = {
            seed: max(available[seed], key=lambda j: (iou[seed][anchor, j], -j), default=None)
            for seed in DONORS
        }
        if any(j is None or iou[seed][anchor, j] < .50 for seed, j in best.items()):
            continue
        pairs[anchor] = best
        for seed, j in best.items():
            available[seed].remove(j)
    return pairs


def array_from(path, block):
    logit = block["logit"]
    filename = logit["filename"]
    require(Path(filename).name == filename, "Unsafe logit path")
    raw = read_exact(path / filename, logit["sha256"])
    require(len(raw) == logit["bytes"], "Logit bytes mismatch")
    array = np.load(BytesIO(raw), allow_pickle=False)
    require(array.dtype == np.float32 and array.shape == (logit["rows"], 144, 144)
            and np.all(np.isfinite(array)), "Unexpected logit shape, type, or values")
    return array


def stock_soft(torch, logits, indices, y, x, inside_count):
    image = torch.nn.functional.interpolate(torch.from_numpy(logits[None, None]), size=(GRID, GRID),
                                            mode="bilinear", align_corners=False)[0, 0].numpy()
    stock_indices = np.flatnonzero((image > 0).ravel(order="F"))
    require(np.array_equal(stock_indices, indices), "Stock CPU interpolation sign does not reproduce RLE")
    return float(expit(image[y, x].astype(np.float64)).sum() - inside_count)


def inspect_signal(torch, helper, rows, anchor, matches, logits, signal):
    a = rows[5202][anchor]
    d = {seed: rows[seed][matches[seed]] for seed in DONORS}
    require(signal["rank"] == a["rank"] and
            signal["donor_ranks"] == {str(seed): d[seed]["rank"] for seed in DONORS},
            "Greedy association identity differs")
    area = a["area"]
    areas = [area, *(d[seed]["area"] for seed in DONORS)]
    cap = 2 * min(8, area // 10)
    mean_delta = sum(areas) / 3 - area
    median_delta = median(areas) - area
    size_delta = cap * math.log2(area / math.sqrt(32 * 1024)) / 2.5
    require(signal["median_delta"] == median_delta and signal["cap"] == cap,
            "Exact median or cap differs")
    for key, expected in (("mean_delta", mean_delta), ("size_delta", size_delta)):
        require(math.isclose(signal[key], expected, rel_tol=0, abs_tol=1e-12), key + " differs")

    require(a["logit_row"] is not None, "Eligible anchor has no native logit plane")
    logit = logits[a["logit_row"]]
    indices = helper.foreground_indices(a["rle"])
    y, x, inside_count = band_points(indices)
    exact = stock_soft(torch, logit, indices, y, x, inside_count)
    analytic = float(expit(bilinear(logit, y, x, GRID)).sum() - inside_count)
    stock_error = abs(signal["soft_delta"] - exact)
    analytic_error = abs(signal["soft_delta"] - analytic)
    require(stock_error <= STOCK_DELTA_ATOL, "Stored soft delta differs from stock CPU interpolation")

    return median_delta, stock_error, analytic_error, len(y)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("Closed local verification only")
    sys.addaudithook(offline)
    import torch
    torch.set_num_threads(1)

    helper = import_rle_helper(args.rle_helper)
    stage = args.closed_stage
    result = json.loads(read_exact(stage / "RESULT_SAFE.json", args.result_sha256))
    require(result["status"] == "HRSID_FIT_SEED_AREA_CONFIRM_CAPTURE_ACTUAL" and
            result["protocol_sha256"] == PROTOCOL and result["FIT"] is True and
            all(result[k] is False for k in ("DEV", "TEST", "GT_opened")),
            "Unexpected prospective scope or result protocol")
    arms = {int(arm["seed"]): arm for arm in result["arms"]}
    require(set(arms) == set(SEEDS) and len(result["arms"]) == 3, "Unexpected arm roster")
    by_seed = {s: helper.check_arm(stage / str(s), arms[s], s in DONORS) for s in SEEDS}
    coverage = result["coverage"]
    signal_sha = coverage["signals_sha256"]
    stored = helper.json_lines(read_exact(stage / "SIGNALS_PRIVATE.jsonl", signal_sha))
    require(len(stored) == 36 and sum(arm["model_forwards"] for arm in arms.values()) == 108,
            "Unexpected number of signal rows or stock forwards")

    counts = Counter({"add": 0, "drop": 0, "zero": 0})
    covered_groups, changed_groups = set(), set()
    eligible_count = representative_count = band_cells = 0
    stock_max = analytic_max = analytic_max_per_cell = 0.0
    for row_index, image_signals in enumerate(stored):
        blocks = {s: by_seed[s][row_index] for s in SEEDS}
        anchor_block = blocks[5202]
        for field in ("ordinal", "group", "basename"):
            require(image_signals[field] == anchor_block[field] and
                    all(b[field] == anchor_block[field] for b in blocks.values()),
                    "Seed or signal image identity differs")
        require(all(b["rgb_sha256"] == anchor_block["rgb_sha256"] for b in blocks.values()),
                "Seed image hashes differ")
        rows = {s: blocks[s]["candidates"] for s in SEEDS}
        matrices = {s: helper.mask_matrix(rows[s]) for s in SEEDS}
        roster = {s: representatives(overlaps(m, m)) for s, m in matrices.items()}
        iou = {s: overlaps(matrices[5202], matrices[s]) for s in DONORS}
        eligible = {i for i, a in enumerate(rows[5202]) if 32 <= a["area"] < 1024}
        representatives_eligible = [i for i in roster[5202] if i in eligible]
        eligible_count += len(eligible)
        representative_count += len(representatives_eligible)
        pairs = first_exclusive(representatives_eligible, roster, iou)
        signals = image_signals["signals"]
        require(len(signals) == len(pairs), "Greedy coverage count differs")
        anchor_logits = array_from(stage / "5202", anchor_block)
        for (anchor, pair), signal in zip(pairs.items(), signals, strict=True):
            delta, stock_error, analytic_error, size = inspect_signal(torch, helper, rows, anchor, pair,
                                                                     anchor_logits, signal)
            counts["add" if delta > 0 else "drop" if delta < 0 else "zero"] += 1
            covered_groups.add(anchor_block["group"])
            if delta:
                changed_groups.add(anchor_block["group"])
            stock_max = max(stock_max, stock_error)
            analytic_max = max(analytic_max, analytic_error)
            analytic_max_per_cell = max(analytic_max_per_cell, analytic_error / size)
            band_cells += size

    covered, nonzero = sum(counts.values()), counts["add"] + counts["drop"]
    passed = covered >= 60 and len(covered_groups) >= 20 and nonzero >= 30 and len(changed_groups) >= 15
    expected = {
        "eligible": eligible_count,
        "representative_eligible": representative_count,
        "covered_both": covered,
        "covered_groups": len(covered_groups),
        "nonzero_median_area_offset": nonzero,
        "nonzero_groups": len(changed_groups),
        "offsets": dict(counts),
        "gate": "COVERAGE_PASS" if passed else "COVERAGE_STOP",
    }
    require(all(coverage[k] == v for k, v in expected.items()), "Independent coverage summary differs")
    out = {
        "status": "HRSID_FIT_SEED_AREA_CONFIRM_INDEPENDENT_ACCEPTED",
        "result_sha256": args.result_sha256,
        "signals_sha256": signal_sha,
        "signals_verified": True,
        "FIT": True,
        "DEV": False,
        "TEST": False,
        "GT_opened": False,
        "images": len(stored),
        "band_cells_verified": band_cells,
        "stock_soft_max_abs_difference": stock_max,
        "stock_soft_atol": STOCK_DELTA_ATOL,
        "analytic_soft_max_abs_difference": analytic_max,
        "analytic_soft_max_abs_difference_per_band_cell": analytic_max_per_cell,
        "analytic_soft_is_descriptive": True,
        "coverage": expected,
    }
    payload = (json.dumps(out, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.safe_output.open("xb") as output:
        output.write(payload)
    print(payload.decode(), end="")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("closed-stage", "rle-helper", "safe-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    main(parser.parse_args())


