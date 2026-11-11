"""Score fixed FIT masks with one bounded, exact-real residual per routed cell."""
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import NamedTuple


BOUND = Fraction(2)
SIZE = 800
NODES = 144
IMAGES = 16
CELLS = 8
P316_SHA = "4b5574f3b32dba4c2963c57a4637eaa3695f1118a75cc3b7a36ec92bd0c60319"
P327_SOURCE_SHA = "b835b8d6634ba9cd9a4e6327645e76277944d814072593bb0822ce1ed7b9a8e9"
P327_RESULT_SHA = "e324a7daeab45b81761f2eb7c4a3ddc38ba324a89f78d1c5a932d9012c82bcd5"
CAPTURE_STATUS = "HRSID_FIXED_QUERY_CELL_DENSE_FIT_CAPTURE_ACTUAL"
RESULT_STATUS = "HRSID_FIXED_QUERY_CELL_RESIDUAL_FIT_ORACLE_COMPLETE"
HATS = {
    "midpoint": (Fraction(1, 2), Fraction(1, 2)),
    "x_quarter": (Fraction(1, 4), Fraction(1, 2)),
    "x_three_quarters": (Fraction(3, 4), Fraction(1, 2)),
    "y_quarter": (Fraction(1, 2), Fraction(1, 4)),
    "y_three_quarters": (Fraction(1, 2), Fraction(3, 4)),
}


class Edit(NamedTuple):
    delta_i: int
    delta_u: int
    flips: int
    coefficient: Fraction

    @property
    def margin(self):
        return 10 * self.delta_i - 9 * self.delta_u


def require(condition, message):
    if not condition:
        raise ValueError(message)


def optimize_cell(logits, hat, target):
    """Maximize exact .90 margin under a single bounded sign-preserving basis."""
    transitions = {}
    base_i = base_u = current_i = current_u = 0
    for logit, weight, foreground in zip(logits, hat, target, strict=True):
        require(weight > 0, "Routed pixel must be strictly inside the hat support")
        if logit > 0:
            base_i += int(foreground)
            base_u += int(not foreground)
        threshold = -logit / weight
        if threshold < -BOUND:
            current_i += int(foreground)
            current_u += int(not foreground)
        elif threshold < BOUND:
            positive, negative = transitions.get(threshold, (0, 0))
            transitions[threshold] = (positive + int(foreground), negative + int(not foreground))

    def result(coefficient):
        return Edit(current_i - base_i, current_u - base_u,
                    abs(current_i + current_u - base_i - base_u), coefficient)

    def order(edit):
        return (edit.margin, -edit.flips, -abs(edit.coefficient), -edit.coefficient)

    best = max(Edit(0, 0, 0, Fraction(0)), result(-BOUND), key=order)
    levels = sorted(transitions) + [BOUND]
    for level, next_level in zip(levels[:-1], levels[1:], strict=True):
        positive, negative = transitions[level]
        current_i += positive
        current_u += negative
        # Strictly above this breakpoint, strictly below the next one.
        edit = result((level + next_level) / 2)
        best = max(best, edit, key=order)
    return best


def hat_value(position, peak):
    return min(position / peak, (1 - position) / (1 - peak))


def axis_support():
    support = [[] for _ in range(NODES - 1)]
    for pixel in range(SIZE):
        cell, numerator = divmod((2 * pixel + 1) * NODES - SIZE, 2 * SIZE)
        if 0 <= cell < NODES - 1 and numerator:
            support[cell].append((pixel, Fraction(numerator, 2 * SIZE)))
    return support


def cell_samples(plane, cy, cx, support):
    v00, v01, v10, v11 = (
        Fraction(float(plane[y, x]))
        for y, x in ((cy, cx), (cy, cx + 1), (cy + 1, cx), (cy + 1, cx + 1))
    )
    for y, fy in support[cy]:
        left, right = (1 - fy) * v00 + fy * v10, (1 - fy) * v01 + fy * v11
        for x, fx in support[cx]:
            yield y, x, fx, fy, (1 - fx) * left + fx * right


def read_pinned(path, expected, limit):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and 0 < path.stat().st_size <= limit, "Canonical bounded closed input required")
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Closed input digest differs")
    return data


def load_source(path, pin, name):
    data = read_pinned(path, pin, 1 << 20)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def verify_capture(a):
    # Read the original process receipt before any scientific input.
    terminal = json.loads(read_pinned(a.capture_terminal, a.capture_terminal_sha256, 1 << 20))
    require(terminal["original_os_wait"] is True and type(terminal["original_os_exit"]) is int
            and terminal["original_os_exit"] == 0 and terminal["process_group_closed"] is True
            and terminal["stop_reason"] is None and terminal["worker_sha256"] == a.capture_source_sha256,
            "The original dense capture and its job must have closed cleanly")
    require(terminal["result"]["sha256"] == a.capture_report_sha256
            and a.capture_terminal.parent == a.capture_report.parent,
            "Original wait must bind the closed capture report")
    report_data = read_pinned(a.capture_report, a.capture_report_sha256, 1 << 20)
    require(len(report_data) == terminal["result"]["bytes"], "Original result size differs")
    report = json.loads(report_data)
    expected = {
        "status": CAPTURE_STATUS, "FIT": True, "DEV": False, "TEST": False, "GT_opened": False, "AP": False,
        "optimizer_updates": 0, "model_forward_calls": IMAGES, "images": IMAGES,
        "candidate_rows_reproduced": 1391, "prior_fp32_band_points_reproduced": 9429,
        "selected_native_masks_reproduced": IMAGES, "cells_per_candidate_limit": CELLS,
        "dense_grid_bytes": IMAGES * NODES**2 * 4,
    }
    require(all(type(report[key]) is type(value) and report[key] == value for key, value in expected.items()),
            "Exact frozen FIT dense capture scope required")
    route_data = read_pinned(a.capture_report.parent / "ROUTE_PRIVATE.json", report["route_sha256"], 4 << 20)
    dense = read_pinned(a.capture_report.parent / "DENSE_LOGITS_PRIVATE.bin", report["dense_grid_sha256"], 2 << 20)
    routes = json.loads(route_data)
    require(type(routes) is list and len(routes) == IMAGES and len(dense) == report["dense_grid_bytes"],
            "Exact FIT route and plane census required")
    return report, routes, dense


def route_cells(np, plane):
    corners = np.stack((plane[:-1, :-1], plane[1:, :-1], plane[:-1, 1:], plane[1:, 1:])).astype(np.float64)
    y, x = np.nonzero((corners.min(axis=0) <= 0) & (corners.max(axis=0) > 0))
    risk = np.abs(corners.mean(axis=0)[y, x])
    positions = np.lexsort((x, y, risk))[:CELLS]
    return [[int(y[i]), int(x[i])] for i in positions]


def score_pair(plane, cells, original_mask, target_mask):
    support = axis_support()
    metrics = {key: [0, 0, 0, 0] for key in HATS}
    pixels = 0
    for cy, cx in cells:
        samples = list(cell_samples(plane, cy, cx, support))
        pixels += len(samples)
        require(all(bool(original_mask[y, x]) == (z > 0) for y, x, _, _, z in samples),
                "Exact-real baseline sign differs from stock FP32 inside a routed cell")
        if target_mask is None:
            continue
        logits = [row[4] for row in samples]
        target = [bool(target_mask[y, x]) for y, x, _, _, _ in samples]
        for method, (peak_x, peak_y) in HATS.items():
            weights = [hat_value(fx, peak_x) * hat_value(fy, peak_y) for _, _, fx, fy, _ in samples]
            result = optimize_cell(logits, weights, target)
            values = result.delta_i, result.delta_u, result.flips, int(result.flips > 0)
            metrics[method] = [a + b for a, b in zip(metrics[method], values, strict=True)]
    return metrics, pixels


def score(a):
    capture, routes, dense = verify_capture(a)
    import numpy as np

    p327 = load_source(a.p327_source, P327_SOURCE_SHA, "_p362_p327_score")
    p317 = p327.load_p317(a.p317_source)
    prior = json.loads(read_pinned(a.p327_result, P327_RESULT_SHA, 1 << 20))
    require(prior["status"] == "P327_HRSID_FIT114_TILED_BAND_FIXED_MATCH_SCORE_COMPLETE"
            and prior["p316_candidate_stream_sha256"] == P316_SHA,
            "Exact original fixed-match P327 reference required")

    cohort, index = p317.load_scope(a.bundle)
    selection = p327.select_candidates(p317, a.p316_candidates, P316_SHA, cohort)
    # Validate labels-free routing and bind it to the original selected candidates.
    work = []
    for i, (route, row) in enumerate(zip(routes, selection, strict=True)):
        payload = dense[i * NODES**2 * 4:(i + 1) * NODES**2 * 4]
        require(set(route) == {"image_id", "candidate", "cells", "plane_sha256"}
                and route["image_id"] == row["image_id"] and route["candidate"] == row["candidate"]
                and sha256(payload).hexdigest() == route["plane_sha256"],
                "Closed route, original P327 candidate or plane identity differs")
        plane = np.frombuffer(payload, dtype="<f4").reshape(NODES, NODES)
        require(np.isfinite(plane).all() and route_cells(np, plane) == route["cells"],
                "Exact GT-free bounded routing must reproduce")
        work.append((row, plane, route["cells"]))
    require(sum(len(cells) for _, _, cells in work) == capture["total_routed_cells"],
            "Routed cell count differs from original capture")

    truth = p317.ground_truth(index, cohort)
    methods = ("raw", *HATS)
    metrics = {method: Counter() for method in methods}
    paired = {method: Counter() for method in HATS if method != "midpoint"}
    census = Counter(images=IMAGES, routed_cells=capture["total_routed_cells"])
    for row, plane, cells in work:
        selected = row["index"]
        predictions = row["predictions"]
        target_rles = truth[row["image_id"]]
        matches = p317.original_matches(predictions, target_rles)
        target = matches.get(selected)
        pred_mask = p317.codec.decode(predictions[selected]).astype(bool, copy=False)
        target_mask = p317.codec.decode(target).astype(bool, copy=False) if target is not None else None
        edits, pixels = score_pair(plane, cells, pred_mask, target_mask)
        census.update(original_all_candidates=len(predictions), original_all_fixed_matches=len(matches),
                      eligible_gt=len(target_rles), selected_fixed_matches=int(target is not None),
                      selected_unmatched=int(target is None), original_grid_pixels_routed=pixels)
        if target is None:
            continue
        intersection = int((pred_mask & target_mask).sum())
        union = int((pred_mask | target_mask).sum())
        require(2 * intersection >= union > 0, "Original fixed P327 match must satisfy IoU .50")
        old_margin = 10 * intersection - 9 * union
        census.update(original_at_75=int(4 * intersection >= 3 * union),
                      original_at_90=int(old_margin >= 0), original_grid_pixels_routed_matched=pixels)
        candidate_margins = {}
        for method, (di, du, flips, cells_changed) in {"raw": [0, 0, 0, 0], **edits}.items():
            new_i, new_u = intersection + di, union + du
            margin = 10 * new_i - 9 * new_u
            require(0 <= new_i <= new_u and new_u > 0 and margin >= old_margin,
                    "Exact optimal edited IoU counts must remain feasible")
            candidate_margins[method] = margin
            metrics[method].update(
                fixed_iou_sum=new_i / new_u, margin_gain_90=margin - old_margin,
                recover_90=int(old_margin < 0 <= margin), lose_90=int(old_margin >= 0 > margin),
                at_90=int(margin >= 0), pixels_flipped=flips, active_cells=cells_changed,
                candidates_changed=int(flips > 0),
            )
        for method, counts in paired.items():
            change = candidate_margins["midpoint"] - candidate_margins[method]
            counts.update(midpoint_margin_win=int(change > 0), midpoint_margin_tie=int(change == 0),
                          midpoint_margin_loss=int(change < 0), midpoint_minus_control_margin=change,
                          midpoint_only_at_90=int(candidate_margins["midpoint"] >= 0 > candidate_margins[method]),
                          control_only_at_90=int(candidate_margins[method] >= 0 > candidate_margins["midpoint"]))

    for key in ("images", "eligible_gt", "original_all_candidates", "original_all_fixed_matches",
                "selected_fixed_matches", "selected_unmatched", "original_at_75", "original_at_90"):
        require(census[key] == prior["census"][key], f"Original P327 cohort or matching differs: {key}")
    for values in metrics.values():
        values["mean_fixed_iou"] = values.pop("fixed_iou_sum") / census["selected_fixed_matches"]
    require(metrics["raw"]["mean_fixed_iou"] == prior["primary"]["raw"]["mean_fixed_iou"],
            "Original P327 fixed-IoU readout differs")
    return dict(
        status=RESULT_STATUS, FIT=True, DEV=False, TEST=False, AP=False, training=False, optimizer_updates=0,
        model_forward_calls=0, gt_assisted=True, gt_assisted_unmatched_actions=0,
        census=dict(census), arms={method: dict(values) for method, values in metrics.items()},
        midpoint_vs_control={method: dict(counts) for method, counts in paired.items()},
        policy=dict(cells_per_candidate_limit=CELLS, coefficient_bounds=[-2, 2],
                    primary_threshold="0.90", fixed_original_match_iou="0.50",
                    hat_centers_xy={name: [str(x), str(y)] for name, (x, y) in HATS.items()},
                    same_gt_free_routed_cells=True, exact_real_breakpoints=True,
                    exact_real_baseline_sign_on_routed_pixels_verified_against_stock_FP32=True,
                    optimized_coefficients_replayed_in_FP32=False, output_masks_saved=False,
                    output_scores_or_matching_changed=False),
        provenance=dict(capture_report_sha256=a.capture_report_sha256, capture_terminal_sha256=a.capture_terminal_sha256,
                        capture_worker_sha256=a.capture_source_sha256, dense_grid_sha256=capture["dense_grid_sha256"],
                        route_sha256=capture["route_sha256"], p316_candidate_stream_sha256=P316_SHA,
                        p327_result_sha256=P327_RESULT_SHA, fit_bundle_sha256=p317.BUNDLE_SHA),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("capture-report", "capture-terminal", "p327-source", "p327-result", "p317-source",
                 "bundle", "p316-candidates", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("capture-report-sha256", "capture-terminal-sha256", "capture-source-sha256"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1",
            "Closed FIT-only CPU score required")
    require(a.output.is_absolute() and a.output.parent.resolve(strict=True) == a.output.parent
            and not a.output.exists() and not a.output.is_symlink(), "Fresh canonical aggregate output required")

    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Offline closed scoring only")

    sys.addaudithook(offline)
    payload = (json.dumps(score(a), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with a.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=RESULT_STATUS, result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
